from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.cache_utils import Cache
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from torch import nn
from model.configuration import GraphLlavaConfig, GraphConfig
from model.graph_modal.himol import HiMolGraphTowerV2, GraphFeature, HiMolGraphTower
from model.graph_modal.graphmvp import GraphMVP
from model.custom_llama import CustomLlamaForCausalLM
import torch
from deepspeed.moe.layer import MoE
from typing import Optional, Union, Tuple, List
from transformers.generation import GenerationMixin
import logging
from model.custom_phi3 import CustomPhi3ForCausalLM
from torch_geometric.data import Batch
from helper_utils import NestedTensor
from dataclasses import dataclass
from copy import deepcopy
from model.custom_llama import MoELlamaDecoderLayerForward, MoELlamaModelForward
from model.custom_phi3 import MoEPhiModel_forward, MoEPhiDecoderLayer_forward
from model.lottery_wrap import create_lottery
from model.projector_factory import create_projector
from model.graph_modal.moleculestm import GNN_graphpred

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


MODEL_CLS_MAP = {
    "phi3-mini": CustomPhi3ForCausalLM,
    "Phi3-medium": CustomPhi3ForCausalLM,
    "llama3-1b": CustomLlamaForCausalLM,
    "tinyllama": CustomLlamaForCausalLM
}


@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    moe_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


graph_tower_map = {
    "himol": HiMolGraphTowerV2,
    "graphmvp": GraphMVP,
    "moleculestm": GNN_graphpred
}
class GraphTower(nn.Module):
    def __init__(self, config: GraphConfig):
        super().__init__()
        self.gnn = graph_tower_map.get(config.model_name, None)(
            config.encoder_num_layer,
            config.hidden_size,
            config.encoder_JK,
            config.encoder_drop_ratio,
            config.encoder_gnn_type
        )
        if self.gnn is None:
            raise ValueError(f"Doesn't support graph tower type {config.model_name}")
        
        self.apply(self._init_weight)
        
    def _init_weight(self, module):
        pass
        
    def forward(self, *args, **kwargs):
        return self.gnn(*args, **kwargs)
    
    def encode_mol(self, *args, **kwargs):
        # legacy graph ecnoder compatibility
        return self.gnn.encode_mol(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        self.gnn.load_state_dict(*args, **kwargs)


class LlavaPreTrainedModel(PreTrainedModel):
    config_class = GraphLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa
    
    
class GraphLlavaForConditionalGeneration(LlavaPreTrainedModel, GenerationMixin):
    def __init__(self, config: GraphLlavaConfig):
        """The class for the Graph LLaVA model

        Args:
            config (GraphLlavaConfig): Config of GraphLLaVA
        """
        super().__init__(config)
        # GRAPH TOWER ==============
        self.graph_tower = GraphTower(config.graph_config)
        
        # PROJECTOR ===================
        # self.mm_projector = pooled_moe(config.graph_config.hidden_size, config.text_config.hidden_size,"type1")
        self.mm_projector = create_projector(config.graph_config, config.text_config, config.projector_config)
        
        # LLM =================================
        self.vocab_size = config.text_config.vocab_size
        lm_cls = MODEL_CLS_MAP[config.language_backbone_name]
        print("Equipped with", config.language_backbone_name)
        self.language_model = lm_cls._from_config(
            config.text_config,
            attn_implementation=config.text_config._attn_implementation, 
            torch_dtype=self.config.text_config.torch_dtype
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.config.hidden_size = self.config.text_config.hidden_size
        
        # self.apply(self._init_weight)
        
    def load_graph(self, path2graph: str) -> None:
        """Load graph ckpt to the model's graph tower

        Args:
            path2graph (str): path to the graph ckpt, e.g.(himol_encoder.pth)
        """
        print("Loading graph ckpt from", path2graph)
        self.graph_tower.gnn.load_weight(path2graph)
        
    def load_projector(self, path2proj: str):
        """Load projector weight

        Args:
            path2proj (str): path to the projector weight
        """
        print("Lodaing projector from", path2proj)
        state_dict = torch.load(path2proj)
        print("Projector State Dict:", state_dict.keys())
        state_dict = {k.split("mm_projector.")[1]: v for k, v in state_dict.items()}
        self.mm_projector.load_state_dict(state_dict)
        
    def load_language_model(self):
        """Load LLM, the LLM type & path is specified in the text config
        """
        print("Loading LLM ckpt from", self.config.text_config._name_or_path)
        self.language_model = self.language_model.from_pretrained(
            self.config.text_config._name_or_path,
            config=self.config.text_config,
            torch_dtype=self.config.text_config.torch_dtype,
            attn_implementation=self.config.text_config._attn_implementation
            )
        
    def rand_init_projector(self):
        # Let's create the projector again, this can perfectly simulate
        # the enviroment without _no_init_weight context manager
        self.mm_projector = create_projector(self.config.graph_config, self.config.text_config, self.config.projector_config)
        for name, module in self.mm_projector.named_modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        
    def replace_mlp_with_moe(self):
        FORWARD_MAP = {
            "llama3-1b": [MoELlamaDecoderLayerForward, MoELlamaModelForward],
            "phi3-mini": [MoEPhiDecoderLayer_forward, MoEPhiModel_forward]
        }
        if self.config.moe_config.train_modules is not None and len(self.config.moe_config.train_modules) > 0:
            for n, p in self.named_parameters():
                if any(name in n for name in self.config.moe_config.train_modules):
                    continue
                else:
                    p.requires_grad = False

        num_layers = self.config.text_config.num_hidden_layers

        moe_layers_idx = self.config.moe_config.moe_layers_idx
        if self.config.moe_config.moe_layers_idx is not None:
            self.config.moe_config.moe_mode = 'custom'
            assert len(self.config.moe_config.moe_layers_idx) <= num_layers
            assert max(self.config.moe_config.moe_layers_idx) < num_layers
            assert min(self.config.moe_config.moe_layers_idx) >= 0
        else:
            if self.config.moe_config.moe_mode == "first_half":
                moe_layers_idx = list(range(0, num_layers // 2))
            elif self.config.moe_config.moe_mode == "second_half":
                moe_layers_idx = list(range(num_layers // 2, num_layers))
            elif self.config.moe_config.moe_mode == "sparse":
                moe_layers_idx = list(range(num_layers))[::4]
            elif self.config.moe_config.moe_mode == "dense":
                moe_layers_idx = list(range(num_layers))
            elif self.config.moe_config.moe_mode == "second_quarter":
                moe_layers_idx = list(range(num_layers - (num_layers // 4), num_layers))
            else:
                raise NotImplementedError(
                    f'Only support ["first_half", "second_half", "second_quarter", "sparse", "dense"], but found {self.config.moe_config.moe_mode}')

        self.config.moe_config.moe_layers_idx = moe_layers_idx
        if type(self.config.moe_config.num_experts) == int:
            self.config.moe_config.num_experts = [self.config.moe_config.num_experts] * len(moe_layers_idx)
        assert len(self.config.moe_config.num_experts) == len(moe_layers_idx)
        
        cnt = 0
        for num_experts, layer_num in zip(self.config.moe_config.num_experts, moe_layers_idx):
            pretrained_state_dict = self.language_model.model.layers[layer_num].mlp.state_dict()
            expert = deepcopy(self.language_model.model.layers[layer_num].mlp)
            del(self.language_model.model.layers[layer_num].mlp)
            moe_layer = MoE(
                self.config.text_config.hidden_size,
                expert=expert,
                num_experts=num_experts,
                ep_size=self.config.moe_config.ep_size,
                k=self.config.moe_config.top_k_experts,
                capacity_factor=self.config.moe_config.capacity_factor,
                eval_capacity_factor=self.config.moe_config.eval_capacity_factor,
                min_capacity=self.config.moe_config.min_capacity,
                use_residual=self.config.moe_config.use_residual,
            )
            # moe_layer.requires_grad_(True)
            if self.config.moe_config.enable_lottery_trick:
                if cnt > 3:  # NOTE: for our poor VRAM, let's add only on last 4 layers...
                    print("Gradient mask enabled, prune", self.config.moe_config.pruning_percent, "percent")
                    # in-place modification
                    create_lottery(moe_layer, self.config.moe_config.pruning_percent)
                self.language_model.model.layers[layer_num].mlp = moe_layer
            else:
                self.language_model.model.layers[layer_num].mlp = moe_layer
                
            for e in self.language_model.model.layers[layer_num].mlp.deepspeed_moe.experts.deepspeed_experts:  # check weight
                loaded_state_dict = e.state_dict()
                assert all([torch.allclose(pretrained_state_dict[k], v) for k, v in loaded_state_dict.items()])
                assert all([torch.allclose(loaded_state_dict[k], v) for k, v in pretrained_state_dict.items()])
                
            cnt += 1
                
        # ipdb.set_trace()
        print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe_config.num_experts, moe_layers_idx)])
        
        MoEDecoderForward, MoEModelForward = FORWARD_MAP[self.config.language_backbone_name]
        
        for m in self.language_model.model.layers:
            m.forward = MoEDecoderForward(m)
        print(f'replace DecoderLayer.forward to MoEDecoderLayer.forward')
        self.language_model.model.forward = MoEModelForward(self.language_model.model)
        print(f'replace Model.forward to MoEModel.forward')

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds
    
    def encode_mol_v2(self, mol, batch_size, device) -> torch.Tensor:
        # 1. Encode feature with graph encoder
        def himol_encode(mol, batch_size):
            return self.graph_tower.float()(Batch.from_data_list(mol), batch_size)
            
        def mvp_stm_encode(mol, batch_size):
            _, h_node = self.graph_tower.float().encode_mol(mol, proj=False, return_node_feats=True)
            return h_node
            
        encode_map = {
            "himol": himol_encode,
            "graph_mvp": mvp_stm_encode,
            "moleculestm": mvp_stm_encode
        }
        h_node = encode_map[self.config.graph_config.model_name](mol, batch_size)
        # 2. Convert feature to desired shape if it's GraphFeature(only HiMol encoder has)
        # also get the dtype
        if isinstance(h_node, GraphFeature):
            if getattr(self.mm_projector, "data_shape", None) is not None:
                h_node = h_node.pad_to(self.mm_projector.data_shape)
            elif getattr(self.mm_projector, "level", None) is not None:
                h_node = h_node.get_level(self.mm_projector.level, pad=True)
            else:
                raise ValueError(f"Projector type {type(self.mm_projector)} is not built for GraphFeature")
            dtype = h_node.dtype
        else:
            dtype = h_node.dtype
            
        # 3. Judge and convert h_node to float32
        if dtype == torch.bfloat16:
            if isinstance(h_node, NestedTensor):
                h_node = NestedTensor(h_node.tensors.float(), h_node.mask)
            else:
                h_node = h_node.float()
                
        # 4. Judge and feed h_node to projector
        aux_loss = None
        graph_features = None
        if "type" in self.config.projector_config.projector_type:
            assert isinstance(h_node, NestedTensor), "MoE projector requires NestedTensor!"
            graph_features, aux_loss = self.mm_projector.float()(h_node.to(device))
        else:
            graph_features = self.mm_projector.float()(h_node.to(device))
            
        if isinstance(graph_features, list):
            graph_features = [f.to(dtype) for f in graph_features]
        else:
            graph_features = graph_features.to(dtype)
            
        return graph_features, aux_loss
    
    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.Tensor, 
        past_key_values: list[torch.FloatTensor], 
        labels: torch.LongTensor, 
        graphs: torch.FloatTensor
    ):
        # <image>
        # [[0.2, 0.3, 0.4], [0.2, 0.3, 0.4], [0.1, 0.3, 0.4]]
        # graph token: [[0.3, 0.1, 0.5]]
        # [[0.3, 0.1, 0.5], [0.2 ,0.3, 0.4], [0.2, 0.3, 0.4], [0.1, 0.3, 0.4]]
        graph_tower = self.graph_tower  # get graph encoder
        """
        NOTE
        Borrowed the code from Transformers legacy processing
        Checked the code, it can generate correct attention mask in forward prediction
        """
        if graph_tower is None or graphs is None or input_ids.shape[1] == 1:
            # In case input_ids.shape[1] == 1 & graphs==None & past_key_values != None, we are in the case of
            # generation with cache
            if past_key_values is not None and graph_tower is not None and graphs is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                # Structure of kv cache:
                # tuple[tuple[torch.Tensor]]
                # first tuple: layers
                # second tuple: K, V
                # torch.Tensor: B, num_head, cache_length, head_dim
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                
            return input_ids, attention_mask, None, past_key_values, None, labels, None
        
        if type(graphs) is list:
            # skip `to_data_list`
            assert len(graphs) == input_ids.shape[0]
        else:
            graphs = graphs.to_data_list()
        
        # encode graphs and append to graph features to a list
        batch_size = input_ids.shape[0]
        projector_aux_loss = None
        if self.config.graph_config.model_name == "himol":
            graph_features, projector_aux_loss = self.encode_mol_v2(graphs, batch_size, input_ids.device)
        else:
            # legacy GNNs need to encode graphs one by one along batch dimension
            graph_features = []
            for graph in graphs:
                graph_feat, _ = self.encode_mol_v2(graph, batch_size, input_ids.device)
                graph_features.append(graph_feat)
            
        # print("Original Inside graph feature====", graph_features[0])
            
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_graph_idx = 0
        # input_ids: [B, L]
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == self.config.image_token_index).sum() == 0:  # no image token in current ids
                cur_input_embeds = self.language_model.model.embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (0. * self.mm_projector(graph_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_graph_idx += 1
                continue
            # find the position of image token(they use image token to mark the graph)
            graph_token_indices = torch.where(cur_input_ids == self.config.image_token_index)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while graph_token_indices.numel() > 0:
                cur_graph_features = graph_features[cur_graph_idx]
                if self.config.graph_config.model_name == "himol" and len(cur_graph_features.shape) == 1:
                    cur_graph_features = cur_graph_features.unsqueeze(0)
                    
                graph_token_start = graph_token_indices[0]  # obtain the position from tuple
                # we are tuning mm projector and using image start end token
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    # 0:graph_token-1, embed 0, 1, ..., graph_token_pos-2
                    cur_new_input_embeds.append(self.language_model.model.embed_tokens(cur_input_ids[:graph_token_start-1]).detach())
                    # graph_token_pos-1:graph_token_pos, embed graph_token_pos-1
                    cur_new_input_embeds.append(self.language_model.model.embed_tokens(cur_input_ids[graph_token_start-1:graph_token_start]))
                    # graph features
                    cur_new_input_embeds.append(cur_graph_features)
                    # graph_token_pos+1:graph_token_pos+2, embed graph_token_pos+1
                    cur_new_input_embeds.append(self.language_model.model.embed_tokens(cur_input_ids[graph_token_start+1:graph_token_start+2]))
                    # total embedding of: 0,1,...,graph_token_pos-2, graph_token_pos-1, graph_features, graph_token_pos+1
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[graph_token_start:graph_token_start+1])
                        cur_labels = cur_labels[graph_token_start+2:]
                else:  # Maybe not using im start end? True <image>
                    # 0:graph_token_pos, embed 0,1,..., graph_token_pos-1
                    cur_new_input_embeds.append(self.language_model.model.embed_tokens(cur_input_ids[:graph_token_start]))
                    # append graph features
                    cur_new_input_embeds.append(cur_graph_features.type_as(cur_new_input_embeds[0]))
                    # total embedding of 0,1,...,graph_token_pos-1, graph_features
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        # For checking anything beyond <image>
                        cur_labels = cur_labels[graph_token_start+1:]
                        
                cur_graph_idx += 1
                # skipping input ids before <image> ================
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[graph_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[graph_token_start+1:]
                    
                # ==================================================
                # after replacement, there won't be any graph tokens
                graph_token_indices = torch.where(cur_input_ids == self.config.image_token_index)[0]
                
            # Check if we have something left after <image>
            if cur_input_ids.numel() > 0:
                # embed them 
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.language_model.model.embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.language_model.model.embed_tokens(cur_input_ids))
                    
                if labels is not None:
                    cur_new_labels.append(cur_labels)
                    
            # send them to device
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # concat to a complete embedded sentence
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            # append to new input embeds
            new_input_embeds.append(cur_new_input_embeds)
            # concat labels to a compelte sentence and append to new labels
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
                
        # handle varied length in this batch
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            # find maximum length
            max_len = max(x.shape[0] for x in new_input_embeds)
            
            # list to store length aligned embeds
            new_input_embeds_align = []
            # iterate batches
            for cur_new_embed in new_input_embeds:
                # Pad zeros to the sentence whose length is smaller than maxlen
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed, 
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), # shape of (maxlen - l, c)
                            dtype=cur_new_embed.dtype, 
                            device=cur_new_embed.device)
                    ), 
                    dim=0)
                new_input_embeds_align.append(cur_new_embed)
                
            # stack them to a complete batch
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
            
            # Pad labels with IGNORE_INDEX
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label, 
                            torch.full(
                                (max_len - cur_new_label.shape[0],), 
                                IGNORE_INDEX, 
                                dtype=cur_new_label.dtype, 
                                device=cur_new_label.device)
                        ), 
                        dim=0)
                    new_labels_align.append(cur_new_label)
                    
                new_labels = torch.stack(new_labels_align, dim=0)
                
            # recalibrate attention masks
            if attention_mask is not None:
                if new_labels is not None:
                    new_attention_mask = []
                    _new_labels = new_labels  # _new_labels is new_labels, which means cur_new_labels is cur_new_labels_align
                    for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                        new_attn_mask_pad_left = torch.full(  # usually this will extend HF attention mask for our new graph token
                            (cur_new_labels.shape[0] - labels.shape[1],), 
                            True, 
                            dtype=attention_mask.dtype, 
                            device=attention_mask.device)
                        new_attn_mask_pad_right = torch.full(  # under what circumstance this thing is greater than 0?
                            (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), 
                            False, 
                            dtype=attention_mask.dtype, 
                            device=attention_mask.device)
                        cur_new_attention_mask = torch.cat(
                            (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), 
                            dim=0)
                        new_attention_mask.append(cur_new_attention_mask)
                    attention_mask = torch.stack(new_attention_mask, dim=0)
                    assert attention_mask.shape == new_labels.shape
                else:
                    attention_mask = torch.cat(
                        (  # usually this will extend HF attention mask for our new graph token
                            torch.full(  # B x new_token
                                (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), 
                                True, 
                                dtype=attention_mask.dtype, 
                                device=attention_mask.device
                            ), 
                            attention_mask
                        ),
                        dim=1,
                    )
                    assert attention_mask.shape == new_input_embeds.shape[:2]
        else:  # no varied length
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2], f"attention_mask.shape: {attention_mask.shape} != new_input_embeds.shape[:2]: {new_input_embeds.shape[:2]}"
                
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_((attention_mask == 0), 1)
        
        return None, attention_mask, position_ids, past_key_values, new_input_embeds, new_labels, projector_aux_loss
    
    def moe_forward(
        self,
        labels, 
        logits, 
        outputs, 
        return_dict,
        projector_aux_loss
    ):
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            model_loss = loss.item()
            moe_loss, moe_losses = None, []
            if len(outputs[-1]) > 0 and isinstance(outputs[-1], list):
                moe_loss_list = outputs[-1]
                for moe_loss in moe_loss_list:
                    if moe_loss is not None:
                        moe_losses.append(moe_loss)
                moe_loss = self.config.moe_config.router_aux_loss_coef * sum(moe_losses)
                if labels is not None:
                    if self.config.projector_aux_loss_coeff is not None and projector_aux_loss is not None:
                        loss += self.config.projector_aux_loss_coeff * projector_aux_loss
                    else:
                        projector_aux_loss = None
                        
                    loss += moe_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            output = (moe_loss,) + output if moe_loss is not None else output
            return (loss,) + output if loss is not None else output
        
        return_class = MoECausalLMOutputWithPast(
            loss=loss,
            moe_loss=sum(moe_losses) if len(moe_losses) > 0 else None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            moe_loss_list=getattr(outputs, "moe_loss_list", None),
        )
        return_class.projector_aux_loss = projector_aux_loss
        return_class.proj_aux_coeff = self.config.projector_aux_loss_coeff
        return_class.model_loss = model_loss
        return_class.router_aux_coeff = self.config.moe_config.router_aux_loss_coef
        return_class.labels = labels  # for spin training
        
        return return_class
        
    def vanilla_forward(
        self,
        labels, 
        logits, 
        return_dict, 
        outputs,
        projector_aux_loss,
        is_spin_training
        ):
        loss = None
        model_loss = None
        if labels is not None and not is_spin_training:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            model_loss = loss.item()
            if projector_aux_loss is not None and self.config.projector_aux_loss_coeff is not None:
                loss += self.config.projector_aux_loss_coeff * projector_aux_loss
            else:
                projector_aux_loss = None
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            output = (projector_aux_loss,) + output if projector_aux_loss is not None else output
            return (loss,) + output if loss is not None else output
        
        return_class = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return_class.projector_aux_loss = projector_aux_loss
        return_class.proj_aux_coeff = self.config.projector_aux_loss_coeff
        return_class.model_loss = model_loss
        return_class.labels = labels
        
        return return_class
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graphs: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        is_spin_training = False
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, projector_aux_loss = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, graphs)

        outputs = self.language_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.language_model.lm_head(hidden_states)
        logits = logits.float()
        if self.config.moe_enable:
            return self.moe_forward(labels, logits, outputs, return_dict, projector_aux_loss)
        else:
            return self.vanilla_forward(labels, logits, return_dict, outputs, projector_aux_loss, is_spin_training)
        
    """
    NOTE
    Borrowed code from transformers
    """
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graphs": kwargs.get("graphs", None),
            }
        )
        
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
    

