from arguments import TrainingArguments, ModelArguments, DataArguments
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoConfig 
from transformers import AutoTokenizer
from model.configuration import GraphConfig, GraphLlavaConfig, MoEConfig, ProjectorConfig
from model.modelling_llava import GraphLlavaForConditionalGeneration
from helper_utils import no_init_weights
from data_pipeline import conversation_lib
import torch
from torch import nn
import os
from model.projector_factory import Type1MoEProjector, Type2MoEProjector
from deepspeed.moe.layer import MoE


def find_all_linear_names(model: nn.Module) -> list:
    """
    Find all modules that is nn.Linear
    Args:
        model (nn.Module): The model used to find linear modules

    Returns:
        list: list of linear modules
    """
    cls = torch.nn.Linear  # we are going to find all nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():  # iterate all modules
        if isinstance(module, cls):  # If it's nn.Linear
            names = name.split('.')  # split the name, name rule: xx.xx
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')  # exclude lm_head
    return list(lora_module_names)


def find_linear_without_moe(model: nn.Module) -> list:
    """Find all linear modules except for graph_tower, mm_projector and lm_head

    Args:
        model (nn.Module): Model

    Returns:
        list: list of found modules
    """
    cls = torch.nn.Linear
    lora_module_names = list()
    for name, module in model.named_modules():
        if ("graph_tower" not in name) and ("mm_projector" not in name) and ("lm_head" not in name) and isinstance(module, cls) and ("deepspeed_moe" not in name):
            lora_module_names.append(name)
            
    return lora_module_names


def create_selfies_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    # 1. Init all configs
    graph_config = GraphConfig(
        model_name=model_args.graph_tower,
        encoder_num_layer=model_args.gin_num_layers,
        hidden_size=model_args.gin_hidden_dim,
        encoder_JK='last',
        encoder_drop_ratio=model_args.drop_ratio,
        encoder_gnn_type='gin'
    )
    # default override to torch.bfloat16 for flash attention
    text_config = AutoConfig.from_pretrained(
        model_args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        )
    projector_config = ProjectorConfig(
        projector_type=model_args.mm_projector_type,
        moe_type=model_args.projector_moe_type,
        head_type=model_args.head_type,
        use_mlp=model_args.use_mlp,
        level=model_args.level,
        use_head_weight=model_args.use_head_weight,
        num_query=model_args.num_query,
        num_heads=model_args.num_heads
    )
    config = GraphLlavaConfig(
        graph_config, 
        text_config, 
        projector_config=projector_config,
        moe_enable=model_args.moe_enable,
        language_backbone_name=model_args.language_backbone_name,
        projector_aux_loss_coeff=model_args.projector_aux_loss_coeff
        )
    config.use_cache = False
    text_config.use_cache = False
    # 2. Instantiate tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    
    text_config.moe_enable = config.moe_enable = False
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    # 3. Load pre-trained LLM, projector and GNN
    model.load_language_model()
    model.rand_init_projector()
    model.load_graph(model_args.graph_init_checkpoint)
    
    # 4. Apply LoRA
    # import lora related functions
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(  # initailize a LoRA Config
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_linear_without_moe(model),  # add lora to all modules that is nn.Linear
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if torch.cuda.is_available():
        print("Moving to cuda for faster warping...")
        model.to("cuda")
        
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)  # add lora according to lora_config
    training_args.lora_enable = True
    model.to("cpu")
    
    # 5. set parameters, since LoRA freeze all parameters, we activate projector here
    model.mm_projector.requires_grad_(True)
    
    return tokenizer, model


def create_stage1_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Stage 1 model:
    
    - 🔥 mm_projector
    - 🥶 graph tower
    - 🥶 LLM

    Args:
        model_args (ModelArguments): Model arguments
        training_args (TrainingArguments): Training arguments

    Returns:
        tuple[PreTrainedTokenizer, PreTrainedModel]: tokenizer for the specific model and the model itself
    """
    # 1. Init all configs
    graph_config = GraphConfig(
        model_name=model_args.graph_tower,
        encoder_num_layer=model_args.gin_num_layers,
        hidden_size=model_args.gin_hidden_dim,
        encoder_JK='last',
        encoder_drop_ratio=model_args.drop_ratio,
        encoder_gnn_type='gin'
    )
    text_config = AutoConfig.from_pretrained(
        model_args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        )
    projector_config = ProjectorConfig(
        projector_type=model_args.mm_projector_type,
        moe_type=model_args.projector_moe_type,
        head_type=model_args.head_type,
        use_mlp=model_args.use_mlp,
        level=model_args.level,
        use_head_weight=model_args.use_head_weight,
        num_query=model_args.num_query,
        num_heads=model_args.num_heads
    )
    config = GraphLlavaConfig(
        graph_config, 
        text_config,
        moe_config=None,
        projector_config=projector_config,
        moe_enable=False,
        language_backbone_name=model_args.language_backbone_name,
        projector_aux_loss_coeff=model_args.projector_aux_loss_coeff
        )
    config.use_cache = False
    text_config.use_cache = False
    # 2. Instantiate tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)

    text_config.moe_enable = config.moe_enable = False
    
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    # 3. Load pre-trained LLM, random init projector and load graph ckpt
    model.load_language_model()
    model.rand_init_projector()
    model.load_graph(model_args.graph_init_checkpoint)
    
    # 4. Set parameters that will be trained
    model.requires_grad_(False)
    model.mm_projector.requires_grad_(True)
    
    return tokenizer, model


def create_stage2_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Stage 2 model
    
    - 🔥 mm_projector
    - 🔥 LoRA
    - 🥶 graph tower
    - 🥶 LLM

    Args:
        model_args (ModelArguments): Model arguments
        training_args (TrainingArguments): Training arguments

    Returns:
        tuple[PreTrainedTokenizer, PreTrainedModel]: tokenizer for the specific model and the model itself
    """
    # 1. Init all configs
    graph_config = GraphConfig(
        model_name=model_args.graph_tower,
        encoder_num_layer=model_args.gin_num_layers,
        hidden_size=model_args.gin_hidden_dim,
        encoder_JK='last',
        encoder_drop_ratio=model_args.drop_ratio,
        encoder_gnn_type='gin'
    )
    # default override to torch.bfloat16 for flash attention
    text_config = AutoConfig.from_pretrained(
        model_args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        )
    projector_config = ProjectorConfig(
        projector_type=model_args.mm_projector_type,
        moe_type=model_args.projector_moe_type,
        head_type=model_args.head_type,
        use_mlp=model_args.use_mlp,
        level=model_args.level,
        # use_head_weight=model_args.use_head_weight,
        # num_query=model_args.num_query,
        # num_heads=model_args.num_heads
    )
    config = GraphLlavaConfig(
        graph_config, 
        text_config, 
        projector_config=projector_config,
        moe_enable=model_args.moe_enable,
        language_backbone_name=model_args.language_backbone_name,
        projector_aux_loss_coeff=model_args.projector_aux_loss_coeff
        )
    config.use_cache = False
    text_config.use_cache = False
    # 2. Instantiate tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    
    text_config.moe_enable = config.moe_enable = False
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    # 3. Load pre-trained LLM, projector and GNN
    model.load_language_model()
    model.load_projector(model_args.pretrain_mm_mlp_adapter)
    model.load_graph(model_args.graph_init_checkpoint)
    
    # 4. Apply LoRA
    # import lora related functions
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(  # initailize a LoRA Config
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        use_rslora=True,
        target_modules=find_linear_without_moe(model),  # add lora to all modules that is nn.Linear
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if torch.cuda.is_available():
        print("Moving to cuda for faster warping...")
        model.to("cuda")
        
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)  # add lora according to lora_config
    training_args.lora_enable = True
    model.to("cpu")
    
    # 5. set parameters, since LoRA freeze all parameters, we activate projector here
    model.mm_projector.requires_grad_(True)
    
    return tokenizer, model


def create_lora2lora_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    tokenizer, model = load_lora_model(
        training_args.stage2_model,
        model_args.base_model,
        model_args.graph_init_checkpoint,
        use_flash_attn=True
    )
    
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(  # initailize a LoRA Config
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_linear_without_moe(model),  # add lora to all modules that is nn.Linear
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if torch.cuda.is_available():
        print("Moving to cuda for faster warping...")
        model.to("cuda")
        
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)  # add lora according to lora_config
    training_args.lora_enable = True
    model.to("cpu")
    
    # 5. set parameters, since LoRA freeze all parameters, we activate projector here
    model.mm_projector.requires_grad_(True)
    
    return tokenizer, model


def create_stage3_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Stage 3 model
    
    - 🔥 MoE layers
    - 🥶 mm_projector
    - 🥶 graph tower
    - 🥶 rest of the LLM

    Args:
        model_args (ModelArguments): Model arguments
        training_args (TrainingArguments): Training arguments

    Returns:
        tuple[PreTrainedTokenizer, PreTrainedModel]: tokenizer for the specific model and the model itself
    """
    tokenizer, model = load_lora_model(
        training_args.stage2_model,
        model_args.base_model,
        model_args.graph_init_checkpoint,
        use_flash_attn=True
    )
    
    moe_config = MoEConfig(
        moe_mode = model_args.moe_mode,
        moe_layers_idx=model_args.moe_layers_idx,
        ep_size=model_args.ep_size,
        num_experts=model_args.num_experts,
        top_k_experts=model_args.top_k_experts,
        capacity_factor=model_args.capacity_factor,
        eval_capacity_factor=model_args.eval_capacity_factor,
        min_capacity=model_args.min_capacity,
        use_residual=model_args.use_residual,
        router_aux_loss_coef=model_args.router_aux_loss_coef,
        enable_lottery_trick=model_args.enable_gradient_mask,
        pruning_percent=model_args.pruning_percent
    )
    
    model.config.moe_enable = True,
    model.config.text_config.moe_enable = True
    model.config.moe_config = moe_config
    training_args.moe_enable = model_args.moe_enable = True
    
    if torch.cuda.is_available():
        print("Moving to CUDA for faster creation")
        model.to("cuda")
        
    model.replace_mlp_with_moe()
    model.to("cpu")
    torch.cuda.empty_cache()
    
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "mm_projector" not in name and "deepspeed_moe" in name:
            param.requires_grad = True
    # model.mm_projector.requires_grad_(True)
    
    print("Adding additional input gradients")
    model.language_model.enable_input_require_grads()
    
    return tokenizer, model
    
    
def create_moe_lora_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    # 1. Init all configs
    graph_config = GraphConfig(
        model_name=model_args.graph_tower,
        encoder_num_layer=model_args.gin_num_layers,
        hidden_size=model_args.gin_hidden_dim,
        encoder_JK='last',
        encoder_drop_ratio=model_args.drop_ratio,
        encoder_gnn_type='gin'
    )
    # default override to torch.bfloat16 for flash attention
    text_config = AutoConfig.from_pretrained(
        model_args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        )
    projector_config = ProjectorConfig(
        projector_type=model_args.mm_projector_type,
        moe_type=model_args.projector_moe_type,
        head_type=model_args.head_type,
        use_mlp=model_args.use_mlp,
        level=model_args.level,
        # use_head_weight=model_args.use_head_weight,
        # num_query=model_args.num_query,
        # num_heads=model_args.num_heads
    )
    config = GraphLlavaConfig(
        graph_config, 
        text_config, 
        projector_config=projector_config,
        moe_enable=model_args.moe_enable,
        language_backbone_name=model_args.language_backbone_name,
        projector_aux_loss_coeff=model_args.projector_aux_loss_coeff
        )
    config.use_cache = False
    text_config.use_cache = False
    # 2. Instantiate tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    # 3. Load pre-trained LLM, projector and GNN
    model.load_language_model()
    model.load_projector(model_args.pretrain_mm_mlp_adapter)
    model.load_graph(model_args.graph_init_checkpoint)
    
    # 4. create moe model
    moe_config = MoEConfig(
        moe_mode = model_args.moe_mode,
        moe_layers_idx=model_args.moe_layers_idx,
        ep_size=model_args.ep_size,
        num_experts=model_args.num_experts,
        top_k_experts=model_args.top_k_experts,
        capacity_factor=model_args.capacity_factor,
        eval_capacity_factor=model_args.eval_capacity_factor,
        min_capacity=model_args.min_capacity,
        use_residual=model_args.use_residual,
        router_aux_loss_coef=model_args.router_aux_loss_coef,
        enable_lottery_trick=model_args.enable_gradient_mask,
        pruning_percent=model_args.pruning_percent
    )
    
    model.config.moe_enable = True,
    model.config.text_config.moe_enable = True
    model.config.moe_config = moe_config
    training_args.moe_enable = model_args.moe_enable = True
    
    if torch.cuda.is_available():
        print("Moving to CUDA for faster creation")
        model.to("cuda")
        
    model.replace_mlp_with_moe()
    model.to("cpu")
    torch.cuda.empty_cache()
    
    # 5. Apply LoRA
    # import lora related functions
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(  # initailize a LoRA Config
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_linear_without_moe(model),  # do not add lora to any MoE layers, but any layer except that
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if torch.cuda.is_available():
        print("Moving to cuda for faster warping...")
        model.to("cuda")
        
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)  # add lora according to lora_config
    training_args.lora_enable = True
    model.to("cpu")
    
    # 5. set parameters, since LoRA freeze all parameters, we activate projector here
    model.mm_projector.requires_grad_(True)
    # 6. set all moe layers active
    for name, module in model.named_modules():
        if isinstance(module, MoE):
            module.requires_grad_(True)
    
    return tokenizer, model
    
    

def load_pretrain_model(
    model_path: str,
    language_backbone: str,
    graph_path: str,
    use_flash_attn: bool
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load a model from stage 1 pre-training

    Args:
        model_path (str): path to the pre-training folder(the one contains mm_projector.bin)
        language_backbone (str): path to the language backbone(e.g., phi-3 mini, llama-3.2)
        graph_path (str): path to graph checkpoint(e.g. himol_encoder.pth)

    Returns:
        tuple[PreTrainedTokenizer, PreTrainedModel]: tokenizer for the specific model and the model itself
    """
    config = GraphLlavaConfig.from_pretrained(model_path)
    if use_flash_attn:
        config._attn_implementation = "flash_attention_2"
        config.text_config._attn_implementation = "flash_attention_2"
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    model.load_language_model()
    tokenizer = AutoTokenizer.from_pretrained(language_backbone)
    model.load_graph(graph_path)
    files = os.listdir(model_path)
    if "mm_projector.bin" in files:
        print("Detected mm_projector.bin in", model_path)
        model.load_projector(os.path.join(model_path, "mm_projector.bin"))
        
    return tokenizer, model

def load_lora_model(
    model_path: str,
    language_backbone: str,
    graph_path: str,
    use_flash_attn: bool
) -> tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]:
    """Load a LoRA fine-tuned model from stage 2

    ## Args:
        model_path (str): path to the lora fine-tuned folder(the one contains adapter_model.safetensors)
        language_backbone (str): path to the language backbone(e.g., phi-3 mini, llama-3.2)
        graph_path (str): path to graph checkpoint(e.g. himol_encoder.pth)
        use_flash_attn (bool): Whether to use flash attention

    ## Raises:
        NotImplementedError: If no non_lora_trainables.bin exists in the model_path, something happend to the saving
        or the parameter activation in the training, please check!

    ## Returns:
        tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]: tokenizer for the specific model and the model itself
    """
    # 1. Get config from the model_path folder
    config = GraphLlavaConfig.from_pretrained(model_path)
    if use_flash_attn:
        config._attn_implementation = "flash_attention_2"
        config.text_config._attn_implementation = "flash_attention_2"
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    tokenizer = AutoTokenizer.from_pretrained(language_backbone)
    # 2. Load language model, graph ckpt
    model.load_language_model()
    model.load_graph(graph_path)
    
    # 3. Load mm_projector
    print('Loading additional LLaVA weights...')
    # Read and process non-lora-trainables, specifically, mm projector
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(
            os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        print("Non-lora trainables:", non_lora_trainables.keys())
    else:
        print("No Non-lora weights detected!")
        raise NotImplementedError
        
    non_lora_trainables = {k.split("mm_projector.")[1]: v for k, v in non_lora_trainables.items()}
        
    model.mm_projector.load_state_dict(non_lora_trainables)
    
    # 4. Load LoRA weights and merge LoRA
    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print("Moving to CUDA")
    model.to(torch.device("cuda"))
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print("Moving back to CPU")
    model.to(torch.device("cpu"))
    print('Model is loaded...')
    torch.cuda.empty_cache()
    
    # 5. If MoE projector is used, we intialize our model with deepspeed engine
    if isinstance(model.mm_projector, Type1MoEProjector) or isinstance(model.mm_projector, Type2MoEProjector):
        import deepspeed
        deepspeed.init_distributed(dist_backend='nccl')
        # Initialize the DeepSpeed-Inference engine
        ds_engine = deepspeed.init_inference(model,
                                                # mp_size=2,
                                                # dtype=torch.half,
                                                checkpoint=None,
                                                replace_with_kernel_inject=False)
        model = ds_engine.module
        
    return tokenizer, model


def load_lora2lora_model(
    model_path: str,
    language_backbone: str,
    graph_path: str,
    use_flash_attn: bool
) -> tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]:
    base_lora_path, this_lora_path = model_path.split("+")
    tokenizer, model = load_lora_model(base_lora_path, language_backbone, graph_path, use_flash_attn)
    
    print('Loading additional LLaVA weights...')
    # Read and process non-lora-trainables, specifically, mm projector
    if os.path.exists(os.path.join(this_lora_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(
            os.path.join(this_lora_path, 'non_lora_trainables.bin'), map_location='cpu')
        print("Non-lora trainables:", non_lora_trainables.keys())
    else:
        print("No Non-lora weights detected!")
        raise NotImplementedError
        
    non_lora_trainables = {k.split("mm_projector.")[1]: v for k, v in non_lora_trainables.items()}
    model.mm_projector.load_state_dict(non_lora_trainables)
    
    from peft import PeftModel
    print('Loading LoRA weights...')
    # Load the second lora adapter
    model = PeftModel.from_pretrained(model, this_lora_path)
    print("Moving to CUDA")
    model.to(torch.device("cuda"))
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print("Moving back to CPU")
    model.to(torch.device("cpu"))
    print('Model is loaded...')
    torch.cuda.empty_cache()
    
    return tokenizer, model


def load_full_model(
    model_path: str,
    language_backbone: str,
    graph_path: str,
    use_flash_attn: bool
):
    
    model = GraphLlavaForConditionalGeneration.from_pretrained(
        model_path,
        _attn_implementation="flash_attention_2" if use_flash_attn else None
        )
    tokenizer = AutoTokenizer.from_pretrained(language_backbone)
    import deepspeed
    deepspeed.init_distributed(dist_backend='nccl')
    # Initialize the DeepSpeed-Inference engine
    ds_engine = deepspeed.init_inference(model,
                            # mp_size=2,
                            # dtype=torch.half,
                            checkpoint=None,
                            replace_with_kernel_inject=False)
    model = ds_engine.module
    return tokenizer,model

    
MODEL_STAGE_MAP = {
    "stage1": create_stage1_model,
    "stage2": create_stage2_model,
    "stage3": create_stage3_model,
    "selfies_pretrain": create_selfies_model,
    "lora2lora": create_lora2lora_model,
    "moe+lora": create_moe_lora_model,
}

def create_model(
    model_args: ModelArguments, 
    data_args: DataArguments, 
    training_args: TrainingArguments
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    # 1. Create tokenizer, model
    tokenizer, model = MODEL_STAGE_MAP[training_args.training_recipe](model_args, training_args)
    
    # 2. Set correct conversation template
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    if tokenizer.pad_token is None and model_args.version == "llama3":
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    elif model_args.version == "tinyllama":
        tokenizer.pad_token = tokenizer.unk_token
    print("Using conversation template of", model_args.version)
    print("Conversation template:", conversation_lib.default_conversation)
    
    # 3. Align arguments
    data_args.graph_tower = model_args.graph_tower
    training_args.moe_enable = model_args.moe_enable
    training_args.projector_aux_coeff = model_args.projector_aux_loss_coeff
    training_args.router_aux_coeff = model_args.router_aux_loss_coef
    if "type" in model_args.mm_projector_type:
        training_args.moe_enable = True
        
    return tokenizer, model