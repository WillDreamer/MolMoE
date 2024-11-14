from torch import nn
import torch
from deepspeed.moe.layer import MoE
from helper_utils import NestedTensor 
from model.configuration import ProjectorConfig


class Type1MoE(nn.Module):
    def __init__(self, hidden_size: int = 300):
        super().__init__()
        module = nn.Linear(hidden_size, hidden_size)
        self.moe_layer = MoE(
            hidden_size=hidden_size,
            expert=module,
            num_experts=3,
            ep_size=1,
            k=1,
            capacity_factor=1,
            eval_capacity_factor=2,
            min_capacity=0,
            use_residual=False
        )
        
    def forward(self, features: torch.Tensor):
        # features: B, level, N, d
        B, L, N, D = features.shape
        features = features.permute(0, 2, 1, 3).reshape(B*N, L, D)
        features = self.moe_layer(features)
        
        return features[0].reshape(B, N, L, D).permute(0, 2, 1, 3), features[1]
    
    
class Type2MoE(nn.Module):
    def __init__(self, hidden_size: int = 300):
        super().__init__()
        module = nn.Linear(hidden_size, hidden_size)
        self.moe_layer = MoE(
            hidden_size=hidden_size,
            expert=module,
            num_experts=3,
            ep_size=1,
            k=1,
            capacity_factor=1,
            eval_capacity_factor=2,
            min_capacity=0,
            use_residual=False
        )
        
    def forward(self, features: torch.Tensor):
        # features: B, N, d
        features = self.moe_layer(features)
        
        return features[0], features[1]
    
    
class GAPHead(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size, use_mlp) -> None:
        super().__init__()
        if not use_mlp:
            self.projector = nn.Linear(mm_hidden_size, hidden_size)
        else:
            self.projector == nn.Sequential(
                nn.Linear(mm_hidden_size, mm_hidden_size*2),
                nn.GELU(),
                nn.Linear(mm_hidden_size*2, hidden_size)
                )
        
    @staticmethod
    def parallel_mean_pooling(features: NestedTensor):
        if len(features.tensors.shape) == 4:
            # We are in B, L, N, D setting
            tensors = features.tensors.flatten(1, 2)  # B, tokens, d
            masks = features.mask.flatten(1, 2)       # B, tokens
        else:
            # We are in B, N, D setting
            tensors, masks = features.decompose() # B, tokens, d
        
        not_masks = ~masks
        
        # Sum and count the non-masked elements
        sum_features = tensors.sum(dim=1)  # B, d
        count_nonmasked = not_masks.sum(dim=1, keepdim=True)  # B, 1
        
        # Avoid division by zero
        count_nonmasked = count_nonmasked.clamp(min=1)
        effective_features = sum_features / count_nonmasked
    
        return effective_features
    
    def forward(self, x: NestedTensor):
        x = NestedTensor(self.projector(x.tensors), x.mask)
        x = self.parallel_mean_pooling(x)
        
        return x
    

class WeightedHead(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size, use_mlp):
        super().__init__()
        if not use_mlp:
            self.projector = nn.Linear(mm_hidden_size, hidden_size)
        else:
            self.projector == nn.Sequential(
                nn.Linear(mm_hidden_size, mm_hidden_size*2),
                nn.GELU(),
                nn.Linear(mm_hidden_size*2, hidden_size)
                )
        
        self.combiner = nn.Linear(mm_hidden_size, 3)
    
    @staticmethod 
    def pool_one_level(x: torch.Tensor, mask: torch.Tensor):
        batch = []
        for feat, m in zip(x, mask):
            batch.append(feat[m].mean(0))
            
        return torch.stack(batch, dim=0)
    
    def pool_levels(self, x: NestedTensor):
        # B, N, D
        node, motif, graph = x.tensors.permute(1, 0, 2, 3).unbind(0)
        # B, N
        not_mask = ~x.mask
        node_m, motif_m, graph_m = not_mask.permute(1, 0, 2).unbind(0)
        
        graph_pooled = self.pool_one_level(graph, graph_m)
        motif_pooled = self.pool_one_level(motif, motif_m)
        node_pooled = self.pool_one_level(node, node_m)
        
        return graph_pooled, motif_pooled, node_pooled
        
                
    def forward(self, graph_feature, x):
        # B, level, N, D
        x = NestedTensor(self.projector(x.tensors), x.mask)
        weights = self.combiner(graph_feature.mean(tuple(range(1, graph_feature.dim() - 1)))).softmax(-1)
        node, motif, graph = weights.unbind(0)
        
        graph_token, motif_token, node_token = self.pool_levels(x)
        
        feature = graph * graph_token + motif * motif_token + node * node_token
        
        return feature
        
        
class Type2MoEProjector(nn.Module):
    def __init__(self, mm_hidden_size: int, hidden_size: int, use_mlp: bool, *args, **kwargs):
        """Type 2 MoE projector, tokens are flattened through L*N, requires data
        shape of B, N, D. This is also called a token-level projector. Type 2 projector
        only supports GAP head

        Args:
            mm_hidden_size (int): hidden size of graph encoder
            hidden_size (int): hidden size of LLM
            use_mlp (bool): whether to use MLP to finally project the dimension
        """
        super().__init__()
        self.moe_module = Type2MoE(mm_hidden_size)
        self.data_shape = "bnd"
        
        self.head = GAPHead(mm_hidden_size, hidden_size, use_mlp)
        
    def forward(self, x: NestedTensor):
        outputs = self.moe_module(x.tensors)
        features, aux_loss = NestedTensor(outputs[0], x.mask), outputs[1]
        features = self.head(features)
        
        return features, aux_loss
        
        
class Type1MoEProjector(nn.Module):
    def __init__(self, mm_hidden_size: int, hidden_size: int, use_mlp: bool, head_type: str, *args, **kwargs):
        """Type 1 MoE projector, where features are padded to B, L, N, D,
        the MoE module selects three levels, this is also called a hierarchical projector, 
        Type 1 projector supports GAP head and Weighted head

        Args:
            mm_hidden_size (int): hidden size of graph encoder
            hidden_size (int): hidden size of LLM
            use_mlp (bool): whether to use MLP to finally project the dimension
            head_type (str): type of the head, choose from ('gap', 'weighted')

        Raises:
            NotImplementedError: Must choose head type from ('gap', 'weighted')
        """
        super().__init__()
        self.moe_module = Type1MoE(mm_hidden_size)
        self.data_shape = "blnd"
        
        if head_type == "gap":
            self.head = GAPHead(mm_hidden_size, hidden_size, use_mlp)
        elif head_type == "weighted":
            self.head = WeightedHead(mm_hidden_size, hidden_size, use_mlp)
        else:
            raise NotImplementedError(f"Cannot build head type {head_type}")
        
    def forward(self, features: NestedTensor):
        outputs = self.moe_module(features.tensors)
        features, aux_loss = NestedTensor(outputs[0], features.mask), outputs[1]
        features = self.head(features)
        
        return features, aux_loss
        
        
class MultiLevelLinear(nn.Module):
    def __init__(self, mm_hidden_size: int, hidden_size: int, use_mlp: bool, *args, **kwargs):
        """Multilevel linear projector, requires data shape of B, N, D, and uses
        GAP head. 

        Args:
            mm_hidden_size (int): hidden size of graph encoder
            hidden_size (int): hidden size of LLM
            use_mlp (bool): whether to use MLP to finally project the dimension
        """
        self.head = GAPHead(mm_hidden_size, hidden_size, use_mlp)
        self.data_shape = "bnd"
        
    def forward(self, x: NestedTensor):
        x = self.head(x)
        
        return x
    
class SingleLevelLinear(nn.Module):
    def __init__(self, mm_hidden_size: int, hidden_size: int, use_mlp: bool, level: str, *args, **kwargs):
        """Single level feature + linear projector, specify a level of feature

        Args:
            mm_hidden_size (int): hidden size of graph encoder
            hidden_size (int): hidden size of LLM
            use_mlp (bool): whether to use MLP to finally project the dimension
            level (str): choose from ('node', 'motif', 'graph')
        """
        super().__init__()
        self.head = GAPHead(mm_hidden_size, hidden_size, use_mlp)
        self.level = level
        assert level in ("node", "motif", "graph"), f"Unkown level {level}"
        
    def forward(self, x: NestedTensor):
        x = self.head(x)
        
        return x
    
class NaiveLinear(nn.Module):
    def __init__(self, mm_hidden_size: int, hidden_size: int, *args, **kwargs):
        super().__init__()
        self.weight = nn.Linear(mm_hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor):
        x = self.weight(x)
        
        return x
    
        
GET_PROJECTOR = {
    "multilevel_type1": Type1MoEProjector,
    "multilevel_type2": Type2MoEProjector,
    "multilevel_linear": MultiLevelLinear,
    "singlelevel_linear": SingleLevelLinear,
    "naive_linear": NaiveLinear
}

def create_projector(graph_config, text_config, projector_config):
    arguments = {
        "mm_hidden_size": graph_config.hidden_size,
        "hidden_size": text_config.hidden_size,
        "use_mlp": projector_config.use_mlp,
        "level": projector_config.level,
        "head_type": projector_config.head_type
    }
    
    return GET_PROJECTOR[projector_config.projector_type](**arguments)