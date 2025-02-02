import torch
import torch.nn as nn
import re
from deepspeed.moe.layer import MoE
from enum import Enum, auto



class NestedTensor(object):
    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor):
        self.tensors: torch.Tensor = tensors
        self.mask: torch.Tensor = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask
    
    @property
    def shape(self):
        return self.tensors.shape

    def __repr__(self):
        return str(self.tensors)

GRAPHLEVEL = {
    "NODE": 0,
    "MOTIF": 1,
    "GRAPH": 2
}

GETPOOLER = {
    "max": torch.max,
    "mean": torch.mean
}

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    

def masked_pooling(features: NestedTensor, pooling_method: str="mean"):
    tensors = features.tensors.flatten(1, 2) 
    masks = features.mask.flatten(1, 2) 
    
    not_masks = ~masks
    
    pooler = GETPOOLER[pooling_method]
    effective_features = []
    for feature, mask in zip(tensors, not_masks):
        
        effective_features.append(pooler(feature[mask], dim=0))
        
    effective_features = torch.stack(effective_features, dim=0)
    
    return effective_features

def parallel_mean_pooling(features: NestedTensor, pooling_method: str = "mean"):
    tensors = features.tensors.flatten(1, 2)  
    masks = features.mask.flatten(1, 2)       
    
    not_masks = ~masks
    
    
    sum_features = tensors.sum(dim=1)  
    count_nonmasked = not_masks.sum(dim=1, keepdim=True)  
    
    
    count_nonmasked = count_nonmasked.clamp(min=1)
    
    
    if pooling_method == "mean":
        effective_features = sum_features / count_nonmasked
    
    return effective_features


def level_pooling(features: NestedTensor, pooling_method: str="mean"):
    tensors, masks = features.decompose()
    pooler = GETPOOLER[pooling_method]
    
    
    
    not_mask = ~masks
    effective_features = []
    for feature, mask in zip(tensors, not_mask):
        node_feature = pooler(feature[0][mask[0]], dim=0)
        motif_feature = pooler(feature[1][mask[1]], dim=0)
        if torch.isnan(motif_feature).any():
            motif_feature = torch.zeros_like(node_feature)
        graph_feature = pooler(feature[2][mask[2]], dim=0)
        
        this_batch_feature = torch.stack([node_feature, motif_feature, graph_feature], dim=0)
        effective_features.append(this_batch_feature)
        
    effective_features = torch.stack(effective_features, dim=0)
    
    return effective_features

class type1_moe(nn.Module):
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
        
        B, L, N, D = features.shape
        features = features.permute(0, 2, 1, 3).reshape(B*N, L, D)
        features = self.moe_layer(features)
        
        return features[0].reshape(B, N, L, D).permute(0, 2, 1, 3), features[1]
    
    
class type2_moe(nn.Module):
    def __init__(self, hidden_size: int = 300):
        super().__init__()
        module = nn.Linear(hidden_size, hidden_size)
        self.graph_moe = MoE(
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
        self.motif_moe = MoE(
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
        self.node_moe = MoE(
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
        
        graph, motif, node = features.permute(1, 0, 2, 3).unbind(0)
        graph = self.graph_moe(graph)
        motif = self.motif_moe(motif)
        node = self.node_moe(node)
        
        features = torch.stack([graph[0], motif[0], node[0]], dim=1)
        
        return features
    
        
class pooled_moe(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size, type):
        super().__init__()
        if type == "type1":
            self.moe = type1_moe(mm_hidden_size)
        elif type == "type2":
            self.moe = type2_moe(mm_hidden_size)
        elif type == "linear":
            self.moe = nn.Identity()
        else:
            raise NotImplementedError
        
        self.projector = nn.Linear(mm_hidden_size, hidden_size)
        
    def forward(self, features: NestedTensor) -> torch.Tensor:
        outputs = self.moe(features.tensors)
        features, aux_loss = NestedTensor(outputs[0], features.mask), outputs[1]
        features = NestedTensor(self.projector(features.tensors), features.mask)
        
        features = parallel_mean_pooling(features, "mean")
        
        return features, aux_loss
    

class single_level_linear(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size, feature_level:str= "GRAPH"):
        super().__init__()
        self.projector = nn.Linear(mm_hidden_size, hidden_size)
        self.feature_index = GRAPHLEVEL[feature_level]
        
    def forward(self, x: NestedTensor):
        feature, mask = x.decompose()
        
        feature, mask = feature[:, self.feature_index], mask[:, self.feature_index]
        
        not_mask = ~mask
        effective_features = []
        for feat, m in zip(feature, not_mask):
            if self.feature_index == GRAPHLEVEL["NODE"]:
                pooler = GETPOOLER["mean"]
                effective_features.append(pooler(feat[m], dim=0))
            else:
                effective_features.append(feat[m])
            
        effective_features = torch.cat(effective_features, dim=0)
        
        effective_features = self.projector(effective_features)
        
        return effective_features
    
    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.projector.load_state_dict(state_dict, strict)
        

def build_xmodal_projector(
    config, 
    delay_load=False, 
    **kwargs
    ):
    """
    
    Args:
        config: Config of the projector
        delay_load: Discarded

    Raises:
        ValueError: Must provide linear , mlp or identity type projector

    Returns:
        _type_: _description_
    """
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':  
        
        return single_level_linear(config.mm_hidden_size, config.hidden_size)
    

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:  
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    if "multilevel" in projector_type:
        return pooled_moe(config.mm_hidden_size, config.hidden_size, projector_type.split("_")[1])
    
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')