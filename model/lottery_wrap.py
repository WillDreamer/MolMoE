from deepspeed.moe.layer import MoE
import torch



def create_lottery(moe_layer: MoE, percent):
    
    
    
    
    experts = moe_layer.deepspeed_moe.experts.deepspeed_experts
    for name, param in experts.named_parameters():
        prune_by_percent_once(percent, None, param)
        

class GradMask():
    def __init__(self, mask) -> None:
        self.mask = mask
        
    def __call__(self, grad: torch.Tensor):
        return self.mask.type_as(grad).to(grad.device) * grad
    

def prune_by_percent_once(percent: float, mask: torch.BoolTensor, final_weight: torch.Tensor):
    """Return new masks that involve pruning the smallest of the final weights.

    Args:
    percents: A dictionary determining the percent by which to prune each layer.
        Keys are layer names and values are floats between 0 and 1 (inclusive).
    masks: A dictionary containing the current masks. Keys are strings and
        values are numpy arrays with values in {0, 1}.
    final_weights: The weights at the end of the last training run. A
        dictionary whose keys are strings and whose values are numpy arrays.

    Returns:
    A dictionary containing the newly-pruned masks.
    """
    
    
    
    
    
    
    
    
    
    
    is_weight = False
    if len(final_weight.shape) == 2:
        in_feature, out_feature = final_weight.shape
        is_weight = True
    elif len(final_weight.shape) == 1:
        num_feature = final_weight.shape
    else:
        raise NotImplementedError()
    
    
    
    flattend_weight = final_weight.flatten(0, -1)
    
    if mask is not None:
        sorted_weights = torch.sort(torch.abs(flattend_weight[mask == 1]))[0]
    else:
        sorted_weights = torch.sort(torch.abs(flattend_weight))[0]
    
    
    cutoff_index = torch.round(torch.tensor(percent * sorted_weights.shape[0])).to(torch.int)
    cutoff = sorted_weights[cutoff_index]
    
    
    prune_indices = torch.where(torch.abs(flattend_weight) <= cutoff)[0]
    
    
    randindices = torch.randperm(len(prune_indices))
    
    prune_indices = prune_indices[randindices[:len(prune_indices)//2]]
    
    
    
    
    
    
    mask = torch.ones(flattend_weight.shape, dtype=flattend_weight.dtype, device=flattend_weight.device)
    mask[prune_indices] = 0
    
    if is_weight:
        mask = mask.reshape(in_feature, out_feature)
        
    mask_hook = GradMask(mask)
        
    a = final_weight.register_hook(mask_hook)
        
    return final_weight