from deepspeed.moe.layer import MoE
import torch



def create_lottery(moe_layer: MoE, percent):
    # TODO
    # 1. get parameters inside MoE
    # 2. hook all parameters
    # This is a pointer, right?
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
    # TODO: find the smallest magnitude
    
    # 1. find smallest magnitude Done
    # 2. random choose the weight Done
    # 3. random intialize them(But the initialization needs to be carefully designed??...) Done
    # NOTE: checked OK with toy sample:
    # a = torch.randn(4, 4)
    # pruned_weight = prune_by_percent_once(0.3, None, a)
    # print(pruned_weight)
    # NOTE: Weights should be flattened
    is_weight = False
    if len(final_weight.shape) == 2:
        in_feature, out_feature = final_weight.shape
        is_weight = True
    elif len(final_weight.shape) == 1:
        num_feature = final_weight.shape
    else:
        raise NotImplementedError()
    
    # flatten all for torch.sort
    # for weight: in_f * out_f, for bias: no changes
    flattend_weight = final_weight.flatten(0, -1)
    
    if mask is not None:
        sorted_weights = torch.sort(torch.abs(flattend_weight[mask == 1]))[0]
    else:
        sorted_weights = torch.sort(torch.abs(flattend_weight))[0]
    
    # Determine the cutoff for weights to be pruned.
    cutoff_index = torch.round(torch.tensor(percent * sorted_weights.shape[0])).to(torch.int)
    cutoff = sorted_weights[cutoff_index]
    
    # find indexes for weights to prune
    prune_indices = torch.where(torch.abs(flattend_weight) <= cutoff)[0]
    
    # random choose here
    randindices = torch.randperm(len(prune_indices))
    # choose half of the indices
    prune_indices = prune_indices[randindices[:len(prune_indices)//2]]
    # random intialize the pruned weights
    # 1. set to zero
    # 2. [x] mask
    # 3. coefficient
    # final_weight[prune_indices] = torch.randn(len(prune_indices))
    # NOTE: I think we should hook the backward after the reshape operation
    mask = torch.ones(flattend_weight.shape, dtype=flattend_weight.dtype, device=flattend_weight.device)
    mask[prune_indices] = 0
    # reshape back
    if is_weight:
        mask = mask.reshape(in_feature, out_feature)
        
    mask_hook = GradMask(mask)
        
    a = final_weight.register_hook(mask_hook)
        
    return final_weight