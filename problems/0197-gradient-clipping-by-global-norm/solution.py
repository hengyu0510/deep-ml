import torch

def clip_gradients_by_global_norm(gradients: list[torch.Tensor], max_norm: float) -> list[torch.Tensor]:
    """
    Clip gradients by global norm.
    
    Args:
        gradients: List of gradient tensors
        max_norm: Maximum allowed global norm
    
    Returns:
        List of clipped gradient tensors
    """
    # You should to learn how to operate on list
    # 遍历list中元素，for a in list
    total_norm = torch.sqrt(sum((g ** 2).sum() for g in gradients)) #遍历之后是tensor，按照torch的来
    if max_norm >= total_norm:
        return gradients
    else:
        return [g * max_norm/total_norm for g in gradients] 