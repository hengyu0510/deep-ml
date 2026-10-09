import torch

def reshape_matrix(a, new_shape) -> torch.Tensor:
    """
    Reshape a 2D matrix `a` to shape `new_shape` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a tensor of shape `new_shape`, or an empty tensor on mismatch.
    """
    # Dimension check
    if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
        return torch.tensor([])
    # Convert to tensor and reshape
    a_t = torch.as_tensor(a, dtype=torch.float)
    
    flat_vector = torch.flatten(a_t) #展平
    length = new_shape[0] * new_shape[1]
    ans = torch.zeros(new_shape, dtype=a_t.dtype) #统一数据类型
    for i in range (new_shape[0]):
        for j in range (new_shape[1]):
            ans[i,j] = flat_vector[i*new_shape[1]+j]
    return ans