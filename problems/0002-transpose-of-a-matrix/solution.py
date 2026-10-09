import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix using PyTorch.
    
    Args:
        a: A 2D matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A transposed torch.Tensor
    """
    a_t = torch.as_tensor(a)
    m, n = a_t.shape #直接取形状
    ans = torch.zeros(n,m)
    for i in range(n):
        for j in range(m):
            ans[i,j] = a_t[j,i]
    return ans
    