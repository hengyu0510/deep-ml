import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    m, n = a_t.shape
    if mode == "row":
        ans = torch.zeros(m,dtype = a_t.dtype)
        ans = a_t.sum(dim=1)/n
    elif mode == "column":
        ans = torch.zeros(m,dtype = a_t.dtype)
        ans = a_t.sum(dim=0)/m
    return ans
