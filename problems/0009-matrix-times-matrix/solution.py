import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    a_t=torch.tensor(a)
    b_t=torch.tensor(b)
    if a_t.shape[1] != b_t.shape[0]:
        return -1
    return torch.matmul(torch.tensor(a),torch.tensor(b))
