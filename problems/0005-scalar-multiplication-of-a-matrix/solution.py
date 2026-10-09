import torch

def scalar_multiply(matrix, scalar) -> torch.Tensor:
    """
    Multiply each element of a 2D matrix by a scalar using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of the same shape.
    """
    # Convert input to tensor
    m_t = torch.as_tensor(matrix, dtype=torch.float)
    m,n =m_t.shape
    for i in range(m):
        for j in range(n):
            m_t[i,j] *= scalar
    return m_t
