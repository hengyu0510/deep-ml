import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    detT = torch.linalg.det(T_t) #torch中求行列式，linalg（线性代数）
    detS = torch.linalg.det(S_t)
    if torch.isclose(detT, torch.tensor(0.)):
        return torch.tensor(-1.)
    elif torch.isclose(detS, torch.tensor(0.)):
        return torch.tensor(-1.)
    else :
        return T_t.inverse() @ A_t @ S_t
