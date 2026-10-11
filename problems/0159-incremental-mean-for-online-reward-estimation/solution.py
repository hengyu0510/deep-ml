import torch

def incremental_mean(Q_prev: torch.Tensor, k: int, R: torch.Tensor) -> torch.Tensor:
    """
    Q_prev: previous mean estimate (torch.Tensor)
    k: number of times the action has been selected (int)
    R: new observed reward (torch.Tensor)
    Returns: new mean estimate (torch.Tensor)
    """
    # Your code here
    return Q_prev + 1/k * (R-Q_prev)