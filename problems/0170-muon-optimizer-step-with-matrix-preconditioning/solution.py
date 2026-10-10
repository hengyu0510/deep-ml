import torch
from typing import Tuple

def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    # Implement the Newton-Schulz iteration for matrix orthogonalization/preconditioning
    X_t = G/ ( eps + torch.linalg.norm(G))
    m,n=G.shape
    transpose = False
    if m>n:
        X_t = X_t.t()
        transpose = True
    for i in range(steps):
        A = X_t @ X_t.t()
        X_t = 3.4445 * X_t + (-4.7750 * A + 2.0315 *(A @ A)) @ X_t
    if transpose == True:
        return X_t.t()
    return X_t #得到更新方向O

def muon_step(theta: torch.Tensor, B: torch.Tensor, grad: torch.Tensor, 
              eta: float, mu: float, ns_steps: int = 5, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    theta: torch.Tensor, shape (M, N)
    B: torch.Tensor, shape (M, N)
    grad: torch.Tensor, shape (M, N)
    eta: float (learning rate)
    mu: float (momentum coefficient)
    ns_steps: int (Newton-Schulz steps)
    eps: float (numerical stability)
    Returns: updated theta, updated B
    """
    # Implement the Muon optimizer update step
    B_t = mu * B + grad
    O_t = newton_schulz5(B_t,ns_steps,eps)
    scale = ((theta.shape[0] * theta.shape[1]) ** 0.5) / (torch.linalg.norm(B_t)+eps)
    theta_t = theta - eta * scale * O_t
    return theta_t , B_t