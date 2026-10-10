import torch

def adamw_update(w: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor, t: int, lr: float, beta1: float, beta2: float, epsilon: float, weight_decay: float) -> tuple:
    """
    Perform one AdamW optimizer step.
    Args:
      w: parameter tensor (torch.Tensor)
      g: gradient tensor (torch.Tensor)
      m: first moment tensor (torch.Tensor)
      v: second moment tensor (torch.Tensor)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """
    m_t = beta1 * m + (1-beta1) * g
    v_t = beta2 * v + (1-beta2) * g * g
    m_that = m_t / (1-beta1**t)
    v_that = v_t / (1-beta2**t)
    w_t = w - lr * weight_decay * w
    w_t = w_t - lr * m_that /(v_that**0.5 + epsilon) 
    return w_t , m_t , v_t