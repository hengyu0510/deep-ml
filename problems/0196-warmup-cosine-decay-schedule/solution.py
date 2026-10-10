import torch

def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> torch.Tensor:
    """
    Compute learning rate schedule with linear warmup and cosine decay.
    
    Args:
        T: Total number of training steps
        W: Number of warmup steps
        lr_max: Maximum learning rate (reached after warmup)
        lr_min: Minimum learning rate (reached at end of training)
    
    Returns:
        Tensor of learning rates for each step
    """
    # Your code here
    lr_steps=torch.zeros(T,dtype=torch.float)
    for i in range(W):
        lr_steps[i]=i/W * lr_max
    for j in range(W,T):
        lr_steps[j]=lr_min+0.5*(lr_max-lr_min)*(1+torch.cos(torch.tensor(((j-W)/(T-W))*torch.pi)))
    return lr_steps