import torch

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> torch.Tensor:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (pi_theta / pi_theta_old).
        A: List of advantage estimates.
        pi_theta_old: List of old policy probabilities.
        pi_theta_ref: List of reference policy probabilities.
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value as a torch.Tensor.
    """
    # 先全部转成tensor
    rhos_t= torch.tensor(rhos,dtype=torch.float)
    A_t = torch.tensor(A,dtype=torch.float)
    pi_theta_old_t = torch.tensor(pi_theta_old,dtype=torch.float)
    pi_theta_ref_t = torch.tensor(pi_theta_ref,dtype=torch.float)

    pi_theta = rhos_t * pi_theta_old_t
    KL_term = rhos_t*(pi_theta_ref_t / pi_theta - torch.log(pi_theta_ref_t / pi_theta) -1 )
    J_GRPO= 1 / A_t.shape[0] * (torch.min(rhos_t*A_t,torch.clamp(rhos_t, min=1-epsilon, max=1+epsilon) * A_t) - beta * KL_term).sum()
    return J_GRPO

    # torch中有现成的clip函数torch.clamp(x,min=,max=)
    