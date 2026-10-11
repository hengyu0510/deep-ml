import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    # 掩码自注意力
    # 减去max的目的是1.避免exp的上溢 2.至少有为0的元素，不至于太负而exp之后都是0
    score = torch.matmul(Q,K.t()) / Q.shape[1]
    score += mask
    score = torch.softmax(score-score.max(), dim=1)
    return torch.matmul(score,V)
