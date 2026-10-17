import torch

def multiquery_attention(X: torch.Tensor, W_queries: list, W_key: torch.Tensor, W_value: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_queries: List of query weight tensors, each (d_model, d_k), one per head
        W_key: Shared key weight tensor of shape (d_model, d_k)
        W_value: Shared value weight tensor of shape (d_model, d_v)
        W_out: Output projection tensor of shape (num_heads * d_v, d_model)
    
    Returns:
        Output tensor of shape (seq_len, d_model), rounded to 4 decimal places
    """
    K = torch.matmul(X , W_key)
    V = torch.matmul(X , W_value)
    Q_queries = []
    attn_H = []
    for W_q in W_queries:
        Q_queries.append(torch.matmul(X, W_q))  # each (seq_len, d_k)
    for Q_h in Q_queries:
        score_h = torch.matmul(Q_h,K.t()) / (K.shape[-1]**0.5)
        attn_h = torch.matmul(torch.softmax(score_h,dim=-1),V)
        attn_H.append(attn_h)
    concat = torch.cat(attn_H, dim=-1) #list的元素进行一个雷霆的拼接
    return torch.round(torch.matmul(concat,W_out),decimals=4)