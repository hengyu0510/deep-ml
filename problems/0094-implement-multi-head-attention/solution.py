import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = torch.matmul(X,W_q)
    K = torch.matmul(X,W_k)
    V = torch.matmul(X,W_v)
    return (Q,K,V)

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    score = torch.matmul(Q,K.t()) / (Q.shape[1]**0.5)
    score = torch.softmax(score,dim=1)
    return torch.matmul(score,V)

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    # 很好的练习reshape的题目
    seq_len , d_k = Q.shape
    sub_d = d_k // n_heads #这里需要整除 
    Q = Q.reshape(seq_len,n_heads,sub_d).transpose(0,1)
    K = K.reshape(seq_len,n_heads,sub_d).transpose(0,1)
    V = V.reshape(seq_len,n_heads,sub_d).transpose(0,1)
    score = torch.matmul(Q,K.transpose(-1,-2)) / (sub_d ** 0.5)
    score = torch.softmax(score,dim=-1)
    return torch.matmul(score,V).transpose(0,1).reshape(seq_len,d_k)