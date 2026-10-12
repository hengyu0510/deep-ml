import torch
import math

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    # Your code here
    if position == 0 or d_model <=0:
        return -1
    position_embedding = torch.zeros(position,d_model,dtype=torch.float16)
    
    def calculate_pe(position: int , d_model: int):
        sequence = torch.zeros(d_model)
        for i in range (0,d_model): #注意i是从1开始的！
            if i % 2 == 0 : 
                sequence[i] = math.sin(position / 10000 ** ( i / d_model))
            else:
                sequence[i] = math.cos(position / 10000 ** ( (i-1) / d_model))
        return sequence

    for i in range(position):
        position_embedding[i] = calculate_pe(i,d_model)

    return position_embedding