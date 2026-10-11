import numpy as np

def build_attention_mask(token_ids: np.ndarray, pad_token_id: int) -> np.ndarray:
    """
    Build a binary attention mask from padded token ids.

    Args:
        token_ids: array of shape (batch_size, seq_len)
        pad_token_id: integer id used for padding

    Returns:
        Integer mask array of shape (batch_size, seq_len) with 1 for real
        tokens and 0 for padding positions.
    """
    # 只能用numpy，草！
    num_token = token_ids.shape[0]
    token_len = token_ids.shape[1]
    ans = np.zeros((num_token,token_len)) #torch和numpy写法略有不同
    for i in range (num_token):
        for j in range (token_len):
            if token_ids[i,j] != pad_token_id :
                ans[i,j]=1
            else :
                ans[i,j]=0
    return ans