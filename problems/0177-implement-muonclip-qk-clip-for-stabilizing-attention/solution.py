import torch

@torch.no_grad()
def muonclip_qk_clip(W_q: torch.Tensor, W_k: torch.Tensor, x: torch.Tensor, t: float, alpha: float = 0.5, eps: float = 1e-7):
	"""PyTorch version of MuonClip qk-clip (no grad). Returns (W_q_new, W_k_new, clipped, max_post)."""
	# compute q, k, scores; rescale if needed; round outputs
	# 本题需要考虑Batch
	Q = torch.matmul(W_q,x)
	K = torch.matmul(W_k,x)
	score = torch.matmul(Q,K.transpose(-1,-2)) / (Q.shape[2] ** 0.5) #用transpose(-1,-2)更通用，对于高维的tensor
	M = score.max().item() #score.max()是tensor，这里还需要转成float
	eta = min(1,t / M)
	if t>=M:
		clip_sign = False
	else:
		clip_sign = True
	W_q = (eta ** alpha) * W_q
	W_k = (eta ** (1-alpha)) * W_k
	max_post = eta * M
	return W_q.round(decimals=4) , W_k.round(decimals=4) , clip_sign , max_post
