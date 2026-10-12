import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    features_t = torch.tensor(features,dtype=torch.float32)
    weights_t = torch.tensor(weights,dtype=torch.float32)
    labels_t = torch.tensor(labels,dtype=torch.float32)
    labels_hat = torch.sigmoid(torch.matmul(features_t,weights_t)+bias)
    mse = F.mse_loss(labels_t , labels_hat)
    return torch.round(labels_hat,decimals=4).tolist() , torch.round(mse,decimals=4).item()