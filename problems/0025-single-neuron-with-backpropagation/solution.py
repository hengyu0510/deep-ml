import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    updated_weights = initial_weights
    updated_bias = initial_bias
    mse = []
    for i in range(epochs):
        predict = torch.matmul(features,updated_weights)+updated_bias
        predict = nn.Sigmoid()(predict)
        mse.append(torch.round(torch.mean((predict-labels)**2),decimals=4).item())
        for j in range(features.shape[1]):
            weights_gradient = torch.mean(2*(predict - labels) * (predict) * (1-predict) * features.t()[j])
            updated_weights[j] = updated_weights[j] - learning_rate * weights_gradient 
        bias_gradient = torch.mean(2*(predict - labels) * (predict) * (1-predict))
        updated_bias = updated_bias - learning_rate * bias_gradient
    
    return torch.round(updated_weights, decimals=4).tolist(), round(updated_bias.item(),4),  mse