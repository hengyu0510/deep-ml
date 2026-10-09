import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    # Your implementation here
    num_features,n_observations = v_t.shape
    def calculate_covariance(feature1,feature2) -> torch.Tensor:
        X1=torch.as_tensor(feature1,dtype=torch.float)
        X2=torch.as_tensor(feature2,dtype=torch.float)
        length=X1.shape[0] #number of n_observations,shape返回的是tuple
        X1_m=X1.mean()
        X2_m=X2.mean()
        X1 = X1 - X1_m
        X2 = X2 - X2_m
        # X1 * X2的返回值是vector
        return (X1 * X2).sum() / (length-1)
    
    ans = torch.zeros(num_features,num_features,dtype=torch.float)
    for i in range(num_features):
        for j in range(num_features):
            ans[i,j]=calculate_covariance(v_t[i],v_t[j])
    return ans