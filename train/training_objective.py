import torch

def self_regression_loss(
    original,
    reconstructed
):
    """
    Calculate the self-regression loss between the original and reconstructed data."
    """
    # Calculate the mean squared error between the original and reconstructed data
    mse_loss = torch.mean((original - reconstructed) ** 2)
    
    return mse_loss

def entropy_eval(
    hidden,
):
    """
    Calculate the entropy of the hidden layer.
    """
    # Calculate the probabilities of 0s and 1s in the hidden layer
    p_1 = torch.mean(hidden, dim=0)
    entropy = -torch.sum(torch.mul(p_1, torch.log2(p_1 + 1e-10)) + torch.mul((1 - p_1), torch.log2(1 - p_1 + 1e-10)))

    return entropy

def entropy_of_binary_hidden(
    hidden,
    threshold = 0, # Default threshold for entropy, used to limit entropy to a certain threshold by the optimization process
):
    """
    Calculate the entropy of the binary hidden layer.
    """
    # Calculate the probabilities of 0s and 1s in the hidden layer
    p_1 = torch.mean(hidden, dim=0)
    entropy = -torch.sum(torch.mul(p_1, torch.log2(p_1 + 1e-10)) + torch.mul((1 - p_1), torch.log2(1 - p_1 + 1e-10)))
    if entropy < threshold:
        return 0

    return entropy

def covarience_of_binary_hidden(
    hidden,
):
    """
    Calculate the covariance of the binary hidden layer.
    """
    # Calculate the covariance of the hidden layer
    covarience = torch.cov(hidden.T)
    cov_sum = torch.sum(torch.abs(covarience)) - torch.sum(torch.abs(torch.diag(covarience)))

    return cov_sum

def L1_norm(
    activation,
):
    """
    Calculate the L1 norm of the activation.
    """
    # Calculate the L1 norm of the activation
    l1_norm = torch.sum(torch.abs(activation))

    return l1_norm