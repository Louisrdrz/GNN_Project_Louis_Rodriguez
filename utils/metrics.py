import torch

def mean_absolute_error(preds, targets):
    """
    Calculate Mean Absolute Error (MAE) between predictions and targets.
    Args:
        preds (torch.Tensor): Predicted values.
        targets (torch.Tensor): Actual values.
    Returns:
        float: Mean absolute error.
    """
    return torch.mean(torch.abs(preds - targets)).item()
