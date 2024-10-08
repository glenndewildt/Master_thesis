import torch
import torch.nn as nn
import numpy as np

class ConcordanceCorrelationCoefficientLoss(nn.Module):
    def __init__(self):
        super(ConcordanceCorrelationCoefficientLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()
        
        mean_true = torch.mean(y_true, dim=1, keepdim=True)
        mean_pred = torch.mean(y_pred, dim=1, keepdim=True)

        var_true = torch.var(y_true, dim=1, keepdim=True)
        var_pred = torch.var(y_pred, dim=1, keepdim=True)

        covariance = torch.mean((y_true - mean_true) * (y_pred - mean_pred), dim=1, keepdim=True)

        ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        loss = 1 - ccc  # Convert CCC to a loss (1 - ccc)

        return torch.mean(loss)
        
class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, y_pred, y_true):

        y_pred = y_pred.float()
        y_true = y_true.float()

        mean_y_pred = torch.mean(y_pred, dim=1, keepdim=True)
        mean_y_true = torch.mean(y_true, dim=1, keepdim=True)

        y_pred_centered = y_pred - mean_y_pred
        y_true_centered = y_true - mean_y_true

        covariance = torch.sum(y_pred_centered * y_true_centered, dim=1)
        std_y_pred = torch.sqrt(torch.sum(y_pred_centered ** 2, dim=1))
        std_y_true = torch.sqrt(torch.sum(y_true_centered ** 2, dim=1))

        pearson_correlation = covariance / (std_y_pred * std_y_true)
        loss = 1 - pearson_correlation  # Convert correlation to a loss (1 - r)

        return torch.mean(loss)