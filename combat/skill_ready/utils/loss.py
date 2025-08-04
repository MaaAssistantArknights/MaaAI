import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        """
        Focal Loss implementation

        Args:
            alpha (float or tensor): Weighting factor for rare class (default: 1.0)
            gamma (float): Focusing parameter to down-weight easy examples (default: 2.0)
            weight (tensor, optional): Manual rescaling weight given to each class
            reduction (str): Specifies the reduction to apply to the output:
                           'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where C = number of classes
            targets: (N,) where each value is 0 ≤ targets[i] ≤ C-1
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
