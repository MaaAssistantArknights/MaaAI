import torch
import torch.nn as nn
import torch.nn.functional as F


def _reduce_loss(loss, reduction):
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


class CostSensitiveCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', confusion_cost=None, confusion_weight=0.0):
        super().__init__()
        if weight is not None:
            self.register_buffer('weight', weight.float())
        else:
            self.weight = None

        if confusion_cost is not None:
            self.register_buffer('confusion_cost', confusion_cost.float())
        else:
            self.confusion_cost = None

        self.reduction = reduction
        self.confusion_weight = confusion_weight

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()
        ce_loss = F.nll_loss(log_probs, targets, weight=self.weight, reduction='none')

        if self.confusion_cost is not None and self.confusion_weight > 0:
            confusion_penalty = (probs * self.confusion_cost[targets]).sum(dim=1)
            ce_loss = ce_loss + self.confusion_weight * confusion_penalty

        return _reduce_loss(ce_loss, self.reduction)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean', confusion_cost=None,
                 confusion_weight=0.0):
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
        if weight is not None:
            self.register_buffer('weight', weight.float())
        else:
            self.weight = None

        if confusion_cost is not None:
            self.register_buffer('confusion_cost', confusion_cost.float())
        else:
            self.confusion_cost = None

        self.reduction = reduction
        self.confusion_weight = confusion_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where C = number of classes
            targets: (N,) where each value is 0 ≤ targets[i] ≤ C-1
        """
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        ce_loss = -target_log_probs

        if self.weight is not None:
            ce_loss = ce_loss * self.weight[targets]

        pt = target_log_probs.exp()

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.confusion_cost is not None and self.confusion_weight > 0:
            confusion_penalty = (probs * self.confusion_cost[targets]).sum(dim=1)
            focal_loss = focal_loss + self.confusion_weight * confusion_penalty

        return _reduce_loss(focal_loss, self.reduction)
