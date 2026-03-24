"""
Combined segmentation loss: Focal Loss + Dice Loss.

Total Loss = focal_weight * FocalLoss(logits, targets)
           + dice_weight  * DiceLoss(logits, targets)

FocalLoss which operates on raw logits.
DiceLoss applies sigmoid internally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DICELoss(nn.Module):
    """
    Dice Loss for binary and multilabel segmentation.

    Args:
        smooth (float): Smoothing constant to avoid division by zero.
        reduction (str): 'mean' | 'sum' | 'none'

    Inputs:
        preds (torch.Tensor): Predicted probabilities, shape (N, C, H, W).
        targets (torch.Tensor): Ground truth, same shape, values 0 or 1.

    Returns:
        torch.Tensor: Dice loss (scalar if reduction is 'mean' or 'sum').
    """
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean') -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds (torch.Tensor): Predicted probabilities, shape (N, C, H, W).
            targets (torch.Tensor): Ground truth, same shape, values 0 or 1.

        Returns:
            torch.Tensor: Dice loss.
        """
        preds = preds.float()
        targets = targets.float()
        preds = torch.sigmoid(preds)
        # Flatten spatial dims
        N, C = preds.shape[:2]
        preds = preds.view(N, C, -1)
        targets = targets.view(N, C, -1)

        intersection = (preds * targets).sum(-1)
        union = preds.sum(-1) + targets.sum(-1)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice_score

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

class FocalLoss(nn.Module):
    """
    Focal Loss for binary and multilabel classification/segmentation.

    This loss is suitable for multilabel problems, where each sample (or pixel) can belong to multiple classes.
    It applies a sigmoid activation internally and computes the binary cross-entropy for each class/channel.

    Args:
        alpha (float): Weighting factor for the rare class (default: 0.25).
        gamma (float): Focusing parameter to minimize easy examples (default: 2).
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum' (default: 'none').

    Inputs:
        inputs (torch.Tensor): Logits tensor of shape (N, C, ...) or (N, ...).
        targets (torch.Tensor): Ground truth tensor of same shape as inputs, with values 0 or 1.

    Returns:
        torch.Tensor: Loss tensor. If reduction is 'none', same shape as inputs; else scalar.
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'none'
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the focal loss between `inputs` and the ground truth `targets`.

        Args:
            inputs (torch.Tensor): Logits tensor of shape (N, C, ...) or (N, ...).
            targets (torch.Tensor): Ground truth tensor of same shape as inputs, with values 0 or 1.

        Returns:
            torch.Tensor: Loss tensor. If reduction is 'none', same shape as inputs; else scalar.
        """
        inputs = inputs.float()
        targets = targets.float()
        # BCE with logits computes sigmoid + BCE in a numerically stable way
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                f"Invalid value for arg 'reduction': '{self.reduction}'. Supported: 'none', 'mean', 'sum'."
            )

class CombinedLoss(nn.Module):
    """Weighted sum of Focal Loss and Dice Loss.

    Args:
        focal_alpha:  Focal loss weighting factor (balances pos/neg).
        focal_gamma:  Focal loss modulating factor (focuses on hard examples).
        dice_weight:  Scalar weight for the dice loss term.
        focal_weight: Scalar weight for the focal loss term.
        dice_smooth:  Epsilon for dice denominator (numerical stability).
    """

    def __init__(
        self,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        dice_smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        focal = self.focal_loss(logits,targets)
        dice = self.dice_loss(logits,targets)

        total = self.focal_weight * focal + self.dice_weight * dice

        components = {
            "focal": focal.detach(),
            "dice": dice.detach(),
            "total": total.detach(),
        }

        if torch.isnan(total):
            print("🚨 NaN detected in loss!")
            print("logits:", logits.min().item(), logits.max().item())
            print("targets:", targets.min().item(), targets.max().item())

        return total, components

