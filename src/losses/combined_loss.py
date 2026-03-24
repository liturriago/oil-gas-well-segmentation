"""
Combined segmentation loss: Focal Loss + Dice Loss.

Total Loss = focal_weight * FocalLoss(logits, targets)
           + dice_weight  * DiceLoss(logits, targets)

FocalLoss which operates on raw logits.
DiceLoss applies sigmoid internally.
"""

import torch
import torch.nn as nn
from src.losses.dice import DiceLoss
from src.losses.focal import FocalLoss

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

