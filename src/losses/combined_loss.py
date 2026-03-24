"""
Combined segmentation loss: Focal Loss + Dice Loss.

Total Loss = focal_weight * FocalLoss(logits, targets)
           + dice_weight  * DiceLoss(logits, targets)

FocalLoss is sourced from ``torchvision.ops.sigmoid_focal_loss`` which
operates on raw logits.  DiceLoss applies sigmoid internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

from src.losses.dice_loss import DiceLoss


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

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined loss.

        Args:
            logits:  Raw model output, shape (N, 1, H, W).
            targets: Binary ground truth, shape (N, 1, H, W) or (N, H, W).

        Returns:
            Tuple of:
              - ``total_loss``: scalar combined loss.
              - ``components``: dict with keys ``'focal'``, ``'dice'``, ``'total'``
                for logging purposes.
        """
        # Normalise target shape and dtype
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)          # (N, 1, H, W)
        targets = targets.float()

        components: dict[str, torch.Tensor] = {}

        # --- Focal Loss ---
        # sigmoid_focal_loss returns per-element losses; reduce to mean
        focal = sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="mean",
        )
        components["focal"] = focal

        # --- Dice Loss ---
        dice = self.dice_loss(logits, targets)
        components["dice"] = dice

        # --- Weighted sum ---
        total = self.focal_weight * focal + self.dice_weight * dice
        components["total"] = total

        return total, components
