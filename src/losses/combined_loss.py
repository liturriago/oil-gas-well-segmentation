"""
Combined segmentation loss: Focal Loss + Dice Loss.

Total Loss = focal_weight * FocalLoss(logits, targets)
           + dice_weight  * DiceLoss(logits, targets)

FocalLoss which operates on raw logits.
DiceLoss applies sigmoid internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.losses.dice_loss import DiceLoss

class CombinedLoss(nn.Module):
    """Weighted sum of Focal Loss and Dice Loss.

    Args:
        focal_alpha:  Focal loss weighting factor (balances pos/neg).
        focal_gamma:  Focal loss modulating factor (focuses on hard examples).
        dice_weight:  Scalar weight for the dice loss term.
        focal_weight: Scalar weight for the focal loss term.
        dice_smooth:  Epsilon for dice denominator (numerical stability).
        logit_clamp:  Clamp logits to this range [-clamp, clamp] before loss.
    """

    def __init__(
        self,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        dice_smooth: float = 1e-6,
        logit_clamp: float = 20.0,
    ) -> None:
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.logit_clamp = logit_clamp

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
        # -----------------------------
        # 1. Shape + dtype
        # -----------------------------
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        with torch.cuda.amp.autocast(enabled=False):

            logits_fp32 = logits.float()
            targets_fp32 = targets.float()

            # -----------------------------
            # 2. 🔥 Clamp logits (CRÍTICO)
            # -----------------------------
            logits_fp32 = torch.clamp(logits_fp32, -self.logit_clamp, self.logit_clamp)

            # -----------------------------
            # 3. Focal Loss (estable)
            # -----------------------------
            probs = torch.sigmoid(logits_fp32)

            # 🔒 clamp probabilidades (evita log(0))
            probs = probs.clamp(1e-6, 1.0 - 1e-6)

            pt = torch.where(targets_fp32 == 1, probs, 1 - probs)

            alpha_t = torch.where(
                targets_fp32 == 1,
                torch.tensor(self.focal_alpha),
                torch.tensor(1 - self.focal_alpha),
            )

            focal_weight = (1 - pt).pow(self.focal_gamma)
            focal = -alpha_t * focal_weight * torch.log(pt)

            focal = focal.mean()

            # -----------------------------
            # 4. Dice Loss (ya estable)
            # -----------------------------
            dice = self.dice_loss(logits_fp32, targets_fp32)

        # -----------------------------
        # 5. Total
        # -----------------------------
        total = self.focal_weight * focal + self.dice_weight * dice

        components = {
            "focal": focal.detach(),
            "dice": dice.detach(),
            "total": total.detach(),
        }

        # -----------------------------
        # 6. Debug opcional (pro)
        # -----------------------------
        if torch.isnan(total):
            print("🚨 NaN detected in loss!")
            print("logits:", logits.min().item(), logits.max().item())
            print("probs:", probs.min().item(), probs.max().item())
            print("targets:", targets.min().item(), targets.max().item())

        return total, components

