"""
Custom Dice Loss for binary segmentation.

Design decisions:
  - Sigmoid is applied *internally* so the loss accepts raw logits.
  - Loss is computed batch-wise per sample (NOT flattened globally) to avoid
    the class-imbalance bias introduced by global flattening.
  - A smooth term (epsilon) prevents division by zero on all-negative batches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Binary Dice Loss operating on raw logits.

    Args:
        smooth: Additive epsilon for numerator and denominator to prevent
                division by zero. Default: 1e-6.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean binary Dice loss over the batch.

        Args:
            logits:  Raw model output, shape (N, 1, H, W).
            targets: Binary ground truth, shape (N, 1, H, W) or (N, H, W).
                     Values must be in {0, 1}.

        Returns:
            Scalar loss tensor ``1 - dice``.
        """
        # Ensure targets are float and same shape as logits
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)              # (N, 1, H, W)
        targets = targets.float()

        probs = torch.sigmoid(logits)                  # (N, 1, H, W)

        # Flatten spatial dims per sample: (N, H*W)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)          # (N,)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)        # (N,)

        dice_per_sample = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_per_sample.mean()

        return dice_loss
