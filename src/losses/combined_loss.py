import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        targets = targets.float()

        probs = torch.sigmoid(logits)

        probs = probs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(
        self,
        focal_alpha=0.75,
        focal_gamma=2.0,
        dice_weight=1.0,
        focal_weight=1.0,
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        targets = targets.float()

        focal = sigmoid_focal_loss(
            logits,
            targets,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="mean",
        )

        dice = self.dice_loss(logits, targets)

        return self.focal_weight * focal + self.dice_weight * dice
