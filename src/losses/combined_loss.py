import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits are raw scores, so apply sigmoid
        probs = torch.sigmoid(logits)
        
        # Flatten predictions and targets
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        # Focal Loss
        focal = sigmoid_focal_loss(
            logits, 
            targets.float(), 
            alpha=self.focal_alpha, 
            gamma=self.focal_gamma, 
            reduction="mean"
        )
        
        # Dice Loss
        dice = self.dice_loss(logits, targets)
        
        return self.focal_weight * focal + self.dice_weight * dice
