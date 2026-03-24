import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
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