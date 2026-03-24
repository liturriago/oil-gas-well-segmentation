import torch
import torch.nn as nn
import torch.nn.functional as F

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