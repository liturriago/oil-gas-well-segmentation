"""
Unit tests for loss functions: DiceLoss, FocalLoss, and CombinedLoss.
"""

import pytest
import torch
from src.losses import DiceLoss, FocalLoss, CombinedLoss

class TestDiceLoss:
    """Tests for DiceLoss."""

    def test_dice_forward(self):
        criterion = DiceLoss(smooth=1.0)
        preds = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = criterion(preds, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Mean reduction by default

    def test_dice_reductions(self):
        preds = torch.randn(2, 1, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 8, 8)).float()
        
        # None
        loss_none = DiceLoss(reduction='none')(preds, targets)
        assert loss_none.shape == (2, 1)
        
        # Sum
        loss_sum = DiceLoss(reduction='sum')(preds, targets)
        assert torch.allclose(loss_sum, loss_none.sum())

class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_focal_forward(self):
        criterion = FocalLoss()
        inputs = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = criterion(inputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_focal_reductions(self):
        inputs = torch.randn(2, 1, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 8, 8)).float()
        
        # None
        loss_none = FocalLoss(reduction='none')(inputs, targets)
        assert loss_none.shape == (2, 1, 8, 8)
        
        # Sum
        loss_sum = FocalLoss(reduction='sum')(inputs, targets)
        assert torch.allclose(loss_sum, loss_none.sum())

class TestCombinedLoss:
    """Tests for CombinedLoss."""

    def test_combined_forward(self):
        criterion = CombinedLoss()
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = criterion(logits, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_weighted_sum(self):
        dw, fw = 2.0, 0.5
        ds = 1e-6
        fa = 0.75
        fg = 2.0
        criterion = CombinedLoss(
            dice_weight=dw, 
            focal_weight=fw, 
            dice_smooth=ds, 
            focal_alpha=fa, 
            focal_gamma=fg
        )
        logits = torch.randn(2, 1, 16, 16)
        targets = torch.randint(0, 2, (2, 1, 16, 16)).float()
        
        total_loss = criterion(logits, targets)
        
        # Manually compute with SAME parameters
        dice_loss = DiceLoss(smooth=ds)(logits, targets)
        focal_loss = FocalLoss(alpha=fa, gamma=fg)(logits, targets)
        expected = dw * dice_loss + fw * focal_loss
        
        assert torch.allclose(total_loss, expected, atol=1e-5)

    def test_gradient_flows(self):
        criterion = CombinedLoss()
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = criterion(logits, targets)
        loss.backward()
        assert logits.grad is not None
