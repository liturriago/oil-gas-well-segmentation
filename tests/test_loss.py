"""
Tests for loss functions: DiceLoss and CombinedLoss.
"""

from __future__ import annotations

import pytest
import torch

from src.losses.combined_loss import CombinedLoss
from src.losses.dice_loss import DiceLoss


class TestDiceLoss:
    """Unit tests for :class:`DiceLoss`."""

    def _make_logits(self, value: float, shape=(2, 1, 64, 64)) -> torch.Tensor:
        """Return a constant logit tensor (positive → prob > 0.5)."""
        return torch.full(shape, value)

    def test_perfect_prediction_is_zero(self):
        """When predictions perfectly match targets, dice loss ≈ 0."""
        criterion = DiceLoss()
        # Large positive logit → sigmoid ≈ 1.0
        logits = self._make_logits(10.0)
        targets = torch.ones(2, 1, 64, 64)
        loss = criterion(logits, targets)
        assert loss.item() < 0.01, f"Expected loss ≈ 0, got {loss.item()}"

    def test_all_wrong_prediction_near_one(self):
        """Predicting all positives when all negatives → dice ≈ 0 → loss ≈ 1."""
        criterion = DiceLoss()
        logits = self._make_logits(10.0)
        targets = torch.zeros(2, 1, 64, 64)
        loss = criterion(logits, targets)
        # Dice of 0 → loss = 1
        # With smooth denominator, loss < 1 but close
        assert loss.item() > 0.9, f"Expected loss ≈ 1, got {loss.item()}"

    def test_loss_is_scalar(self):
        criterion = DiceLoss()
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = criterion(logits, targets)
        assert loss.dim() == 0, "Loss must be a scalar"

    def test_loss_range_is_zero_to_one(self):
        criterion = DiceLoss()
        for seed in [0, 42, 123]:
            torch.manual_seed(seed)
            logits = torch.randn(4, 1, 64, 64)
            targets = torch.randint(0, 2, (4, 1, 64, 64)).float()
            loss = criterion(logits, targets)
            assert 0.0 <= loss.item() <= 1.0 + 1e-6

    def test_accepts_3d_target(self):
        """Loss should accept targets with shape (N, H, W)."""
        criterion = DiceLoss()
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 64, 64)).float()
        loss = criterion(logits, targets)
        assert torch.isfinite(loss)

    def test_gradient_exists(self):
        criterion = DiceLoss()
        logits = torch.randn(2, 1, 64, 64, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = criterion(logits, targets)
        loss.backward()
        assert logits.grad is not None


class TestCombinedLoss:
    """Unit tests for :class:`CombinedLoss`."""

    def test_returns_tuple(self):
        criterion = CombinedLoss()
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        result = criterion(logits, targets)
        assert isinstance(result, tuple) and len(result) == 2

    def test_component_keys(self):
        criterion = CombinedLoss()
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        _, components = criterion(logits, targets)
        assert "focal" in components
        assert "dice" in components
        assert "total" in components

    def test_total_equals_weighted_sum(self):
        dw, fw = 2.0, 0.5
        criterion = CombinedLoss(dice_weight=dw, focal_weight=fw)
        torch.manual_seed(7)
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        total, comps = criterion(logits, targets)
        expected = fw * comps["focal"] + dw * comps["dice"]
        assert torch.allclose(total, expected, atol=1e-5)

    def test_zero_focal_weight(self):
        """Setting focal_weight=0 should produce only dice contribution."""
        criterion = CombinedLoss(focal_weight=0.0, dice_weight=1.0)
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        total, comps = criterion(logits, targets)
        assert torch.allclose(total, comps["dice"], atol=1e-5)

    def test_gradient_flows(self):
        criterion = CombinedLoss()
        logits = torch.randn(2, 1, 64, 64, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        total, _ = criterion(logits, targets)
        total.backward()
        assert logits.grad is not None
