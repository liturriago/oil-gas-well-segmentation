"""
Tests for loss functions: CombinedLoss.
"""

from __future__ import annotations

import pytest
import torch

from src.losses.combined_loss import CombinedLoss

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
