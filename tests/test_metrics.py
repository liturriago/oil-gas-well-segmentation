"""
Tests for segmentation metrics: Dice, Sensitivity, Specificity edge cases.
"""

from __future__ import annotations

import pytest
import torch

from src.metrics.segmentation_metrics import (
    MetricAccumulator,
    compute_segmentation_metrics,
)


def _logits_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Convert probabilities to logits (inverse sigmoid)."""
    probs = probs.clamp(1e-6, 1 - 1e-6)
    return torch.log(probs / (1 - probs))


class TestComputeSegmentationMetrics:
    """Tests for :func:`compute_segmentation_metrics`."""

    def test_all_correct_foreground(self):
        """All-positive prediction on all-positive target → dice=1, sens=1, spec=1."""
        logits = _logits_from_probs(torch.ones(2, 1, 8, 8) * 0.99)
        targets = torch.ones(2, 1, 8, 8)
        m = compute_segmentation_metrics(logits, targets)
        assert m["dice_per_class"][1] == pytest.approx(1.0, abs=0.01)
        assert m["sensitivity_per_class"][1] == pytest.approx(1.0, abs=0.01)

    def test_all_correct_background(self):
        """All-negative prediction on all-negative target → dice=1 (background), spec=1."""
        logits = _logits_from_probs(torch.ones(2, 1, 8, 8) * 0.01)
        targets = torch.zeros(2, 1, 8, 8)
        m = compute_segmentation_metrics(logits, targets)
        assert m["dice_per_class"][0] == pytest.approx(1.0, abs=0.01)
        assert m["specificity_per_class"][1] == pytest.approx(1.0, abs=0.01)

    def test_all_wrong_fg_prediction(self):
        """Predicting all foreground when gt is all background → dice_fg ≈ 0."""
        logits = _logits_from_probs(torch.ones(2, 1, 8, 8) * 0.99)
        targets = torch.zeros(2, 1, 8, 8)
        m = compute_segmentation_metrics(logits, targets)
        assert m["dice_per_class"][1] < 0.05

    def test_sensitivity_zero_when_all_missed(self):
        """If all foreground pixels are missed (predicted as bg), sensitivity fg = 0."""
        logits = _logits_from_probs(torch.ones(2, 1, 8, 8) * 0.01)
        targets = torch.ones(2, 1, 8, 8)
        m = compute_segmentation_metrics(logits, targets)
        assert m["sensitivity_per_class"][1] < 0.05

    def test_specificity_zero_when_all_fp(self):
        """Predicting all positive when all negative → specificity = 0."""
        logits = _logits_from_probs(torch.ones(2, 1, 8, 8) * 0.99)
        targets = torch.zeros(2, 1, 8, 8)
        m = compute_segmentation_metrics(logits, targets)
        assert m["specificity_per_class"][1] < 0.05

    def test_output_keys_present(self):
        logits = torch.randn(2, 1, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 8, 8)).float()
        m = compute_segmentation_metrics(logits, targets)
        for key in [
            "dice_macro",
            "dice_per_class",
            "sensitivity_macro",
            "sensitivity_per_class",
            "specificity_macro",
            "specificity_per_class",
        ]:
            assert key in m, f"Missing key: {key}"

    def test_per_class_list_length(self):
        logits = torch.randn(2, 1, 8, 8)
        targets = torch.randint(0, 2, (2, 1, 8, 8)).float()
        m = compute_segmentation_metrics(logits, targets)
        assert len(m["dice_per_class"]) == 2
        assert len(m["sensitivity_per_class"]) == 2
        assert len(m["specificity_per_class"]) == 2

    def test_macro_is_mean_of_per_class(self):
        """Macro metric should equal the mean of per-class values."""
        logits = torch.randn(4, 1, 16, 16)
        targets = torch.randint(0, 2, (4, 1, 16, 16)).float()
        m = compute_segmentation_metrics(logits, targets)
        expected_dice_macro = sum(m["dice_per_class"]) / 2
        assert m["dice_macro"] == pytest.approx(expected_dice_macro, abs=1e-5)

    def test_accepts_3d_targets(self):
        """Should work with targets shaped (N, H, W) as well as (N, 1, H, W)."""
        logits = torch.randn(2, 1, 8, 8)
        targets_3d = torch.randint(0, 2, (2, 8, 8)).float()
        m = compute_segmentation_metrics(logits, targets_3d)
        assert "dice_macro" in m

    def test_no_nan_on_all_zeros_target(self):
        """All-zero targets should not produce NaN (division by zero guard)."""
        logits = torch.randn(2, 1, 8, 8)
        targets = torch.zeros(2, 1, 8, 8)
        m = compute_segmentation_metrics(logits, targets)
        for val in m.values():
            if isinstance(val, list):
                for v in val:
                    assert not torch.isnan(torch.tensor(v))
            else:
                assert not torch.isnan(torch.tensor(val))


class TestMetricAccumulator:
    """Tests for :class:`MetricAccumulator`."""

    def test_compute_returns_averages(self):
        acc = MetricAccumulator()
        m1 = {"dice_macro": 0.6, "dice_per_class": [0.7, 0.5]}
        m2 = {"dice_macro": 0.8, "dice_per_class": [0.9, 0.7]}
        acc.update(m1)
        acc.update(m2)
        result = acc.compute()
        assert result["dice_macro"] == pytest.approx(0.7, abs=1e-5)
        assert result["dice_per_class"][0] == pytest.approx(0.8, abs=1e-5)
        assert result["dice_per_class"][1] == pytest.approx(0.6, abs=1e-5)

    def test_reset_clears_state(self):
        acc = MetricAccumulator()
        acc.update({"dice_macro": 0.9})
        acc.reset()
        assert acc.compute() == {}

    def test_empty_accumulator_returns_empty_dict(self):
        acc = MetricAccumulator()
        assert acc.compute() == {}
