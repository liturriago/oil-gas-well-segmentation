"""
Tests for model forward pass correctness.
"""

from __future__ import annotations

import pytest
import torch

from src.models.resunet import ResUNet


@pytest.fixture(params=["resnet18", "resnet34", "resnet50"])
def encoder(request):
    return request.param


class TestResUNetForward:
    """Verify output shapes and dtype for various configurations."""

    def test_output_shape_matches_input_shape(self, encoder: str):
        """Output spatial dims must equal input spatial dims."""
        model = ResUNet(in_channels=4, out_channels=1, encoder=encoder)
        model.eval()
        x = torch.randn(2, 4, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 256, 256), (
            f"Expected (2, 1, 256, 256) but got {out.shape}"
        )

    def test_output_dtype_is_float32(self):
        model = ResUNet(in_channels=4, out_channels=1)
        model.eval()
        x = torch.randn(1, 4, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_no_sigmoid_applied(self):
        """Forward output should contain values outside [0, 1] (raw logits)."""
        model = ResUNet(in_channels=4, out_channels=1)
        model.eval()
        torch.manual_seed(0)
        # Large random input to force extreme logit values
        x = torch.randn(2, 4, 256, 256) * 5.0
        with torch.no_grad():
            out = model(x)
        # At least some values should be outside [0, 1] if sigmoid is NOT applied
        has_values_outside_unit_interval = (out < 0).any() or (out > 1).any()
        assert has_values_outside_unit_interval, "Logits appear to have sigmoid applied"

    def test_non_square_input(self):
        """Model should handle non-square inputs gracefully."""
        model = ResUNet(in_channels=4, out_channels=1)
        model.eval()
        x = torch.randn(1, 4, 128, 192)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 128, 192)

    def test_single_channel_input(self):
        """Model should work with a single input channel (no NIR)."""
        model = ResUNet(in_channels=1, out_channels=1)
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_multi_class_output(self):
        """Binary head can be replaced by multi-class head."""
        model = ResUNet(in_channels=4, out_channels=3)
        model.eval()
        x = torch.randn(2, 4, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3, 256, 256)

    def test_gradient_flows(self):
        """Backward pass should not raise and gradients should be non-None."""
        model = ResUNet(in_channels=4, out_channels=1)
        x = torch.randn(1, 4, 64, 64)
        out = model(x)
        loss = out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"
