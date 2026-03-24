import pytest
import torch
from omegaconf import OmegaConf

from src.config.schema import Config
from src.models.resunet import ResUNet
from src.training.lightning_module import SegmentationModule


def test_config_loading():
    """Test loading and validating the main configuration."""
    cfg_raw = OmegaConf.load("configs/config.yaml")
    cfg_python = OmegaConf.to_container(cfg_raw, resolve=True)
    cfg = Config(**cfg_python)

    assert cfg.training.lr > 0
    assert cfg.model.backbone_name == "resnet34"
    assert cfg.loss.focal_weight >= 0


def test_model_forward():
    """Test a simple forward pass through the model to check connectivity."""
    model = ResUNet(in_channels=4, out_channels=1, backbone_name="resnet34", pretrained=False)

    # 1 batch, 3 channels, 64x64 image
    dummy_input = torch.randn(2, 4, 64, 64)
    out = model(dummy_input)

    # Output should have same spatial dimensions
    assert out.shape == (2, 1, 64, 64)


def test_training_step():
    """Test a single training step on a dummy batch."""
    cfg_raw = OmegaConf.load("configs/config.yaml")
    cfg_python = OmegaConf.to_container(cfg_raw, resolve=True)
    cfg = Config(**cfg_python)

    module = SegmentationModule(cfg)

    dummy_batch = (
        torch.rand((4, 4, 64, 64), dtype=torch.float32),
        torch.randint(0, 2, (4, 64, 64), dtype=torch.float32),
    )

    loss = module.training_step(dummy_batch)

    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
