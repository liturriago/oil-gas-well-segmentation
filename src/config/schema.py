"""
Pydantic config schema for validating Hydra-loaded configurations.

This module defines strict, typed models for all config groups.
Validation is performed at training startup before any heavy computation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class TrainingConfig(BaseModel):
    lr: float = Field(gt=0, description="Peak learning rate")
    batch_size: int = Field(gt=0, description="Per-GPU batch size")
    epochs: int = Field(gt=0, description="Total number of training epochs")
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    use_amp: bool = True
    grad_clip: float = Field(ge=0.0, description="Max gradient norm (0 = disabled)")
    warmup_epochs: int = Field(ge=0, description="Linear warmup epochs before scheduler")


class ModelConfig(BaseModel):
    in_channels: int = Field(ge=1, description="Number of input channels (e.g. 4 for RGB+NIR)")
    out_channels: int = Field(ge=1, description="Number of output segmentation classes")
    encoder: Literal["resnet18", "resnet34", "resnet50"] = "resnet34"


class LossConfig(BaseModel):
    focal_alpha: float = Field(gt=0, le=1, description="Focal loss alpha (class weighting)")
    focal_gamma: float = Field(ge=0, description="Focal loss gamma (focusing parameter)")
    dice_weight: float = Field(ge=0, description="Weight applied to dice loss term")
    focal_weight: float = Field(ge=0, description="Weight applied to focal loss term")
    dice_smooth: float = Field(gt=0, description="Epsilon added to dice denominator")

    @model_validator(mode="after")
    def at_least_one_loss_active(self) -> "LossConfig":
        if self.dice_weight == 0 and self.focal_weight == 0:
            raise ValueError("At least one of dice_weight or focal_weight must be > 0")
        return self


class DataConfig(BaseModel):
    train_path: str
    val_path: str
    image_size: int = Field(gt=0, description="Spatial size of images after resize")
    augmentation: bool = True


class MetricsConfig(BaseModel):
    threshold: float = Field(gt=0, lt=1, description="Sigmoid threshold for binarization")
    average: Literal["macro"] = "macro"


class SystemConfig(BaseModel):
    seed: int = 42
    num_workers: int = Field(ge=0)
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """Root configuration model validated at training startup."""

    training: TrainingConfig
    model: ModelConfig
    loss: LossConfig
    data: DataConfig
    metrics: MetricsConfig
    system: SystemConfig



def validate_config(cfg_dict: dict) -> Config:
    """Parse and validate a raw OmegaConf/dict config into a typed :class:`Config` object.

    Args:
        cfg_dict: Dictionary produced by ``OmegaConf.to_container(cfg, resolve=True)``.

    Returns:
        A fully validated :class:`Config` instance.

    Raises:
        pydantic.ValidationError: On any constraint violation.
    """
    return Config(**cfg_dict)
