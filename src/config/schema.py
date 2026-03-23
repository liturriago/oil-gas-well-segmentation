from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    lr: float = Field(..., gt=0.0)
    batch_size: int = Field(..., gt=0)
    epochs: int = Field(..., gt=0)
    optimizer: str
    scheduler: str
    use_amp: bool
    use_ddp: bool
    num_gpus: int = Field(..., ge=0)


class ModelConfig(BaseModel):
    name: str
    in_channels: int = Field(..., gt=0)
    out_channels: int = Field(..., gt=0)


class LossConfig(BaseModel):
    focal_alpha: float = Field(..., ge=0.0, le=1.0)
    focal_gamma: float = Field(..., ge=0.0)
    dice_weight: float = Field(..., ge=0.0)
    focal_weight: float = Field(..., ge=0.0)


class DataConfig(BaseModel):
    train_path: str
    val_path: str
    augmentation: bool


class MetricsConfig(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)
    average: str


class SystemConfig(BaseModel):
    seed: int
    num_workers: int = Field(..., ge=0)


class Config(BaseModel):
    training: TrainingConfig
    model: ModelConfig
    loss: LossConfig
    data: DataConfig
    metrics: MetricsConfig
    system: SystemConfig
