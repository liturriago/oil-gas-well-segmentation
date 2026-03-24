"""
Main entry point for single-GPU training with AMP.

Launch with:
    python train.py

Hydra configuration is loaded from configs/config.yaml.
Pydantic validates all config fields before any heavy computation begins.
"""

from __future__ import annotations

import random

import numpy as np
from omegaconf import DictConfig, OmegaConf

import hydra
import torch
from torch.amp import GradScaler

from src.config.schema import validate_config, Config
from src.data.dataloader import build_dataloader
from src.engine.train import train_one_epoch
from src.engine.validate import validate_one_epoch
from src.losses.combined_loss import CombinedLoss
from src.losses.focal_loss import FocalLoss
from src.losses.dice_loss import DiceLoss
from src.models.resunet import ResUNet
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import get_logger, log_epoch

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def _build_optimizer(params, cfg: Config) -> torch.optim.Optimizer:
    name = cfg.training.optimizer.lower()
    lr = cfg.training.lr
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name!r}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = cfg.training.scheduler.lower()
    if name == "cosine":
        t_max = max(1, cfg.training.epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name!r}")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _run(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Device: %s", device)
    _logger.info("Configuration:\n%s", cfg.model_dump())

    # ------------------------------------------------------------------ Model
    model = ResUNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        encoder=cfg.model.encoder,
    ).to(device)

    # ------------------------------------------------------------------ Loss
    if cfg.loss.loss_type == "combined":
        criterion = CombinedLoss(
            focal_alpha=cfg.loss.focal_alpha,
            focal_gamma=cfg.loss.focal_gamma,
            dice_weight=cfg.loss.dice_weight,
            focal_weight=cfg.loss.focal_weight,
            dice_smooth=cfg.loss.dice_smooth,
            dice_reduction=cfg.loss.dice_reduction,
            focal_reduction=cfg.loss.focal_reduction,
        )
    elif cfg.loss.loss_type == "focal":
        criterion = FocalLoss(
            alpha=cfg.loss.focal_alpha,
            gamma=cfg.loss.focal_gamma,
            reduction=cfg.loss.focal_reduction,
        )
    elif cfg.loss.loss_type == "dice":
        criterion = DiceLoss(
            smooth=cfg.loss.dice_smooth,
            reduction=cfg.loss.dice_reduction,
        )
    else:
        raise ValueError(f"Unknown loss type: {cfg.loss.loss_type!r}")

    # ------------------------------------------------------------------ Optimizer & Schedulers
    optimizer = _build_optimizer(model.parameters(), cfg)
    main_sched = _build_scheduler(optimizer, cfg)

    # ------------------------------------------------------------------ AMP Scaler
    scaler: GradScaler | None = (
        GradScaler("cuda") if cfg.training.use_amp and torch.cuda.is_available() else None
    )

    # ------------------------------------------------------------------ DataLoaders
    train_loader = build_dataloader(
        shard_path=cfg.data.train_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.system.num_workers,
        augmentation=cfg.data.augmentation,
        training=True,
    )

    val_loader = build_dataloader(
        shard_path=cfg.data.val_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.system.num_workers,
        augmentation=False,
        training=False,
    )

    # ------------------------------------------------------------------ Checkpoint manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir=cfg.system.checkpoint_dir,
        monitor="dice_per_class[1]",
    )

    # ------------------------------------------------------------------ Training loop
    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            use_amp=cfg.training.use_amp,
            metric_threshold=cfg.metrics.threshold,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=cfg.training.use_amp,
            metric_threshold=cfg.metrics.threshold,
        )

        # ---- LR Scheduling ----
        current_lr = optimizer.param_groups[0]["lr"]
        if train_metrics.get("num_batches", 0) > 0:
            if main_sched is not None:
                main_sched.step()

        # ---- Logging & Checkpointing ----
        log_epoch(epoch, cfg.training.epochs, train_metrics, val_metrics, current_lr)
    ckpt_manager.save(epoch, model, optimizer, main_sched, val_metrics)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(raw_cfg: DictConfig) -> None:
    """Hydra-managed main function."""
    cfg_dict = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg: Config = validate_config(cfg_dict)
    _set_seed(cfg.system.seed)
    _run(cfg)


if __name__ == "__main__":
    main()
