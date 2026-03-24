"""
Main entry point for distributed training.

Launch with torchrun:
    torchrun --nproc_per_node=2 train.py

Or single-GPU:
    python train.py

Hydra configuration is loaded from configs/config.yaml.
Pydantic validates all config fields before any heavy computation begins.
"""

from __future__ import annotations

import os
import warnings
from omegaconf import DictConfig, OmegaConf

import hydra
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from src.config.schema import validate_config, Config
from src.data.dataloader import build_dataloader
from src.engine.ddp_utils import (
    barrier,
    get_rank,
    get_world_size,
    is_main_process,
    setup_ddp,
    teardown_ddp,
    wrap_model_ddp,
)
from src.engine.train import train_one_epoch
from src.engine.validate import validate_one_epoch
from src.losses.combined_loss import CombinedLoss
from src.models.resunet import ResUNet
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import get_logger, log_epoch

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def _build_optimizer(
    params, cfg: Config
) -> torch.optim.Optimizer:
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
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = cfg.training.scheduler.lower()
    if name == "cosine":
        # Subtract warmup epochs from T_max so cosine decay fits remaining epochs
        t_max = max(1, cfg.training.epochs - cfg.training.warmup_epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name!r}")


def _build_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if warmup_epochs <= 0:
        return None
    return torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
    )


# ---------------------------------------------------------------------------
# Training function (called per process)
# ---------------------------------------------------------------------------


def _run(cfg: Config) -> None:
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        _logger.info("Configuration:\n%s", cfg.model_dump())

    # ------------------------------------------------------------------ Model
    model = ResUNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        encoder=cfg.model.encoder,
    )

    if cfg.training.use_ddp and world_size > 1:
        model = wrap_model_ddp(model, rank)
    else:
        model = model.to(device)

    # ------------------------------------------------------------------ Loss
    criterion = CombinedLoss(
        focal_alpha=cfg.loss.focal_alpha,
        focal_gamma=cfg.loss.focal_gamma,
        dice_weight=cfg.loss.dice_weight,
        focal_weight=cfg.loss.focal_weight,
        dice_smooth=cfg.loss.dice_smooth,
    )

    # ------------------------------------------------------------------ Optimizer & Schedulers
    optimizer = _build_optimizer(model.parameters(), cfg)
    warmup_sched = _build_warmup_scheduler(optimizer, cfg.training.warmup_epochs)
    main_sched = _build_scheduler(optimizer, cfg)

    # ------------------------------------------------------------------ AMP Scaler
    scaler: GradScaler | None = (
        GradScaler() if cfg.training.use_amp and torch.cuda.is_available() else None
    )

    # ------------------------------------------------------------------ DataLoaders
    train_loader = build_dataloader(
        shard_path=cfg.data.train_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.system.num_workers,
        augmentation=cfg.data.augmentation,
        training=True,
        world_size=world_size,
        rank=rank,
    )

    val_loader = build_dataloader(
        shard_path=cfg.data.val_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.system.num_workers,
        augmentation=False,
        training=False,
        world_size=world_size,
        rank=rank,
    )

    # ------------------------------------------------------------------ Checkpoint manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir=cfg.system.checkpoint_dir,
        monitor="dice_per_class[1]",   # foreground Dice
    )

    # ------------------------------------------------------------------ Training loop
    for epoch in range(1, cfg.training.epochs + 1):
        barrier()  # sync all ranks at epoch start

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            grad_clip=cfg.training.grad_clip,
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
        if epoch <= cfg.training.warmup_epochs and warmup_sched is not None:
            warmup_sched.step()
        elif main_sched is not None:
            main_sched.step()

        # ---- Logging (rank 0 only) ----
        if is_main_process():
            log_epoch(epoch, cfg.training.epochs, train_metrics, val_metrics, current_lr)
            ckpt_manager.save(epoch, model, optimizer, main_sched, val_metrics)

    teardown_ddp()


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(raw_cfg: DictConfig) -> None:
    """Hydra-managed main function.

    Converts OmegaConf → dict → Pydantic Config, then dispatches training.
    When DDP is active (``use_ddp=true``), this function is invoked once per
    GPU process by ``torchrun`` / ``torch.distributed.launch``.
    """
    cfg_dict = OmegaConf.to_container(raw_cfg, resolve=True)
    cfg: Config = validate_config(cfg_dict)

    # Detect if launched by torchrun (env vars set automatically)
    launched_with_torchrun = "LOCAL_RANK" in os.environ
    use_ddp = cfg.training.use_ddp and launched_with_torchrun

    # ------------------------------------------------------------------ Guard
    if launched_with_torchrun and not cfg.training.use_ddp:
        world_size_env = int(os.environ.get("WORLD_SIZE", 1))
        warnings.warn(
            f"\n{'='*70}\n"
            f"  ⚠️  CONFIGURATION MISMATCH DETECTED\n"
            f"{'='*70}\n"
            f"  Launched with torchrun ({world_size_env} processes) but\n"
            f"  use_ddp: false is set in the config.\n\n"
            f"  Each process will run INDEPENDENTLY without gradient sync,\n"
            f"  reading the same data shards and writing to the same checkpoint\n"
            f"  directory — causing a RACE CONDITION.\n\n"
            f"  To fix, choose ONE of:\n"
            f"    a) Set  use_ddp: true  in configs/config.yaml\n"
            f"    b) Run  python train.py  instead of torchrun\n"
            f"{'='*70}",
            stacklevel=2,
        )

    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", cfg.training.num_gpus))
        setup_ddp(local_rank, world_size, seed=cfg.system.seed)
        _run(cfg)
    else:
        # Single-GPU or CPU fallback — still set the seed
        torch.manual_seed(cfg.system.seed)
        _run(cfg)


if __name__ == "__main__":
    main()
