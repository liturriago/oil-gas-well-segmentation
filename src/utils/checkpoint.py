"""
Checkpoint management: save and load model + optimizer + scheduler state.

Conventions:
  - ``best.pt``              — best model by foreground Dice score.
  - ``last.pt``              — most recent epoch checkpoint.
  - Checkpoints are only written by the main process (rank 0).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    checkpoint_dir: str | Path,
    state: dict[str, Any],
    filename: str,
) -> None:
    """Serialise *state* to ``{checkpoint_dir}/{filename}``.

    Args:
        checkpoint_dir: Directory path (created if it does not exist).
        state:          Arbitrary dict to serialise (e.g. model state dict,
                        optimizer state dict, epoch, metrics).
        filename:       Target file name, e.g. ``"best.pt"`` or ``"last.pt"``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and restore model (and optionally optimizer/scheduler) state.

    Args:
        path:      Path to the ``.pt`` checkpoint file.
        model:     Model instance to restore weights into.
        optimizer: Optimizer instance to restore state (optional).
        scheduler: LR scheduler instance to restore state (optional).
        device:    Device map for loading tensors.

    Returns:
        The full checkpoint dict (contains ``epoch``, ``metrics``, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)

    # Strip "module." prefix when loading a DDP checkpoint into a non-DDP model
    raw_state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in raw_state):
        raw_state = {k.replace("module.", "", 1): v for k, v in raw_state.items()}

    model.load_state_dict(raw_state)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    logger.info(f"Checkpoint loaded ← {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt


class CheckpointManager:
    """Manages best and last checkpoint saving during training.

    Args:
        checkpoint_dir: Directory to store checkpoints.
        monitor:        Metric key to monitor for improvement (higher == better).
                        Default: ``"dice_per_class[1]"`` (foreground Dice).
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "dice_per_class[1]",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.best_value: float = -1.0

    def _get_monitor_value(self, metrics: dict[str, Any]) -> float:
        """Extract the monitored scalar from a (possibly nested) metrics dict."""
        # Handle list keys like "dice_per_class[1]"
        if "[" in self.monitor:
            base, idx_part = self.monitor.rsplit("[", 1)
            idx = int(idx_part.rstrip("]"))
            return float(metrics[base][idx])
        return float(metrics[self.monitor])

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        val_metrics: dict[str, Any],
    ) -> bool:
        """Save last checkpoint; save best checkpoint if monitored metric improved.

        Args:
            epoch:       Current epoch number (1-indexed).
            model:       The model (DDP-wrapped or plain).
            optimizer:   Optimizer.
            scheduler:   LR scheduler.
            val_metrics: Validation metric dict from :func:`validate_one_epoch`.

        Returns:
            ``True`` if a new best checkpoint was saved.
        """
        # Unwrap DDP if necessary
        raw_model = model.module if hasattr(model, "module") else model

        state = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "val_metrics": val_metrics,
        }

        # Always save last
        save_checkpoint(self.checkpoint_dir, state, "last.pt")

        # Check if best
        current = self._get_monitor_value(val_metrics)
        is_best = current > self.best_value
        if is_best:
            self.best_value = current
            save_checkpoint(self.checkpoint_dir, state, "best.pt")
            logger.info(
                f"  ★ New best checkpoint!  {self.monitor} = {current:.4f}"
            )

        return is_best
