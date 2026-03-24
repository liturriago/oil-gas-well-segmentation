"""
Training loop: train_one_epoch.

Supports:
  - Mixed Precision Training (AMP) via torch.cuda.amp
  - Gradient clipping
  - Per-batch metric collection
  - All-reduce loss aggregation across DDP ranks
  - tqdm progress bar on the main process
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.engine.ddp_utils import is_main_process, reduce_tensor
from src.losses.combined_loss import CombinedLoss
from src.metrics.segmentation_metrics import MetricAccumulator, compute_segmentation_metrics


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    grad_clip: float = 0.0,
    epoch: int = 0,
    use_amp: bool = True,
    metric_threshold: float = 0.5,
) -> dict[str, float]:
    """Run one training epoch.

    Args:
        model:            The segmentation model (possibly DDP-wrapped).
        loader:           Training DataLoader yielding ``{"image", "mask"}`` dicts.
        criterion:        Combined Focal+Dice loss.
        optimizer:        Configured optimizer.
        device:           Target device for tensors.
        scaler:           AMP gradient scaler (``None`` if AMP disabled).
        grad_clip:        Max gradient norm (0 = disabled).
        epoch:            Current epoch index (for logging).
        use_amp:          Enable ``torch.cuda.amp.autocast``.
        metric_threshold: Threshold for binarising predictions.

    Returns:
        Dict with averaged training metrics over the epoch.
    """
    model.train()
    accumulator = MetricAccumulator()
    total_loss = 0.0
    total_focal = 0.0
    total_dice = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", disable=not is_main_process())

    for batch in pbar:
        images: torch.Tensor = batch["image"].to(device, non_blocking=True)
        masks: torch.Tensor = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # -------- Forward (with optional AMP) --------
        with autocast(enabled=use_amp):
            logits: torch.Tensor = model(images)           # (N, 1, H, W) logits
            loss, components = criterion(logits, masks)

        # -------- Backward --------
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # -------- Metrics (detached, on local predictions) --------
        with torch.no_grad():
            batch_metrics = compute_segmentation_metrics(
                logits, masks, threshold=metric_threshold
            )
            accumulator.update(batch_metrics)

        loss_val = reduce_tensor(loss.detach()).item()
        total_loss += loss_val
        total_focal += components["focal"].detach().item()
        total_dice += components["dice"].detach().item()
        num_batches += 1

        if is_main_process():
            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                dice=f"{batch_metrics.get('dice_macro', 0):.4f}",
            )

    # Average over batches
    results = accumulator.compute()
    results["loss"] = total_loss / max(num_batches, 1)
    results["loss_focal"] = total_focal / max(num_batches, 1)
    results["loss_dice"] = total_dice / max(num_batches, 1)

    return results
