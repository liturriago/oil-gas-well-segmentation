"""
Training loop: train_one_epoch with AMP support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.losses.combined_loss import CombinedLoss
from src.losses.focal import FocalLoss
from src.losses.dice import DiceLoss
from src.metrics.segmentation_metrics import MetricAccumulator, compute_segmentation_metrics

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: FocalLoss | DiceLoss | CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    epoch: int = 0,
    use_amp: bool = True,
    metric_threshold: float = 0.5,
) -> dict[str, float]:
    """Run one training epoch.

    Args:
        model:            The segmentation model.
        loader:           Training DataLoader yielding ``{"image", "mask"}`` dicts.
        criterion:        Focal, Dice or Combined loss.
        optimizer:        Configured optimizer.
        device:           Target device for tensors.
        scaler:           AMP gradient scaler (``None`` if AMP disabled).
        epoch:            Current epoch index (for logging).
        use_amp:          Enable ``torch.amp.autocast``.
        metric_threshold: Threshold for binarising predictions.

    Returns:
        Dict with averaged training metrics over the epoch.
    """
    model.train()
    accumulator = MetricAccumulator()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")

    for batch in pbar:
        images: torch.Tensor = batch["image"].to(device, non_blocking=True)
        masks: torch.Tensor = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            if scaler is None:
                raise RuntimeError("AMP is enabled but scaler is not initialized.")
            with autocast('cuda'):
                logits = model(images)
                loss = criterion(logits,masks)
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits,masks)
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()

        # -------- Metrics --------
        with torch.no_grad():
            batch_metrics = compute_segmentation_metrics(
                logits, masks, threshold=metric_threshold
            )
            accumulator.update(batch_metrics)

        loss_val = loss.detach().item()
        total_loss += loss_val
        num_batches += 1

        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            dice=f"{batch_metrics.get('dice_macro', 0):.4f}",
        )

    results = accumulator.compute()
    results["loss"] = total_loss / max(num_batches, 1)
    results["num_batches"] = num_batches

    return results
