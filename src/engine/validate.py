"""
Validation loop: validate_one_epoch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from src.losses.combined_loss import CombinedLoss
from src.losses.focal import FocalLoss
from src.losses.dice import DiceLoss
from src.metrics.segmentation_metrics import MetricAccumulator, compute_segmentation_metrics


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: FocalLoss | DiceLoss | CombinedLoss,
    device: torch.device,
    epoch: int = 0,
    use_amp: bool = True,
    metric_threshold: float = 0.5,
) -> dict[str, float]:
    """Run one validation epoch (no gradient computation).

    Args:
        model:            The segmentation model.
        loader:           Validation DataLoader yielding ``{"image", "mask"}`` dicts.
        criterion:        Focal, Dice or Combined loss.
        device:           Target device for tensors.
        epoch:            Current epoch index (for logging).
        use_amp:          Enable ``torch.amp.autocast``.
        metric_threshold: Threshold for binarising predictions.

    Returns:
        Dict with averaged validation metrics over the epoch.
    """
    model.eval()
    accumulator = MetricAccumulator()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"[Val]   Epoch {epoch}")

    for batch in pbar:
        images: torch.Tensor = batch["image"].to(device, non_blocking=True)
        masks: torch.Tensor = batch["mask"].to(device, non_blocking=True)

        if use_amp:
            with autocast('cuda'):
                logits = model(images)
                loss = criterion(logits,masks)
        else:
            logits = model(images)
            loss = criterion(logits,masks)

        batch_metrics = compute_segmentation_metrics(
            logits, masks, threshold=metric_threshold
        )
        accumulator.update(batch_metrics)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{batch_metrics.get('dice_macro', 0):.4f}",
        )

    results = accumulator.compute()
    results["loss"] = total_loss / max(num_batches, 1)

    return results
