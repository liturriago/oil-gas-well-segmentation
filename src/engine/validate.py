"""
Validation loop: validate_one_epoch.

No gradients are computed.  Metrics are accumulated over the whole val split
and returned as a single averaged dict.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from src.engine.ddp_utils import is_main_process
from src.losses.combined_loss import CombinedLoss
from src.metrics.segmentation_metrics import MetricAccumulator, compute_segmentation_metrics


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    epoch: int = 0,
    use_amp: bool = True,
    metric_threshold: float = 0.5,
) -> dict[str, float]:
    """Run one validation epoch (no gradient computation).

    Args:
        model:            The segmentation model (possibly DDP-wrapped).
        loader:           Validation DataLoader yielding ``{"image", "mask"}`` dicts.
        criterion:        Combined Focal+Dice loss.
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
    total_focal = 0.0
    total_dice = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"[Val]   Epoch {epoch}", disable=not is_main_process())

    for batch in pbar:
        images: torch.Tensor = batch["image"].to(device, non_blocking=True)
        masks: torch.Tensor = batch["mask"].to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            logits: torch.Tensor = model(images)
            loss, components = criterion(logits, masks)

        batch_metrics = compute_segmentation_metrics(
            logits, masks, threshold=metric_threshold
        )
        accumulator.update(batch_metrics)

        total_loss += loss.item()
        total_focal += components["focal"].item()
        total_dice += components["dice"].item()
        num_batches += 1

        if is_main_process():
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{batch_metrics.get('dice_macro', 0):.4f}",
            )

    results = accumulator.compute()
    results["loss"] = total_loss / max(num_batches, 1)
    results["loss_focal"] = total_focal / max(num_batches, 1)
    results["loss_dice"] = total_dice / max(num_batches, 1)

    return results
