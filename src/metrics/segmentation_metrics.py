"""
Custom segmentation metrics for binary segmentation: Dice, Sensitivity, Specificity.

Design:
  - Implemented from scratch (NO torchmetrics dependency).
  - Accepts raw logits; applies sigmoid + threshold internally.
  - Reports per-class and macro-average values.
  - Handles division-by-zero gracefully via epsilon.

Metric formulas:
  Dice        = 2·TP / (2·TP + FP + FN)
  Sensitivity = TP / (TP + FN)          [Recall / True Positive Rate]
  Specificity = TN / (TN + FP)          [True Negative Rate]
"""

from __future__ import annotations

import torch


# Output type alias
MetricsDict = dict[str, float | list[float]]

def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-8,
) -> MetricsDict:
    """Compute Dice Score, Sensitivity, and Specificity for binary segmentation.

    Args:
        logits:    Model output **before sigmoid**, shape (N, 1, H, W).
        targets:   Binary ground truth, shape (N, 1, H, W) or (N, H, W).
                   Values must be in {0, 1}.
        threshold: Probability threshold for converting sigmoid output to
                   binary predictions. Default: 0.5.
        epsilon:   Small constant to avoid zero division. Default: 1e-8.

    Returns:
        Dictionary with the following keys:

        .. code-block:: python

            {
                "dice_macro":        float,           # macro-mean over classes
                "dice_per_class":    [float, float],  # [class_0, class_1]
                "sensitivity_macro": float,
                "sensitivity_per_class": [float, float],
                "specificity_macro": float,
                "specificity_per_class": [float, float],
            }

    Notes:
        - Inference is done on CPU-detached tensors to avoid keeping the
          computation graph alive during metric collection.
        - All values are Python floats for easy serialisation / logging.
    """
    # Detach and move to CPU/Float for metric calculation
    logits = logits.detach().float()
    targets = targets.detach().float()

    # Handle (N, H, W) -> (N, 1, H, W) if necessary
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}")

    # 1. Sigmoid -> threshold -> binary predictions
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    
    # Ensure targets are strictly binary (0 or 1)
    gts = (targets >= 0.5).float()

    # ------------------------------------------------------------------
    # Flattening: We treat the entire batch as one pool of pixels
    # ------------------------------------------------------------------
    preds_flat = preds.view(-1)
    gts_flat = gts.view(-1)

    # ------------------------------------------------------------------
    # Confusion Matrix for Class 1 (Foreground)
    # ------------------------------------------------------------------
    tp1 = (preds_flat * gts_flat).sum()
    fp1 = (preds_flat * (1.0 - gts_flat)).sum()
    fn1 = ((1.0 - preds_flat) * gts_flat).sum()
    tn1 = ((1.0 - preds_flat) * (1.0 - gts_flat)).sum()

    # Confusion Matrix for Class 0 (Background) - The Inverse
    tp0, fp0, fn0, tn0 = tn1, fn1, fp1, tp1

    # Internal helper for safety
    def _safe_div(num: torch.Tensor, den: torch.Tensor) -> float:
        return (num / (den + epsilon)).item()

    # ------------------------------------------------------------------
    # Metric Calculations
    # ------------------------------------------------------------------
    # Dice: 2*TP / (2*TP + FP + FN)
    dice_1 = _safe_div(2.0 * tp1, 2.0 * tp1 + fp1 + fn1)
    dice_0 = _safe_div(2.0 * tp0, 2.0 * tp0 + fp0 + fn0)
    
    # Sensitivity (Recall): TP / (TP + FN)
    sens_1 = _safe_div(tp1, tp1 + fn1)
    sens_0 = _safe_div(tp0, tp0 + fn0)
    
    # Specificity: TN / (TN + FP)
    spec_1 = _safe_div(tn1, tn1 + fp1)
    spec_0 = _safe_div(tn0, tn0 + fp0)

    return {
        "dice_macro": (dice_0 + dice_1) / 2.0,
        "dice_per_class": [dice_0, dice_1],
        "sensitivity_macro": (sens_0 + sens_1) / 2.0,
        "sensitivity_per_class": [sens_0, sens_1],
        "specificity_macro": (spec_0 + spec_1) / 2.0,
        "specificity_per_class": [spec_0, spec_1],
    }


class MetricAccumulator:
    """Running accumulator that averages metrics over multiple batches.

    Usage::

        acc = MetricAccumulator()
        for batch in dataloader:
            metrics = compute_segmentation_metrics(logits, targets)
            acc.update(metrics)
        summary = acc.compute()          # averaged over all batches
        acc.reset()                      # ready for next epoch
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: int = 0

    def update(self, metrics: MetricsDict) -> None:
        """Accumulate a single-batch metric dict."""
        self._counts += 1
        for key, value in metrics.items():
            if isinstance(value, list):
                # Store per-class lists as individual scalar keys
                for i, v in enumerate(value):
                    k = f"{key}[{i}]"
                    self._sums[k] = self._sums.get(k, 0.0) + v
            else:
                self._sums[key] = self._sums.get(key, 0.0) + float(value)

    def compute(self) -> MetricsDict:
        """Return averaged metrics over all accumulated batches."""
        if self._counts == 0:
            return {}
        averaged: MetricsDict = {}
        for key, total in self._sums.items():
            avg = total / self._counts
            if "[" in key:
                # Reconstruct list keys
                base, idx_str = key.rsplit("[", 1)
                idx = int(idx_str.rstrip("]"))
                lst = averaged.setdefault(base, [])
                # Grow list if necessary
                while len(lst) <= idx:  # type: ignore[arg-type]
                    lst.append(0.0)     # type: ignore[union-attr]
                lst[idx] = avg          # type: ignore[index]
            else:
                averaged[key] = avg
        return averaged

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._sums.clear()
        self._counts = 0
