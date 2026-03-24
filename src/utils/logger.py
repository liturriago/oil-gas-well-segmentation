"""
Structured epoch logger.

Prints a formatted summary table after each epoch to stdout.
Only the main DDP process (rank 0) should call this to avoid duplicate output.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

# Configure the root logger once at import time so all modules inherit it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_logger(name: str) -> logging.Logger:
    """Return a named :class:`logging.Logger`.

    Args:
        name: Logger name, typically ``__name__``.
    """
    return logging.getLogger(name)


def log_epoch(
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    lr: float,
) -> None:
    """Print a formatted table of training and validation metrics.

    Only numerical scalar values are printed; per-class lists are formatted
    as ``[bg, fg]`` inline.

    Args:
        epoch:         Current epoch (1-indexed).
        total_epochs:  Total number of epochs.
        train_metrics: Dict returned by :func:`train_one_epoch`.
        val_metrics:   Dict returned by :func:`validate_one_epoch`.
        lr:            Current learning rate for display.
    """
    logger = get_logger("epoch")
    sep = "─" * 72

    lines = [
        "",
        sep,
        f"  Epoch {epoch:>3d} / {total_epochs}   |   LR: {lr:.6f}",
        sep,
        f"  {'Metric':<28}  {'Train':>10}  {'Val':>10}",
        "  " + "─" * 52,
    ]

    all_keys = sorted(set(list(train_metrics.keys()) + list(val_metrics.keys())))
    for key in all_keys:
        tr_val = train_metrics.get(key, "—")
        vl_val = val_metrics.get(key, "—")

        def _fmt(v: Any) -> str:
            if isinstance(v, list):
                return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        lines.append(f"  {key:<28}  {_fmt(tr_val):>10}  {_fmt(vl_val):>10}")

    lines.append(sep)
    logger.info("\n".join(lines))
