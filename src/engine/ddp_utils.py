"""
DDP (DistributedDataParallel) setup and teardown utilities.

Provides a clean interface for:
  - Initialising the process group (NCCL backend on GPU, GLOO on CPU)
  - Setting per-rank device, seed, and cuDNN flags
  - Wrapping a model in DDP
  - Tearing down the process group
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(rank: int, world_size: int, seed: int = 42) -> None:
    """Initialise the default NCCL process group.

    Called once per GPU process.  Expects ``MASTER_ADDR`` and ``MASTER_PORT``
    environment variables to be set by the launcher (``torchrun`` handles this
    automatically).

    Args:
        rank:       Index of this GPU process (0 … world_size-1).
        world_size: Total number of GPU processes.
        seed:       Base random seed; per-rank offset is added automatically.
    """
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.cuda.set_device(rank)   # must be set before init_process_group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}") if torch.cuda.is_available() else None,
    )

    # Reproducibility: offset seed per rank so ranks explore different samples
    _set_seed(seed + rank)

    # Deterministic cuDNN for reproducibility (slight speed trade-off)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def teardown_ddp() -> None:
    """Cleanly destroy the default process group after training completes."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model: nn.Module, rank: int) -> DDP:
    """Move *model* to the current GPU and wrap it in :class:`DDP`.

    Args:
        model: The model to wrap (must already have parameters on CPU or target GPU).
        rank:  The GPU rank / device index.

    Returns:
        DDP-wrapped model with ``find_unused_parameters=False`` for efficiency.
    """
    model = model.to(rank)
    return DDP(model, device_ids=[rank], find_unused_parameters=False)


def is_main_process() -> bool:
    """Return ``True`` if this is the rank-0 (main) process.

    Works correctly whether or not DDP is initialised; falls back to ``True``
    when running in a single-GPU context.
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Return the current process rank (0 if DDP not initialised)."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Return total number of processes (1 if DDP not initialised)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Synchronise all processes at this point (no-op if not distributed)."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """All-reduce a scalar tensor across all ranks.

    Args:
        tensor:  The local scalar tensor to reduce.
        average: If ``True``, divide the reduced sum by world_size.

    Returns:
        Reduced tensor (same device as input).
    """
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= get_world_size()
    return rt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
