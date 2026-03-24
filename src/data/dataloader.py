"""
DataLoader factory with support for DDP node splitting and standard sampling.

Building the DataLoader separately from the dataset keeps the dataset module
reusable in non-distributed contexts (notebooks, debugging, testing).
"""

from __future__ import annotations

import torch
import webdataset as wds

from src.data.dataset import build_dataset


def build_dataloader(
    shard_path: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augmentation: bool = True,
    training: bool = True,
    shuffle_buffer: int = 1000,
    world_size: int = 1,
    rank: int = 0,
) -> torch.utils.data.DataLoader:
    """Build a :class:`torch.utils.data.DataLoader` over a WebDataset shard.

    ``wds.split_by_node`` inside :func:`build_dataset` handles DDP shard
    splitting automatically — no DistributedSampler needed.

    Args:
        shard_path:     Path or glob to ``.tar`` shards.
        image_size:     Target spatial resolution.
        batch_size:     Per-GPU batch size.
        num_workers:    Number of DataLoader worker processes.
        augmentation:   Enable random augmentations (training only).
        training:       Enables shuffle + augmentations.
        shuffle_buffer: WebDataset internal shuffle buffer size.
        world_size:     Total number of processes/GPUs (informational).
        rank:           Current process rank (informational).

    Returns:
        Configured :class:`torch.utils.data.DataLoader`.
    """
    dataset = build_dataset(
        shard_path=shard_path,
        image_size=image_size,
        augmentation=augmentation,
        training=training,
        shuffle_buffer=shuffle_buffer,
    )

    # WebDataset is an IterableDataset — batch + collate at the wds level
    batched_dataset = dataset.batched(batch_size, collation_fn=_collate_fn)

    loader = wds.WebLoader(
        batched_dataset,
        batch_size=None,  # already batched above
        shuffle=False,    # shuffle handled internally by wds
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return loader


def _collate_fn(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Stack a list of per-sample dicts into a batched dict of tensors.

    Args:
        samples: List of dicts with keys ``'image'`` and ``'mask'``,
                 each a :class:`torch.Tensor`.

    Returns:
        Dict with ``'image'``: (B, C, H, W) and ``'mask'``: (B, 1, H, W).
    """
    images = torch.stack([s["image"] for s in samples], dim=0)
    masks = torch.stack([s["mask"] for s in samples], dim=0)
    return {"image": images, "mask": masks}
