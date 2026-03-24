"""
DataLoader factory — single GPU, no distributed sampling.
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
) -> torch.utils.data.DataLoader:
    """Build a :class:`torch.utils.data.DataLoader` over a WebDataset shard.

    Args:
        shard_path:     Path or glob to ``.tar`` / ``.bin`` shards.
        image_size:     Target spatial resolution.
        batch_size:     Batch size.
        num_workers:    Number of DataLoader worker processes.
        augmentation:   Enable random augmentations (training only).
        training:       Enables shuffle + augmentations.
        shuffle_buffer: WebDataset internal shuffle buffer size.

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

    batched_dataset = dataset.batched(batch_size, collation_fn=_collate_fn)

    loader = wds.WebLoader(
        batched_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return loader


def _collate_fn(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Stack per-sample dicts into a batched dict of tensors."""
    images = torch.stack([s["image"] for s in samples], dim=0)
    masks = torch.stack([s["mask"] for s in samples], dim=0)
    return {"image": images, "mask": masks}
