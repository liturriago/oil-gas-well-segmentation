"""
DataLoader factory — single GPU, no distributed sampling.
"""

from __future__ import annotations

import torch
import webdataset as wds

from src.data.dataset import build_dataset


def _collate_fn(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Stack per-sample dicts into a batched dict of tensors.

    Args:
        samples: List of dicts with keys ``'image'`` (C, H, W) and
                 ``'mask'`` (1, H, W).

    Returns:
        Dict with ``'image'``: (B, C, H, W) and ``'mask'``: (B, 1, H, W).
    """
    images = torch.stack([s["image"] for s in samples], dim=0)
    masks = torch.stack([s["mask"] for s in samples], dim=0)
    return {"image": images, "mask": masks}


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
        image_size:     Target spatial resolution (H == W).
        batch_size:     Batch size. The last batch may be smaller than this
                        (``partial=True``).
        num_workers:    Number of DataLoader worker processes.
        augmentation:   Enable random augmentations (ignored when training=False).
        training:       Enables internal shuffle when *True*.
        shuffle_buffer: WebDataset shuffle buffer size (samples, not bytes).

    Returns:
        Configured :class:`wds.WebLoader`.
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
        batch_size=None,              # already batched above
        shuffle=False,                # shuffle handled by wds internally
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return loader
