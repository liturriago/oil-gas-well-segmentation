"""
WebDataset-based dataset for multispecral satellite image segmentation.

Expected shard format (.tar with wds-compatible keys), matching the Alberta Wells
dataset generation script:
  sample.rgb_nir.npy  →  uint16 array (H, W, 4)  — R, G, B, NIR channels last
  sample.mask.npy     →  uint16 array (H, W)      — class labels

WebDataset's default `.decode()` automatically deserialises .npy files into
numpy arrays, so no manual byte parsing is needed.

The dataset applies:
  1. uint16 → float32 rescaling to [0, 1]  (divide by 65535)
  2. Deterministic resize (image: bilinear, mask: nearest)
  3. Optional paired augmentations via albumentations
"""

from __future__ import annotations

from typing import Any, Callable

import albumentations as A
import numpy as np
import torch
import webdataset as wds


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------


def build_augmentation(image_size: int, training: bool = True) -> A.Compose:
    """Return an albumentations pipeline for paired image+mask transforms.

    Args:
        image_size: Target spatial size (square resize).
        training:   If *True*, include random flip/rotation augmentations.

    Returns:
        An :class:`albumentations.Compose` transform that expects keyword
        arguments ``image`` (H, W, C float32) and ``mask`` (H, W int32).
    """
    if training:
        transforms = [
            A.Resize(image_size, image_size, interpolation=1),  # bilinear for image
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
    else:
        transforms = [
            A.Resize(image_size, image_size, interpolation=1),
        ]

    return A.Compose(transforms, additional_targets={"mask": "mask"})


# ---------------------------------------------------------------------------
# Sample decoder
# ---------------------------------------------------------------------------

#: uint16 max value used to rescale images to [0, 1].
_UINT16_MAX: float = 65535.0


def _make_sample_decoder(
    image_size: int,
    augmentation: bool,
    training: bool,
) -> Callable[[dict[str, Any]], dict[str, torch.Tensor]]:
    """Build a stateful sample-processing callable.

    After ``wds.WebDataset(...).decode()``, each sample dict already contains
    numpy arrays for ``.npy`` keys — no manual byte parsing needed.

    Expected keys (set by the Alberta Wells dataset generation script):
      ``"rgb_nir.npy"``  → uint16 (H, W, 4)
      ``"mask.npy"``     → uint16 (H, W)

    Processing pipeline per sample:
      1. Extract arrays from decoded sample dict.
      2. Ensure image is HWC layout.
      3. Rescale uint16 → float32 in [0, 1].
      4. Apply paired augmentations (albumentations).
      5. Convert to CHW torch.Tensor.
    """
    aug = build_augmentation(image_size, training=training and augmentation)

    def process(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        # ------------------------------------------------------------------
        # 1. Extract arrays — wds.decode() already returned numpy arrays.
        #    Search by suffix to handle the "__key__" prefix wds may prepend
        #    (e.g. "sample_001.rgb_nir.npy").
        # ------------------------------------------------------------------
        image_key = next(
            (k for k in sample if k.endswith("rgb_nir.npy")), None
        )
        mask_key = next(
            (k for k in sample if k.endswith("mask.npy")), None
        )

        if image_key is None:
            raise KeyError(
                f"Expected a key ending in 'rgb_nir.npy', got: {list(sample.keys())}"
            )
        if mask_key is None:
            raise KeyError(
                f"Expected a key ending in 'mask.npy', got: {list(sample.keys())}"
            )

        image: np.ndarray = sample[image_key]   # uint16, (H, W, 4) or (4, H, W)
        mask: np.ndarray = sample[mask_key]      # uint16, (H, W)

        # ------------------------------------------------------------------
        # 2. Ensure HWC layout (guard for CHW-stored arrays)
        # ------------------------------------------------------------------
        if image.ndim == 3 and image.shape[0] < image.shape[2]:
            image = image.transpose(1, 2, 0)     # CHW → HWC

        # ------------------------------------------------------------------
        # 3. uint16 → float32 in [0, 1]
        # ------------------------------------------------------------------
        image = image.astype(np.float32) / _UINT16_MAX   # (H, W, 4), [0, 1]
        # Binarize: multi_class_seg_maps has values 0=background, 1+=well type.
        # The model head is binary (out_channels=1), so collapse all well classes
        # into a single foreground label. This also prevents focal loss NaN:
        # sigmoid_focal_loss requires targets in [0, 1]; values > 1 cause the
        # BCE gradient to scale by the target value, leading to explosion in fp16.
        mask = (mask > 0).astype(np.int32)               # 0=background, 1=well

        # ------------------------------------------------------------------
        # 4. Augment — albumentations applies nearest-neighbour to masks
        #    automatically when passed via additional_targets={"mask": "mask"}
        # ------------------------------------------------------------------
        result = aug(image=image, mask=mask)
        image = result["image"]   # (H', W', 4) float32
        mask = result["mask"]     # (H', W')    int32

        # ------------------------------------------------------------------
        # 5. HWC → CHW tensor
        # ------------------------------------------------------------------
        image_t = torch.from_numpy(image.transpose(2, 0, 1))          # (4, H, W)
        mask_t = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0) # (1, H, W)

        return {"image": image_t, "mask": mask_t}

    return process


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_dataset(
    shard_path: str,
    image_size: int,
    augmentation: bool = True,
    training: bool = True,
    shuffle_buffer: int = 1000,
) -> wds.WebDataset:
    """Build a :class:`webdataset.WebDataset` for the Alberta Wells dataset.

    Args:
        shard_path:     Path (or glob) to ``.tar`` shard file(s).
        image_size:     Target spatial resolution (H == W after resize).
        augmentation:   Whether to apply random augmentations (training only).
        training:       Enables shuffling when *True*.
        shuffle_buffer: WebDataset internal shuffle buffer size.

    Returns:
        An iterable :class:`webdataset.WebDataset`.

    Notes:
        - Shuffle is applied **before** ``.decode()`` so samples are shuffled
          at the compressed-byte level, reducing memory overhead.
        - ``nodesplitter=None`` means every process reads all shards (correct
          behaviour for single-GPU training with one shard file).
    """
    decoder = _make_sample_decoder(image_size, augmentation, training)

    dataset = wds.WebDataset(
        shard_path,
        nodesplitter=None,    # let every GPU read all shards; use N shards (one
        shardshuffle=False,   # per GPU) for strict partitioning. shuffle() below
        empty_check=False,    # handles sample-level randomisation.
    )

    if training:
        # Shuffle before decode: more efficient (bytes shuffled, not arrays)
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.decode().map(decoder)

    return dataset
