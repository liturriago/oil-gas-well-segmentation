from typing import Tuple
import torch
import numpy as np
import webdataset as wds
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class SegmentationDataset:
    def __init__(
        self,
        shard_path: str,
        image_size: int = 256,
        augmentation: bool = False,
    ):
        self.shard_path = shard_path
        self.image_size = image_size
        self.augmentation = augmentation

        self.resize_img = T.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        self.resize_mask = T.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.NEAREST,
        )

        self.dataset = (
            wds.WebDataset(shard_path)
            .decode()
            .map(self.preprocess)
        )

    def normalize_per_channel(self, image: np.ndarray) -> np.ndarray:
        """
        Normalización por canal (clave para satélite + NIR)
        """
        image = image.astype(np.float32)

        for c in range(image.shape[2]):
            ch = image[:, :, c]
            ch_min = ch.min()
            ch_max = ch.max()
            image[:, :, c] = (ch - ch_min) / (ch_max - ch_min + 1e-6)

        return image

    def preprocess(self, sample):
        image = sample["rgb_nir.npy"]  # (H, W, 4)
        mask = sample["mask.npy"]      # (H, W)

        # Normalización correcta (por canal)
        image = self.normalize_per_channel(image)

        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # (4,H,W)
        mask = torch.from_numpy(mask).unsqueeze(0)        # (1,H,W)

        # Resize correcto
        image = self.resize_img(image)
        mask = self.resize_mask(mask)

        # Binarizar máscara (por seguridad)
        mask = (mask > 0.5).float()

        # Augmentations consistentes
        if self.augmentation:
            if torch.rand(1) > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            if torch.rand(1) > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

        return image, mask

    def get_loader(self, batch_size=8, num_workers=4):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )