import os
import glob
from typing import Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

# Trying multiple imports since opencv-python-headless may be used
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


class SegmentationDataset(Dataset):
    def __init__(self, data_path: str, augmentation: bool = False, in_channels: int = 3):
        self.data_path = data_path
        self.augmentation = augmentation
        self.in_channels = in_channels

        # Assuming a structure like: data_path/images/*.png and data_path/masks/*.png
        self.image_paths = sorted(glob.glob(os.path.join(data_path, "images", "*.*")))
        self.mask_paths = sorted(glob.glob(os.path.join(data_path, "masks", "*.*")))

        # If no images found, create dummy mode (useful for testing when data isn't present)
        self.dummy_mode = len(self.image_paths) == 0
        if self.dummy_mode:
            self.length = 100
        else:
            self.length = len(self.image_paths)
            assert len(self.image_paths) == len(
                self.mask_paths
            ), "Mismatched images and masks count"

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dummy_mode:
            # Return dummy random tensors for testing
            img = torch.randn(self.in_channels, 256, 256)
            mask = torch.randint(0, 2, (1, 256, 256), dtype=torch.float32)
            return img, mask

        if cv2 is None or np is None:
            raise RuntimeError(
                "cv2 and numpy are required. Install with `pip install opencv-python-headless numpy`"
            )

        # Basic loading
        img = cv2.imread(self.image_paths[idx])
        if self.in_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.in_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Normalization
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Binarize mask
        mask = (mask > 0.5).astype(np.float32)

        # Basic ToTensor conversion (HWC -> CHW)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        if self.augmentation:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                img_tensor = F.hflip(img_tensor)
                mask_tensor = F.hflip(mask_tensor)

            # Random vertical flip
            if torch.rand(1) > 0.5:
                img_tensor = F.vflip(img_tensor)
                mask_tensor = F.vflip(mask_tensor)

        return img_tensor, mask_tensor
