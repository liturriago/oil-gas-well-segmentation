import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.dataset import SegmentationDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        augmentation: bool = True,
        image_size: int = 256,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.image_size = image_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SegmentationDataset(
                shard_path=self.train_path,
                augmentation=self.augmentation,
                image_size=self.image_size,
            )
            self.val_dataset = SegmentationDataset(
                shard_path=self.val_path, augmentation=False, image_size=self.image_size
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
