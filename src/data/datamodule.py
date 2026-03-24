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
                shuffle=True,
            )
            self.val_dataset = SegmentationDataset(
                shard_path=self.val_path,
                augmentation=False,
                image_size=self.image_size,
                shuffle=False,
            )

    def train_dataloader(self):
        return self.train_dataset.get_loader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return self.val_dataset.get_loader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
