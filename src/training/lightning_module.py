import torch
import torch.optim as optim
import pytorch_lightning as pl

from src.models.resunet import ResUNet
from src.losses.combined_loss import CombinedLoss
from src.metrics.metrics import get_metrics


class SegmentationModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Model
        self.model = ResUNet(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            backbone_name=cfg.model.name,
            pretrained=True,
        )

        # Loss
        self.criterion = CombinedLoss(
            focal_alpha=cfg.loss.focal_alpha,
            focal_gamma=cfg.loss.focal_gamma,
            dice_weight=cfg.loss.dice_weight,
            focal_weight=cfg.loss.focal_weight,
        )

        # Metrics
        num_classes = cfg.model.out_channels
        self.train_metrics = get_metrics(
            num_classes, threshold=cfg.metrics.threshold, average=cfg.metrics.average
        )
        self.val_metrics = self.train_metrics.clone()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.cfg.training.use_ddp,
        )

        # Calculate metrics
        preds = torch.sigmoid(logits)
        self.train_metrics(preds, y.long() if self.cfg.model.out_channels > 1 else y.int())
        self.log_dict(
            {f"train_{k}": v for k, v in self.train_metrics.compute().items()},
            on_step=False,
            on_epoch=True,
            sync_dist=self.cfg.training.use_ddp,
        )

        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.cfg.training.use_ddp,
        )

        # Calculate metrics
        preds = torch.sigmoid(logits)
        self.val_metrics(preds, y.long() if self.cfg.model.out_channels > 1 else y.int())
        self.log_dict(
            {f"val_{k}": v for k, v in self.val_metrics.compute().items()},
            on_step=False,
            on_epoch=True,
            sync_dist=self.cfg.training.use_ddp,
        )

        return loss

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def configure_optimizers(self):
        # Optimizer
        if self.cfg.training.optimizer.lower() == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.cfg.training.lr)
        elif self.cfg.training.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.cfg.training.lr)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.cfg.training.lr, momentum=0.9)

        # Scheduler
        if self.cfg.training.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.training.epochs
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
