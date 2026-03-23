import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from src.config.schema import Config
from src.data.datamodule import SegmentationDataModule
from src.training.lightning_module import SegmentationModule
from src.utils.callbacks import get_callbacks


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg_dict: DictConfig):
    # Convert DictConfig to a standard dict and validate with Pydantic
    cfg_python = OmegaConf.to_container(cfg_dict, resolve=True)
    cfg = Config(**cfg_python)
    
    # Set seed
    pl.seed_everything(cfg.system.seed)
    
    # Initialize DataModule
    datamodule = SegmentationDataModule(
        train_path=cfg.data.train_path,
        val_path=cfg.data.val_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.system.num_workers,
        augmentation=cfg.data.augmentation,
        in_channels=cfg.model.in_channels
    )
    
    # Initialize LightningModule
    model = SegmentationModule(cfg)
    
    # Setup Trainer configs
    strategy = "ddp" if cfg.training.use_ddp else "auto"
    precision = "16-mixed" if cfg.training.use_amp else 32
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices=cfg.training.num_gpus if cfg.training.num_gpus > 0 else "auto",
        strategy=strategy,
        precision=precision,
        callbacks=get_callbacks()
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
