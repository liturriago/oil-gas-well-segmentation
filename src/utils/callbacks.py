import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor


def get_callbacks():
    """Returns standard callbacks for training."""
    checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-model-{epoch:02d}-{val_f1:.4f}",
        save_top_k=1,
        monitor="val_f1",
        mode="max",
    )

    early_stop = EarlyStopping(monitor="val_f1", patience=10, mode="max")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    return [checkpoint, early_stop, lr_monitor]
