import torch
from torchmetrics import MetricCollection
from torchmetrics.segmentation import DiceScore


class ThresholdWrapper(torch.nn.Module):
    """
    Convierte probabilidades (N,1,H,W) → etiquetas (N,H,W)
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, preds: torch.Tensor):
        # preds: (N,1,H,W)
        preds = (preds > self.threshold).long()
        preds = preds.squeeze(1)  # → (N,H,W)
        return preds


def get_metrics(
    num_classes: int = 2,
    threshold: float = 0.5,
    average: str = "micro",
):
    """
    MetricCollection para segmentación binaria enfocada en foreground.
    """

    threshold_fn = ThresholdWrapper(threshold)

    dice = DiceScore(
        num_classes=num_classes,
        include_background=False,  # 🔥 SOLO foreground
        average=average,
        input_format="index",  # usamos (N,H,W)
    )

    from torchmetrics import Metric

    class WrappedMetric(Metric):
        def __init__(self, metric):
            super().__init__()
            self.metric = metric

        def update(self, preds, target):
            preds = threshold_fn(preds)
            self.metric.update(preds, target)

        def compute(self):
            return self.metric.compute()

        def reset(self):
            self.metric.reset()
            super().reset()

    metrics = MetricCollection(
        {
            "dice_fg": WrappedMetric(dice),
        }
    )

    return metrics
