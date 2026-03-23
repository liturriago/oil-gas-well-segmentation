import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score, Dice


def get_metrics(num_classes: int = 1, threshold: float = 0.5, average: str = "micro"):
    """
    Returns a MetricCollection for training and validation.
    """
    if num_classes == 1:
        f1 = BinaryF1Score(threshold=threshold)
        dice = Dice(threshold=threshold, average=average)
    else:
        f1 = MulticlassF1Score(num_classes=num_classes, average=average)
        dice = Dice(num_classes=num_classes, average=average)

    return MetricCollection({"f1": f1, "dice": dice})
