![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
[![CI](https://img.shields.io/github/actions/workflow/status/liturriago/oil-gas-well-segmentation/ci.yaml?style=for-the-badge&logo=github)](https://github.com/liturriago/oil-gas-well-segmentation/actions/workflows/ci.yaml)

![Banner](./assets/banner.png)

# Oil & Gas Well Segmentation — Multispecral Satellite Imagery

> **Binary segmentation of oil/gas wells from 4-channel (RGB + NIR) satellite imagery using ResUNet with PyTorch pure DDP.**

---

## Features

| Capability | Implementation |
|---|---|
| Architecture | ResUNet (ResNet-{18,34,50} encoder, pretrained ImageNet) |
| Input | 4-channel RGB + NIR, patched to any resolution |
| Loss | Focal Loss (`torchvision.ops`) + Dice Loss (custom) |
| Metrics | Dice Score, Sensitivity, Specificity — per-class + macro |
| Distributed | `torch.nn.parallel.DistributedDataParallel` via `torchrun` |
| Mixed Precision | `torch.cuda.amp.autocast` + `GradScaler` |
| Configuration | Hydra + Pydantic v2 validation |
| Dataset format | WebDataset (`.bin` / `.tar` shards, `.npy` keys) |
| Augmentation | albumentations (paired image+mask transforms) |

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Prepare Data

Data must be WebDataset shards (`.bin` or `.tar`) where each sample has:

```
sample.image.npy   → float32 / uint16  (H, W, 4)  — RGB + NIR channels last
sample.mask.npy    → uint8 / int32     (H, W)      — binary labels {0, 1}
```

Update paths in `configs/config.yaml`:

```yaml
data:
  train_path: data/train.bin
  val_path:   data/val.bin
```

### 3. Single-GPU Training

```bash
python train.py
```

### 4. Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=2 train.py
```

### 5. Override Config on the Fly (Hydra)

```bash
python train.py training.lr=5e-4 training.epochs=100 model.encoder=resnet50
```

---

## Configuration Reference

Edit `configs/config.yaml` to control every aspect of training:

```yaml
training:
  lr: 1.0e-3        # Peak learning rate
  batch_size: 8     # Per-GPU batch size
  epochs: 50
  optimizer: adam   # adam | adamw | sgd
  scheduler: cosine # cosine | step | none
  use_amp: true     # Mixed precision (AMP)
  use_ddp: true     # Distributed training
  num_gpus: 2
  grad_clip: 1.0    # Max gradient norm (0 = disabled)
  warmup_epochs: 5  # Linear warmup epochs

model:
  in_channels: 4    # RGB + NIR
  out_channels: 1   # Binary segmentation
  encoder: resnet34 # resnet18 | resnet34 | resnet50

loss:
  focal_alpha: 0.75
  focal_gamma: 2.0
  dice_weight: 1.0
  focal_weight: 1.0

data:
  image_size: 256
  augmentation: true
  mean: [0.485, 0.456, 0.406, 0.35]  # Per-channel (NOT ImageNet global)
  std:  [0.229, 0.224, 0.225, 0.15]

metrics:
  threshold: 0.5   # Sigmoid threshold for binarisation

system:
  seed: 42
  num_workers: 4
  checkpoint_dir: checkpoints
```

---

## Metrics

All metrics are implemented from scratch (no `torchmetrics`).

| Metric | Formula | Class |
|---|---|---|
| **Dice Score** | 2·TP / (2·TP + FP + FN) | per-class + macro |
| **Sensitivity** | TP / (TP + FN) | per-class + macro |
| **Specificity** | TN / (TN + FP) | per-class + macro |

Example output:

```python
{
  "dice_macro": 0.821,
  "dice_per_class": [0.943, 0.699],          # [background, foreground]
  "sensitivity_macro": 0.814,
  "sensitivity_per_class": [0.901, 0.727],
  "specificity_macro": 0.889,
  "specificity_per_class": [0.802, 0.976],
}
```

---

## Checkpoints

Checkpoints are saved to `checkpoints/` (configurable):

| File | Description |
|---|---|
| `best.pt` | Best model by **foreground Dice** |
| `last.pt` | Most recent epoch |

Loading a checkpoint:

```python
from src.utils.checkpoint import load_checkpoint
from src.models.resunet import ResUNet

model = ResUNet(in_channels=4, out_channels=1)
ckpt = load_checkpoint("checkpoints/best.pt", model)
print(f"Loaded epoch {ckpt['epoch']}")
```

---

## Running Tests

```bash
pytest                       # all tests
pytest tests/test_metrics.py # specific file
pytest -v --tb=short         # verbose
```

---

## Model Architecture

```
Input (N, 4, H, W)
       │
   [Encoder — ResNet34]
   └── s0: (N,  64, H/2,  W/2)   stem
   └── s1: (N,  64, H/4,  W/4)   layer1
   └── s2: (N, 128, H/8,  W/8)   layer2
   └── s3: (N, 256, H/16, W/16)  layer3
   └──  b: (N, 512, H/32, W/32)  bottleneck (layer4)
       │
   [Decoder — Bilinear Upsample + Residual Blocks]
   └── d3: (N, 256, H/16, W/16)  + skip s3
   └── d2: (N, 128, H/8,  W/8)   + skip s2
   └── d1: (N,  64, H/4,  W/4)   + skip s1
   └── d0: (N,  32, H/2,  W/2)   + skip s0
       │
   [Head]
   └── out: (N, 1, H, W) — raw logits (no sigmoid)
```

The NIR channel weight is initialised from the mean of the 3 pretrained RGB
weights to preserve ImageNet pretraining benefits.

---

## License

MIT
