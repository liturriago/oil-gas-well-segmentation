"""
ResUNet: U-Net with a pretrained ResNet encoder.

Architecture overview:
  Encoder: ResNet-{18,34,50} backbone (first conv adapted to 4-channel input).
  Bottleneck: last ResNet stage.
  Decoder: 4 upsampling blocks with skip connections.
  Head: 1×1 conv → raw logits (no sigmoid).

Input shape:  (N, in_channels, H, W)
Output shape: (N, out_channels, H, W)  — logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


# ---------------------------------------------------------------------------
# Helper blocks
# ---------------------------------------------------------------------------


class ConvBnRelu(nn.Sequential):
    """3×3 Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """Pre-activation residual block used inside the decoder."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.shortcut(x))


class DecoderBlock(nn.Module):
    """Bilinear upsample → concat skip → residual refinement."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ResidualBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against off-by-one spatial mismatches
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------


_ENCODER_CONFIGS: dict[str, tuple] = {
    "resnet18": (tvm.resnet18, ResNet18_Weights.IMAGENET1K_V1, [64, 64, 128, 256, 512]),
    "resnet34": (tvm.resnet34, ResNet34_Weights.IMAGENET1K_V1, [64, 64, 128, 256, 512]),
    "resnet50": (tvm.resnet50, ResNet50_Weights.IMAGENET1K_V2, [64, 256, 512, 1024, 2048]),
}


def _build_encoder(name: str, in_channels: int) -> tuple[nn.Module, list[int]]:
    """Instantiate a pretrained ResNet encoder adapted to *in_channels* inputs.

    The first convolutional layer is replaced to accept *in_channels* inputs.
    Pre-trained weights are copied for the first 3 channels; the additional
    NIR channel is initialized by averaging the RGB weights across channels.

    Returns:
        (encoder_module, channel_widths)  where channel_widths has 5 entries
        corresponding to the spatial scales produced by the encoder.
    """
    if name not in _ENCODER_CONFIGS:
        raise ValueError(f"Unknown encoder '{name}'. Choose from {list(_ENCODER_CONFIGS)}")

    builder, weights, channels = _ENCODER_CONFIGS[name]
    backbone: tvm.ResNet = builder(weights=weights)

    # --- Adapt first conv to in_channels ---
    old_conv: nn.Conv2d = backbone.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        # Copy RGB weights
        new_conv.weight[:, :3, ...] = old_conv.weight[:, :3, ...]
        # Extra channels: average of existing weights
        if in_channels > 3:
            extra = old_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, in_channels):
                new_conv.weight[:, i : i + 1, ...] = extra
    backbone.conv1 = new_conv

    return backbone, channels


# ---------------------------------------------------------------------------
# ResUNet
# ---------------------------------------------------------------------------


class ResUNet(nn.Module):
    """ResUNet segmentation model.

    Args:
        in_channels:  Number of input image channels (e.g. 4 for RGB+NIR).
        out_channels: Number of output segmentation classes.
        encoder:      ResNet variant to use as encoder backbone.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        encoder: str = "resnet34",
    ) -> None:
        super().__init__()

        backbone, enc_channels = _build_encoder(encoder, in_channels)

        # ----- Encoder stages -----
        # Stage 0: stem (conv1 + bn + relu) → (B, 64, H/2, W/2)
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        # Pool: → (B, 64, H/4, W/4)
        self.pool = backbone.maxpool
        # Stage 1: layer1 → (B, ch1, H/4, W/4)
        self.enc1 = backbone.layer1
        # Stage 2: layer2 → (B, ch2, H/8, W/8)
        self.enc2 = backbone.layer2
        # Stage 3: layer3 → (B, ch3, H/16, W/16)
        self.enc3 = backbone.layer3
        # Stage 4 (bottleneck): layer4 → (B, ch4, H/32, W/32)
        self.bottleneck = backbone.layer4

        e = enc_channels  # shorthand
        # ----- Decoder stages -----
        # dec3: bottleneck (e[4]) × skip enc3 (e[3])
        self.dec3 = DecoderBlock(e[4], e[3], 256)
        # dec2: 256 × skip enc2 (e[2])
        self.dec2 = DecoderBlock(256, e[2], 128)
        # dec1: 128 × skip enc1 (e[1])
        self.dec1 = DecoderBlock(128, e[1], 64)
        # dec0: 64 × skip enc0 (e[0])  — back to H/2 resolution
        self.dec0 = DecoderBlock(64, e[0], 32)

        # Final upsample ×2 → original resolution
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Segmentation head
        self.head = nn.Conv2d(32, out_channels, kernel_size=1, bias=True)

        # Weight initialization for decoder + head
        self._init_decoder_weights()

    # ------------------------------------------------------------------
    def _init_decoder_weights(self) -> None:
        for module in [self.dec3, self.dec2, self.dec1, self.dec0, self.head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, in_channels, H, W).

        Returns:
            Logit tensor of shape (N, out_channels, H, W).
            **No sigmoid is applied** — use with BCEWithLogitsLoss or
            the combined focal+dice loss.
        """
        # Encoder
        s0 = self.enc0(x)           # (N, 64,   H/2,  W/2)
        s1 = self.enc1(self.pool(s0))  # (N, 64,   H/4,  W/4)
        s2 = self.enc2(s1)          # (N, 128,  H/8,  W/8)
        s3 = self.enc3(s2)          # (N, 256,  H/16, W/16)
        b = self.bottleneck(s3)     # (N, 512,  H/32, W/32)

        # Decoder
        d3 = self.dec3(b, s3)       # (N, 256,  H/16, W/16)
        d2 = self.dec2(d3, s2)      # (N, 128,  H/8,  W/8)
        d1 = self.dec1(d2, s1)      # (N, 64,   H/4,  W/4)
        d0 = self.dec0(d1, s0)      # (N, 32,   H/2,  W/2)

        out = self.final_up(d0)     # (N, 32,   H,    W)
        return self.head(out)       # (N, out_channels, H, W)
