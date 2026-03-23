import torch
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle cases where inputs have different size (e.g. padding adjustments)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            import torch.nn.functional as F

            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, backbone_name="resnet34", pretrained=True):
        super().__init__()

        # Encoder (ResNet)
        # Using a generalized initialization for different ResNet models
        backbone_func = getattr(models, backbone_name)
        # Weights attribute depends on torchvision version, fallback to pretrained
        try:
            self.encoder = backbone_func(weights="DEFAULT" if pretrained else None)
        except TypeError:
            self.encoder = backbone_func(pretrained=pretrained)

        if in_channels != 3:
            # Replace first conv if in_channels is not 3
            old_conv = self.encoder.conv1
            self.encoder.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        self.enc1 = nn.Sequential(
            self.encoder.conv1, self.encoder.bn1, self.encoder.relu
        )  # -> skip1 (64 channels)
        self.enc2 = nn.Sequential(
            self.encoder.maxpool, self.encoder.layer1
        )  # -> skip2 (64 channels)
        self.enc3 = self.encoder.layer2  # -> skip3 (128 channels)
        self.enc4 = self.encoder.layer3  # -> skip4 (256 channels)
        self.enc5 = self.encoder.layer4  # -> bottleneck (512 channels)

        # ResNet34 channels for reference:
        # layer1: 64, layer2: 128, layer3: 256, layer4: 512

        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        # Final output layer
        self.up_final = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # Decoder
        d4 = self.dec4(x5, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        # Final upsampling
        out = self.up_final(d1)
        out = self.out_conv(out)

        return out
