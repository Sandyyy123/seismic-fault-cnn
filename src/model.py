"""
CNN architectures for seismic fault detection.

Two modes:
  - UNet: pixel-wise fault segmentation (preferred for full-section output)
  - ResNetClassifier: patch-level binary classification (fault present / absent)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two-layer conv block with BatchNorm and ReLU, used in U-Net encoder/decoder."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool + ConvBlock (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Bilinear upsample + skip concat + ConvBlock (decoder step)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatches from non-power-of-2 inputs
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net for fault segmentation
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net for pixel-wise seismic fault segmentation.

    Input:  (B, in_channels, H, W)  - seismic amplitude patch
    Output: (B, num_classes, H, W)  - per-pixel fault probability logits
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 64,
        depth: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.depth = depth

        # Encoder
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]
        self.enc0 = ConvBlock(in_channels, filters[0], dropout)
        self.downs = nn.ModuleList(
            [DownBlock(filters[i], filters[i + 1], dropout) for i in range(depth)]
        )

        # Decoder
        self.ups = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.ups.append(UpBlock(filters[i + 1] + filters[i + 1], filters[i], dropout))

        # Output head
        self.head = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.enc0(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        out = skips[-1]
        for i, up in enumerate(self.ups):
            out = up(out, skips[-(i + 2)])

        return self.head(out)


# ---------------------------------------------------------------------------
# ResNet-style patch classifier
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class ResNetClassifier(nn.Module):
    """
    Lightweight ResNet-style classifier for patch-level fault presence detection.

    Input:  (B, in_channels, H, W)
    Output: (B, num_classes) logits
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    arch = cfg["model"]["architecture"]
    if arch == "unet":
        return UNet(
            in_channels=cfg["model"]["in_channels"],
            num_classes=cfg["model"]["num_classes"],
            base_filters=cfg["model"]["base_filters"],
            depth=cfg["model"]["depth"],
            dropout=cfg["model"]["dropout"],
        )
    elif arch == "resnet_clf":
        return ResNetClassifier(
            in_channels=cfg["model"]["in_channels"],
            num_classes=cfg["model"]["num_classes"],
            dropout=cfg["model"]["dropout"],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
