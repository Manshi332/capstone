"""
custom_cnn.py — Custom CNN architecture for ASL recognition.

"""

import torch
import torch.nn as nn
from torchvision import models


# ─── Convolutional Block ──────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BN → ReLU (× repeats) → MaxPool → Dropout"""

    def __init__(self, in_ch: int, out_ch: int, repeats: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(repeats):
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ─── Custom CNN ───────────────────────────────────────────────────────────────

class CustomCNN(nn.Module):

    def __init__(self, num_classes: int = 24, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32,  repeats=2, dropout=0.10),   # → 32×32
            ConvBlock(32,  64,  repeats=2, dropout=0.15),   # → 16×16
            ConvBlock(64,  128, repeats=3, dropout=0.20),   # →  8×8
            ConvBlock(128, 256, repeats=3, dropout=0.20),   # →  4×4
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # → (B, 256, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# ─── Deeper Variant ───────────────────────────────────────────────────────────

class DeepCustomCNN(nn.Module):

    class ResBlock(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(ch)
            self.relu  = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + x)

    def __init__(self, num_classes: int = 24, dropout: float = 0.4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            self.ResBlock(32),
            nn.MaxPool2d(2), nn.Dropout2d(0.05),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            self.ResBlock(64),
            nn.MaxPool2d(2), nn.Dropout2d(0.08),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            self.ResBlock(128),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def ResBlock(self, ch):
        return DeepCustomCNN.ResBlock(ch) if False else _ResBlock(ch)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return self.head(x)


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)




def build_custom_cnn(variant: str = "standard", num_classes: int = 24) -> nn.Module:
    if variant == "deep":
        return DeepCustomCNN(num_classes=num_classes)
    return CustomCNN(num_classes=num_classes)


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    model = CustomCNN(num_classes=24)
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {params:,}")