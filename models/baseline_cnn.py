"""
models/baseline_cnn.py
Custom CNN baseline for CIFAR-10 classification.
Architecture: 6 conv layers with BatchNorm + 3 FC layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class CIFAR10Baseline(nn.Module):
    """
    Simple CNN baseline for CIFAR-10.

    Architecture:
        Block 1: Conv(3→16) → BN → ReLU → Conv(16→32) → BN → ReLU → MaxPool
        Block 2: Conv(32→64) → BN → ReLU → Conv(64→128) → BN → ReLU → MaxPool
        Block 3: Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool
        Head:    Flatten → FC(4096→1024) → ReLU → Dropout(0.5) → FC(1024→512) → ReLU → FC(512→10)

    Input:  (B, 3, 32, 32)
    Output: (B, 10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3,  16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), # inplace = True -> modifies the input tensor directly, rather than creating a new output tensor
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32x32 → 16x16
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32,  64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16x16 → 8x8
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 8x8 → 4x4
        )

        # ── Classifier Head ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def get_baseline_model(num_classes: int = 10, dropout: float = 0.5) -> CIFAR10Baseline:
    return CIFAR10Baseline(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    # Quick sanity check
    model = get_baseline_model()
    dummy = torch.zeros(4, 3, 32, 32)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")