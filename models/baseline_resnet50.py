import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNet50Baseline(nn.Module):
    """
    ResNet-50 baseline adapted for CIFAR-100 (32x32 images).

    CIFAR adaptations (same as ResNet-18 variant):
      - conv1: 7x7 stride-2 → 3x3 stride-1 (preserves spatial resolution)
      - maxpool: replaced with Identity (avoids 8x downscale on 32x32 input)
      - fc: 2048 → num_classes

    Optional dropout inserted between avgpool and fc.
    dropout_rate=0.0 disables it entirely (default — preserves existing behaviour).
    dropout_rate=0.2 is a reasonable starting point for CIFAR-100.

    Dropout is active only during model.train() — automatically off at model.eval().

    Input:  (B, 3, 32, 32)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 100, dropout_rate: float = 0.0):
        super().__init__()
        self.backbone = resnet50(weights=None)

        # Adapt stem for 32x32 CIFAR images
        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.backbone.maxpool = nn.Identity()

        # Replace fc with dropout + linear
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_baseline_model(
    num_classes: int = 100, dropout_rate: float = 0.0
) -> ResNet50Baseline:
    return ResNet50Baseline(num_classes=num_classes, dropout_rate=dropout_rate)


if __name__ == "__main__":
    model = get_baseline_model(num_classes=100)
    print("===== MODEL ARCHITECTURE =====")
    print(model)
    dummy = torch.zeros(4, 3, 32, 32)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # should be (4, 100)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
