# Phase 2: The "Standard Benchmark" Baselines
import torch
import torch.nn as nn 
from torchvision.models import resnet18


# Resnet-18 model
class ResNet18Baseline(nn.Module):
    """
    Resnet-18 baseline for CIFAR-10.

    Architecture:
        Block 1: Conv(3→16) → BN → ReLU → Conv(16→32) → BN → ReLU → MaxPool
        Block 2: Conv(32→64) → BN → ReLU → Conv(64→128) → BN → ReLU → MaxPool
        Block 3: Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool
        Head:    Flatten → FC(4096→1024) → ReLU → Dropout(0.5) → FC(1024→512) → ReLU → FC(512→10)

    Input:  (B, 3, 32, 32)
    Output: (B, 100)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.backbone = resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 64,
            kernel_size = 3,
            stride=1,
            padding=1,
            bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc      = nn.Linear(512, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def get_baseline_model(num_classes: int = 10) -> ResNet18Baseline:
    return ResNet18Baseline(num_classes=num_classes)


if __name__ == "__main__":
    # Quick sanity check
    model = get_baseline_model()
    print("===== MODEL ARCHITECTURE =====")
    print(model)
    dummy = torch.zeros(4, 3, 32, 32)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")