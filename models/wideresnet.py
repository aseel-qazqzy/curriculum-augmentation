"""
models/wideresnet.py

Wide Residual Networks for CIFAR-10 / CIFAR-100.

Exact architecture used in Cubuk et al. (2020) RandAugment paper for CIFAR:
    WideResNet-28-10 with dropout=0.3

Reference:
    Zagoruyko, S., & Komodakis, N. (2016).
    Wide residual networks.
    British Machine Vision Conference (BMVC).
    https://arxiv.org/abs/1605.07146

Used in RandAugment paper (Table 2, CIFAR-10):
    WideResNet-28-10 + Cutout baseline error: 3.7%
    WideResNet-28-10 + RandAugment error:    2.7%  (N=2, M=9)

Architecture (depth=28, widen_factor=10):
    Input:   (B, 3, 32, 32)
    Conv1:   3 → 16 channels, 3×3
    Group 1: 16 → 160 channels, 4 wide blocks, stride=1
    Group 2: 160 → 320 channels, 4 wide blocks, stride=2
    Group 3: 320 → 640 channels, 4 wide blocks, stride=2
    BN → ReLU → AvgPool → FC(640 → num_classes)
    Output:  (B, num_classes)

    Total parameters (depth=28, k=10): ~36.5M

Block structure (pre-activation style):
    BN → ReLU → Conv(3×3) → Dropout → BN → ReLU → Conv(3×3)
    Shortcut: 1×1 Conv if dimensions change, Identity otherwise

Usage:
    model = get_wideresnet(depth=28, widen_factor=10, num_classes=10)
    model = get_wideresnet(depth=28, widen_factor=10, num_classes=100, dropout=0.3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# WIDE RESIDUAL BLOCK
class WideBlock(nn.Module):
    """
    Wide residual block with pre-activation (BN → ReLU → Conv).

    Pre-activation ordering (He et al., 2016 identity mapping paper) is
    standard for WideResNet — it allows the residual path to be
    an identity mapping, improving gradient flow.

    Structure:
        BN → ReLU → Conv(3×3, in→out, stride) → Dropout → BN → ReLU → Conv(3×3, out→out)
        + shortcut: 1×1 Conv(in→out, stride) if dimensions differ, else Identity
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout: float = 0.3):
        super().__init__()

        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.drop  = nn.Dropout(p=dropout)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # Shortcut — only needed when channels change or stride > 1
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.conv1(F.relu(self.bn1(x), inplace=True)))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)



# WIDE RESNET
class WideResNet(nn.Module):
    """
    Wide Residual Network (Zagoruyko & Komodakis, 2016).

    Default configuration matches Cubuk et al. (2020) RandAugment paper:
        depth=28, widen_factor=10, dropout=0.3

    Args:
        depth:        Total network depth. Must satisfy (depth - 4) % 6 == 0.
                      Common values: 16, 22, 28, 34, 40.
        widen_factor: Channel multiplier k. Common values: 1, 2, 4, 8, 10.
                      depth=28, k=10 → WRN-28-10 (paper's CIFAR model).
        dropout:      Dropout rate inside each wide block. Paper uses 0.3.
        num_classes:  Output classes. 10 for CIFAR-10, 100 for CIFAR-100.

    Layer widths: [16, 16·k, 32·k, 64·k]
        WRN-28-10: [16, 160, 320, 640]
    """

    def __init__(self, depth: int = 28, widen_factor: int = 10,
                 dropout: float = 0.3, num_classes: int = 10):
        super().__init__()

        assert (depth - 4) % 6 == 0, \
            f"depth must be 6n+4 (e.g. 16, 22, 28, 34, 40). Got {depth}."
        n = (depth - 4) // 6     # blocks per group
        k = widen_factor

        widths = [16, 16 * k, 32 * k, 64 * k]

        self.conv1  = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.group1 = self._make_group(widths[0], widths[1], n, stride=1, dropout=dropout)
        self.group2 = self._make_group(widths[1], widths[2], n, stride=2, dropout=dropout)
        self.group3 = self._make_group(widths[2], widths[3], n, stride=2, dropout=dropout)
        self.bn     = nn.BatchNorm2d(widths[3])
        self.fc     = nn.Linear(widths[3], num_classes)

        self._init_weights()

    def _make_group(self, in_ch: int, out_ch: int, n: int,
                    stride: int, dropout: float) -> nn.Sequential:
        """Build one group of n wide blocks."""
        blocks = [WideBlock(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(1, n):
            blocks.append(WideBlock(out_ch, out_ch, stride=1, dropout=dropout))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        """Kaiming normal init for Conv, constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)



# FACTORY FUNCTIONS
def get_wideresnet(depth: int = 28, widen_factor: int = 10,
                   dropout: float = 0.3, num_classes: int = 10) -> WideResNet:
    """
    Get WideResNet model.

    Paper configurations:
        WRN-28-10: depth=28, widen_factor=10  (36.5M params, CIFAR SOTA)
        WRN-16-8:  depth=16, widen_factor=8   (11.0M params, faster)
        WRN-40-2:  depth=40, widen_factor=2   (2.2M params,  lightest)

    Usage:
        model = get_wideresnet(depth=28, widen_factor=10, num_classes=10)
    """
    return WideResNet(depth=depth, widen_factor=widen_factor,
                      dropout=dropout, num_classes=num_classes)


def get_wrn28_10(num_classes: int = 10) -> WideResNet:
    """WideResNet-28-10 — exact model from Cubuk et al. (2020) CIFAR experiments."""
    return WideResNet(depth=28, widen_factor=10, dropout=0.3, num_classes=num_classes)


def get_wrn16_8(num_classes: int = 10) -> WideResNet:
    """WideResNet-16-8 — lighter variant for ablation / limited compute."""
    return WideResNet(depth=16, widen_factor=8, dropout=0.3, num_classes=num_classes)


# QUICK TEST

if __name__ == "__main__":

    print("WideResNet — Zagoruyko & Komodakis (2016)\n")
    print("Used in RandAugment (Cubuk et al. 2020) for CIFAR-10:\n")

    configs = [
        ("WRN-28-10  (paper model)", dict(depth=28, widen_factor=10)),
        ("WRN-16-8   (lighter)",     dict(depth=16, widen_factor=8)),
    ]

    dummy = torch.zeros(4, 3, 32, 32)

    for name, kwargs in configs:
        model  = get_wideresnet(**kwargs, num_classes=10)
        out    = model(dummy)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}")
        print(f"    Output shape : {out.shape}")
        print(f"    Parameters   : {params:>12,}")
        print()
