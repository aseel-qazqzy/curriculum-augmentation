"""
models/pyramidnet.py

PyramidNet for CIFAR-10 / CIFAR-100.

Architecture used in Cubuk et al. (2020) RandAugment paper (Table 2, CIFAR-10):
    PyramidNet+ShakeDrop (α=270, depth=272)
    Baseline error:      1.5%
    + RandAugment error: 1.0%  (N=2, M=28)

Reference:
    Han, D., Kim, J., & Kim, J. (2017).
    Deep pyramidal residual networks.
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
    https://arxiv.org/abs/1610.02915

Key idea:
    Unlike standard ResNets where channel width doubles only at downsampling,
    PyramidNet increases the channel width gradually at EVERY layer.
    This gives a smoother feature map progression.

    widths[l] = base_channels + round(alpha * l / total_layers)

    For PyramidNet-110 (α=48, CIFAR):  widths go from 16 → 64
    For PyramidNet-272 (α=200, CIFAR): widths go from 16 → 216  (used in paper)

Architecture:
    Input:     (B, 3, 32, 32)
    Conv1:     3 → 16, 3×3, stride=1
    Layer 1:   n blocks, stride=1, channels grow from 16 to ~16+α/3
    Layer 2:   n blocks, stride=2, channels continue growing
    Layer 3:   n blocks, stride=2, channels continue growing to 16+α
    BN → ReLU → AdaptiveAvgPool → FC
    Output:    (B, num_classes)

    n = (depth - 2) / 9  for bottleneck, or (depth - 2) / 6 for basic

Usage:
    model = get_pyramidnet(depth=110, alpha=48,  num_classes=10)   # lightweight
    model = get_pyramidnet(depth=272, alpha=200, num_classes=10)   # paper model
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# PYRAMID BASIC BLOCK
class PyramidBlock(nn.Module):
    """
    PyramidNet basic residual block (pre-activation, no ReLU before addition).

    Key difference from standard ResNet block:
    - Input and output channels differ at EVERY block (not just at group boundaries)
    - No ReLU before the residual addition (allows identity shortcut even when
      channels increase — zero-padding is used on the shortcut)
    - Pre-activation: BN → Conv → BN → Conv (ReLU only before conv2)

    Shortcut: if in_channels != out_channels, zero-pad the extra channels
              and apply avg-pool if stride > 1 (no learned 1×1 projection).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)  # applied after addition

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Shortcut: avg pool for stride > 1, then zero-pad if channels differ
        if stride > 1:
            self.shortcut_pool = nn.AvgPool2d(
                kernel_size=stride, stride=stride, ceil_mode=True
            )
        else:
            self.shortcut_pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.bn3(out)

        # Shortcut: spatial downsampling + zero-channel-padding
        shortcut = x
        if self.shortcut_pool is not None:
            shortcut = self.shortcut_pool(shortcut)
        if self.in_channels != self.out_channels:
            # Zero-pad the extra channels (no learned projection — paper design)
            pad = out.size(1) - shortcut.size(1)
            shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, pad))

        return F.relu(out + shortcut, inplace=True)


# PYRAMIDNET
class PyramidNet(nn.Module):
    """
    PyramidNet (Han et al., 2017) for CIFAR-10/100.

    The defining feature: channel width increases gradually at every layer
    by a fixed addend (alpha / total_layers), rather than doubling only
    at group boundaries as in standard ResNets.

    Args:
        depth:       Total depth. (depth - 2) must be divisible by 6.
                     Common: 110 (lightweight), 164, 272 (paper SOTA).
        alpha:       Total channel increase across the network.
                     depth=272, alpha=200 → channels 16 → 216 (paper model).
                     depth=110, alpha=48  → channels 16 → 64  (lightweight).
        num_classes: 10 for CIFAR-10, 100 for CIFAR-100.
    """

    def __init__(self, depth: int = 110, alpha: int = 48, num_classes: int = 10):
        super().__init__()

        assert (depth - 2) % 6 == 0, (
            f"depth must be 6n+2 for basic blocks (e.g. 20, 32, 44, 110). Got {depth}."
        )
        n = (depth - 2) // 6  # blocks per group

        # Compute per-layer channel widths — this is the key PyramidNet feature
        # Each of the 3n blocks increases channels by alpha / (3n)
        total_blocks = 3 * n
        self._widths = [
            16 + round(alpha * (i + 1) / total_blocks) for i in range(total_blocks)
        ]

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Build 3 groups of n blocks each
        self.layer1 = self._make_layer(16, self._widths[:n], stride=1)
        self.layer2 = self._make_layer(
            self._widths[n - 1], self._widths[n : 2 * n], stride=2
        )
        self.layer3 = self._make_layer(
            self._widths[2 * n - 1], self._widths[2 * n :], stride=2
        )

        final_channels = self._widths[-1]
        self.bn_final = nn.BatchNorm2d(final_channels)
        self.fc = nn.Linear(final_channels, num_classes)

        self._init_weights()

    def _make_layer(self, in_ch: int, widths: list, stride: int) -> nn.Sequential:
        """
        Build a group of blocks.
        First block may downsample (stride > 1) and always changes channels.
        Subsequent blocks only change channels (stride=1).
        """
        blocks = []
        for i, out_ch in enumerate(widths):
            s = stride if i == 0 else 1
            blocks.append(PyramidBlock(in_ch, out_ch, stride=s))
            in_ch = out_ch
        return nn.Sequential(*blocks)

    def _init_weights(self):
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
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# FACTORY FUNCTIONS
def get_pyramidnet(
    depth: int = 110, alpha: int = 48, num_classes: int = 10
) -> PyramidNet:
    """
    Get PyramidNet model.

    Paper configurations for CIFAR:
        PyramidNet-110 (α=48):   lightweight, fast, ~1.7M params
        PyramidNet-272 (α=200):  paper's SOTA model, ~26M params

    Cubuk et al. (2020) used PyramidNet-272 (α=200) + ShakeDrop
    with RandAugment (N=2, M=28) to achieve 1.0% error on CIFAR-10.

    Usage:
        model = get_pyramidnet(depth=110, alpha=48,  num_classes=10)
        model = get_pyramidnet(depth=272, alpha=200, num_classes=10)
    """
    return PyramidNet(depth=depth, alpha=alpha, num_classes=num_classes)


def get_pyramidnet110(num_classes: int = 10) -> PyramidNet:
    """PyramidNet-110 (α=48) — lightweight variant for ablation / limited compute."""
    return PyramidNet(depth=110, alpha=48, num_classes=num_classes)


def get_pyramidnet272(num_classes: int = 10) -> PyramidNet:
    """PyramidNet-272 (α=200) — exact model from RandAugment paper Table 2."""
    return PyramidNet(depth=272, alpha=200, num_classes=num_classes)


# QUICK TEST
if __name__ == "__main__":
    print("PyramidNet — Han et al. (CVPR 2017)")
    print("Used in RandAugment (Cubuk et al. 2020) for CIFAR-10 SOTA\n")

    configs = [
        ("PyramidNet-110 (α=48,  lightweight)", dict(depth=110, alpha=48)),
        ("PyramidNet-272 (α=200, paper model)", dict(depth=272, alpha=200)),
    ]

    dummy = torch.zeros(4, 3, 32, 32)

    for name, kwargs in configs:
        model = get_pyramidnet(**kwargs, num_classes=10)
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}")
        print(f"    Output shape : {out.shape}")
        print(f"    Parameters   : {params:>12,}")
        print(
            f"    Channel widths (first 5 / last 5): "
            f"{model._widths[:5]} ... {model._widths[-5:]}"
        )
        print()
