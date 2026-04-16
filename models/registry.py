"""
models/registry.py
Simple factory — get any model by name.

Available models:
    baseline_cnn    → custom 6-layer CNN
    resnet18        → ResNet-18, CIFAR-adapted (main thesis model)
    resnet50        → ResNet-50
    wideresnet      → WideResNet-28-10 (Zagoruyko & Komodakis, 2016)
                      Used in RandAugment paper (Cubuk et al., 2020) for CIFAR-10
    wrn28_10        → WideResNet-28-10 (alias)
    wrn16_8         → WideResNet-16-8  (lighter variant)
    pyramidnet      → PyramidNet-110 (α=48)  (Han et al., 2017)
    pyramidnet272   → PyramidNet-272 (α=200) — exact model from RandAugment paper
"""

import torch.nn as nn
import models.baseline_cnn      as custom_cnn
import models.baseline_resnet18 as resnet18_mod
import models.baseline_resnet50 as resnet50_mod
import models.wideresnet        as wrn_mod
import models.pyramidnet        as pyramid_mod


def get_model(name: str, num_classes: int = 10) -> nn.Module:
    """
    Get a model by name.

    Usage:
        model = get_model("resnet18",      num_classes=10)
        model = get_model("wideresnet",    num_classes=10)   # WRN-28-10
        model = get_model("pyramidnet",    num_classes=10)   # PyramidNet-110
        model = get_model("pyramidnet272", num_classes=10)   # PyramidNet-272 (paper)
    """
    name = name.lower().strip()

    if name == "baseline_cnn":
        return custom_cnn.get_baseline_model(num_classes=num_classes)

    elif name == "resnet18":
        return resnet18_mod.get_baseline_model(num_classes=num_classes)

    elif name == "resnet50":
        return resnet50_mod.get_baseline_model(num_classes=num_classes)

    # ── WideResNet (Zagoruyko & Komodakis, 2016) ──────────────
    # Used in Cubuk et al. (2020) RandAugment paper for CIFAR-10
    elif name in {"wideresnet", "wrn28_10", "wrn-28-10"}:
        return wrn_mod.get_wrn28_10(num_classes=num_classes)

    elif name in {"wrn16_8", "wrn-16-8"}:
        return wrn_mod.get_wrn16_8(num_classes=num_classes)

    # ── PyramidNet (Han et al., 2017) ─────────────────────────
    # Used in Cubuk et al. (2020) RandAugment paper for CIFAR-10 SOTA
    elif name == "pyramidnet":
        return pyramid_mod.get_pyramidnet110(num_classes=num_classes)

    elif name in {"pyramidnet272", "pyramidnet-272"}:
        return pyramid_mod.get_pyramidnet272(num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown model: '{name}'. "
            f"Choose from: baseline_cnn, resnet18, resnet50, "
            f"wideresnet, wrn16_8, pyramidnet, pyramidnet272"
        )