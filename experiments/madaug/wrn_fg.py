"""
experiments/madaug/wrn_fg.py

Thin f()/g() interface around this project's WRN-28-10 (models/wideresnet.py, unchanged).

Official MADAug calls three things on the task model:
    gf_model.f(x)            -> penultimate features   (adaptive_augmentor.py: predict_aug_params, explore)
    gf_model.g(features)     -> logits                 (main_higher.py: train, bi-level inner step)
    gf_model.fc.in_features  -> policy input width     (main_higher.py: main, Projection construction)
In the official repo these are methods of networks/wideresnet.py::WideResNet. Here they are
provided by a wrapper that reuses the wrapped network's own modules, so dropout, init,
downsampling and parameters are exactly those of models/wideresnet.py.

f() mirrors models/wideresnet.py::WideResNet.forward up to the classifier; the sanity test
asserts wrapper(x) == net(x) bit-for-bit so any future drift is caught.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WRNFeatureClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    @property
    def fc(self) -> nn.Linear:
        return self.net.fc

    def f(self, x: torch.Tensor) -> torch.Tensor:
        net = self.net
        out = net.conv1(x)
        out = net.group1(out)
        out = net.group2(out)
        out = net.group3(out)
        out = F.relu(net.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(self.f(x))
