"""
experiments/external_baselines/data.py

CIFAR-100 loaders for the RandAugment / TrivialAugment external baselines under the controlled
protocol of the MADAug comparison (experiments/madaug/). Everything except the augmentation
policy is taken from the MADAug controlled pipeline by import, not by copy:

    split          : experiments/madaug/splits/cifar100_val1000.json (49k train / 1k val), load_split
    normalization  : data.datasets.CIFAR100_MEAN / CIFAR100_STD
    cutout         : experiments.madaug.data.CutoutDefault(16) (official darts/MADAug cutout)
    test set       : official CIFAR-100 test set (10k)
    batch / loader : batch 128, shuffle, drop_last=True, 4 workers (as MADAug's train_queue)

Train transform (order of the official TA codebase, TrivialAugment/data.py: the augmenter is
inserted at position 0, i.e. on the raw image, and Cutout is appended after Normalize):
    policy (PIL, aug_lib) -> RandomCrop(32, pad 4) -> RandomHorizontalFlip -> ToTensor
    -> Normalize(CIFAR-100) -> Cutout(16)
Validation / test transform (no randomness):  ToTensor -> Normalize(CIFAR-100)
"""

import os

import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms

from data.datasets import CIFAR100_MEAN, CIFAR100_STD
from experiments.madaug.data import CutoutDefault
from experiments.madaug.make_split import load_split
from experiments.external_baselines.vendor import aug_lib

# Method settings, all from the official TA repository @ e6545d8 (see vendor/SOURCE.txt):
#   randaugment    confs/wresnet28x10_cifar100_b128_maxlr.1_randaug_fixedsesp_200epochs_2_nowarmup.yaml
#                  (N=2, M=14 "from appendix" = RandAugment paper Sec. 4.2, CIFAR-100 WRN-28-10;
#                   augmentation_search_space: fixed_standard = the 14 RandAugment-paper ops)
#   trivialaugment confs/wresnet28x10_cifar100_b128_maxlr.1_ta_wide_nowarmup_200epochs.yaml
#                  (augmentation_search_space: wide_standard; one op, uniformly random strength)
# num_strengths=31 (levels 0..30): TA paper / README ("set_augmentation_space('wide_standard',31)"),
# torchvision and the RandAugment paper's [0, 30] magnitude scale. NOTE: the TA repo's train.py
# default passes 30 (levels 0..29); both span the same full op range, only the grid differs.
METHODS = {
    "randaugment": {
        "space": "fixed_standard",
        "num_strengths": 31,
        "n": 2,
        "m": 14,
    },
    "trivialaugment": {
        "space": "wide_standard",
        "num_strengths": 31,
    },
}

_configured = None  # (space, num_strengths) active in THIS process


class OfficialAugment:
    """Applies the unmodified aug_lib sampler.

    aug_lib keeps its augmentation space in module globals and resets it to the default at import.
    DataLoader workers started with 'spawn' (macOS) re-import the module, so the space is (re)set
    lazily in whichever process calls the transform. The sampler itself is aug_lib's own class.
    """

    def __init__(self, method: str):
        self.method = method
        self.cfg = METHODS[method]
        if method == "randaugment":
            self.sampler = aug_lib.RandAugment(self.cfg["n"], self.cfg["m"])
        else:
            self.sampler = aug_lib.TrivialAugment()

    def ensure_space(self):
        global _configured
        key = (self.cfg["space"], self.cfg["num_strengths"], os.getpid())
        if _configured != key:
            aug_lib.set_augmentation_space(self.cfg["space"], self.cfg["num_strengths"])
            _configured = key

    def __call__(self, img):
        self.ensure_space()
        return self.sampler(img)

    def __repr__(self):
        return f"OfficialAugment({self.method}, {self.cfg}, sampler=aug_lib.{type(self.sampler).__name__})"


def build_transforms(method: str, cutout_length: int = 16):
    train_tf = transforms.Compose(
        [
            OfficialAugment(method),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    if cutout_length != 0:
        train_tf.transforms.append(CutoutDefault(cutout_length))
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return train_tf, test_tf


def get_external_cifar100_loaders(
    method: str,
    data_root: str,
    split_file: str,
    batch_size: int = 128,
    num_workers: int = 4,
    cutout_length: int = 16,
    debug_n_train: int = 0,
    debug_n_eval: int = 0,
):
    train_tf, test_tf = build_transforms(method, cutout_length)

    full_train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=train_tf
    )
    full_train_eval = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=test_tf
    )
    testset = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=test_tf
    )
    train_idx, val_idx = load_split(split_file)
    if debug_n_train:
        train_idx = train_idx[:debug_n_train]
    train_data = Subset(full_train, train_idx)
    val_data = Subset(full_train_eval, val_idx)

    # Smoke tests only: evaluate on the first N test / validation images.
    eval_test = Subset(testset, range(debug_n_eval)) if debug_n_eval else testset
    eval_val = Subset(val_data, range(debug_n_eval)) if debug_n_eval else val_data

    pin = torch.cuda.is_available()
    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin,
        num_workers=num_workers,
    )
    test_queue = torch.utils.data.DataLoader(
        eval_test,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=pin,
        num_workers=num_workers,
    )
    val_eval_queue = torch.utils.data.DataLoader(
        eval_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin,
        num_workers=0,
    )
    return train_queue, test_queue, val_eval_queue, (train_idx, val_idx)
