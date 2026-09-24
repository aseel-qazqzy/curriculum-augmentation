"""
experiments/madaug/data.py

CIFAR-100 loaders for MADAug under the controlled protocol.

The official code has no CIFAR-100 path (dataset.py::get_dataloaders raises ValueError for
'cifar100'). This module reproduces the *structure* of the official search-mode loaders
(dataset.py::get_dataloaders with search=True, train_portion=1 and a search_dataset, i.e.
the reduced_cifar10 path) on the fixed 49k/1k split:

    train_queue  : AugmentDataset(search=True)  -> RandomCrop(32,4) + HFlip + ToTensor, UNnormalized;
                   batch 128, shuffle, drop_last=True        (official trainloader, train_sampler=None)
    search_queue : AugmentDataset(search=False, train=False) -> test transform (no augmentation);
                   batch get_search_divider('wresnet28_10') = 16, shuffle, drop_last=True
    test_queue   : official CIFAR-100 test set, test transform, batch 128
    val_eval     : the same 1k validation images, test transform, batch 128 (logging only)

CutoutDefault and AugmentDataset are copied verbatim from official dataset.py (commit 279b60a).

[madaug-adapt] Differences to official get_dataloaders:
  * CIFAR-100 with CIFAR-100 normalization constants (shared protocol) instead of the
    CIFAR-10 constants the official code would apply to any name containing 'cifar10'.
  * Fixed 49k/1k split from experiments/madaug/splits/cifar100_val1000.json.
  * search_queue uses num_workers=0: the official loop calls next(iter(search_queue)) at every
    search step, which with worker processes re-spawns workers each time. Sampling is identical
    (the RandomSampler runs in the main process either way); only the loading process differs.
  * pin_memory only when CUDA is available.
"""

import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms

from data.datasets import CIFAR100_MEAN, CIFAR100_STD
from experiments.madaug.core.config import get_search_divider
from experiments.madaug.make_split import load_split


# ── verbatim from official dataset.py ─────────────────────────────────────────
class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class AugmentDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, pre_transforms, after_transforms, valid_transforms, search, train
    ):
        super(AugmentDataset, self).__init__()
        self.dataset = dataset
        self.pre_transforms = pre_transforms
        self.after_transforms = after_transforms
        self.valid_transforms = valid_transforms
        self.search = search
        self.train = train

    def __getitem__(self, index):
        if self.search:
            raw_image, target = self.dataset.__getitem__(index)
            image = self.pre_transforms(raw_image)
            image = transforms.ToTensor()(image)
            return image, target
        else:
            img, target = self.dataset.__getitem__(index)
            if self.train:
                img = self.pre_transforms(img)
                img = self.after_transforms(img)
            else:
                if self.valid_transforms is not None:
                    img = self.valid_transforms(img)
            return img, target

    def __len__(self):
        return self.dataset.__len__()


# ── end verbatim ──────────────────────────────────────────────────────────────


def get_madaug_cifar100_loaders(
    data_root: str,
    split_file: str,
    batch_size: int = 128,
    num_workers: int = 4,
    cutout_length: int = 16,
    model_name: str = "wresnet28_10",
    debug_n_train: int = 0,
    debug_n_eval: int = 0,
):
    # Official transform structure (dataset.py, 'cifar10' branch), CIFAR-100 constants.
    transform_train_pre = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    )
    transform_train_after = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    if cutout_length != 0:  # main.sh passes --cutout --cutout_length 16
        transform_train_after.transforms.append(CutoutDefault(cutout_length))

    full_train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=None
    )
    testset = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=None
    )
    train_idx, val_idx = load_split(split_file)
    if debug_n_train:
        train_idx = train_idx[:debug_n_train]
    total_trainset = Subset(full_train, train_idx)
    search_dataset = Subset(full_train, val_idx)

    train_data = AugmentDataset(
        total_trainset,
        transform_train_pre,
        transform_train_after,
        transform_test,
        search=True,
        train=True,
    )
    search_data = AugmentDataset(
        search_dataset,
        transform_train_pre,
        transform_train_after,
        transform_test,
        search=False,
        train=False,
    )
    test_data = AugmentDataset(
        testset,
        transform_train_pre,
        transform_train_after,
        transform_test,
        search=False,
        train=False,
    )

    # Smoke tests only: evaluate on the first N test / validation images.
    eval_test = Subset(test_data, range(debug_n_eval)) if debug_n_eval else test_data
    eval_val = Subset(search_data, range(debug_n_eval)) if debug_n_eval else search_data

    pin = torch.cuda.is_available()
    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin,
        num_workers=num_workers,
    )
    search_queue = torch.utils.data.DataLoader(
        search_data,
        batch_size=get_search_divider(model_name),
        shuffle=True,
        drop_last=True,
        pin_memory=pin,
        num_workers=0,
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

    return train_queue, search_queue, test_queue, val_eval_queue, (train_idx, val_idx)
