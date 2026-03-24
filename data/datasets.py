"""
data/datasets.py
Data loading for CIFAR-10 with train/val/test splits.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

# NORMALIZATION STATS — single source of truth, import from here everywhere
CIFAR10_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR10_STD   = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

CIFAR_STATS = {
    "cifar10":  {"mean": CIFAR10_MEAN,  "std": CIFAR10_STD},
    "cifar100": {"mean": CIFAR100_MEAN, "std": CIFAR100_STD},
}


# TRANSFORMS — delegate to policy classes (single source of truth)
def get_static_transforms(dataset: str = "cifar10"):
    """Standard static augmentation — your main baseline."""
    from augmentations.policies import StaticAugmentation
    policy = StaticAugmentation(dataset=dataset)
    return policy.get_train_transform(), policy.get_val_transform()


def get_no_augmentation_transforms(dataset: str = "cifar10"):
    """No augmentation — absolute floor baseline."""
    from augmentations.policies import NoAugmentation
    policy = NoAugmentation(dataset=dataset)
    return policy.get_train_transform(), policy.get_val_transform()


# DATALOADERS
def get_cifar10_loaders(
    root:          str   = "./data/raw",
    batch_size:    int   = 128,
    val_split:     float = 0.1,
    train_transform = None,
    test_transform  = None,
    num_workers:   int   = 2,
    debug:         bool  = False,
):
    """
    Returns train, validation, and test DataLoaders for CIFAR-10.

    Args:
        root:             Where to download/store data
        batch_size:       Batch size for all loaders
        val_split:        Fraction of training data used for validation (default 10%)
        train_transform:  Transform applied to training set
        test_transform:   Transform applied to val/test sets
        num_workers:      Number of DataLoader workers
        debug:            If True, use tiny subset for fast testing
    """
    if train_transform is None or test_transform is None:
        train_transform, test_transform = get_static_transforms("cifar10")

    # Download datasets
    full_train_dataset = datasets.CIFAR10(root, train=True,  download=True, transform=train_transform)
    test_dataset       = datasets.CIFAR10(root, train=False, download=True, transform=test_transform)

    # Train / val split
    val_size   = int(len(full_train_dataset) * val_split)
    if val_size < 1:
        raise ValueError(f"val_split={val_split} produces an empty validation set.")
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Debug mode: tiny subsets
    if debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(512))
        val_dataset   = torch.utils.data.Subset(val_dataset,   range(128))
        test_dataset  = torch.utils.data.Subset(test_dataset,  range(128))

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"CIFAR-10 loaded | Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    return train_loader, val_loader, test_loader

def get_cifar100_loaders(
    root:          str   = "./data/raw",
    batch_size:    int   = 128,
    val_split:     float = 0.1,
    train_transform = None,
    test_transform  = None,
    num_workers:   int   = 2,
    debug:         bool  = False,
):
    """
    Returns train, validation, and test DataLoaders for CIFAR-100.

    Args:
        root:             Where to download/store data
        batch_size:       Batch size for all loaders
        val_split:        Fraction of training data used for validation (default 10%)
        train_transform:  Transform applied to training set
        test_transform:   Transform applied to val/test sets
        num_workers:      Number of DataLoader workers
        debug:            If True, use tiny subset for fast testing
    """
    if train_transform is None or test_transform is None:
        train_transform, test_transform = get_static_transforms("cifar100")

    # Download datasets
    full_train_dataset = datasets.CIFAR100(root, train=True,  download=True, transform=train_transform)
    test_dataset       = datasets.CIFAR100(root, train=False, download=True, transform=test_transform)

    # Train / val split
    val_size   = int(len(full_train_dataset) * val_split)
    if val_size < 1:
        raise ValueError(f"val_split={val_split} produces an empty validation set.")
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Debug mode: tiny subsets
    if debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(512))
        val_dataset   = torch.utils.data.Subset(val_dataset,   range(128))
        test_dataset  = torch.utils.data.Subset(test_dataset,  range(128))

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"CIFAR-100 loaded | Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar10_loaders(debug=True)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")   # (32, 3, 32, 32)
    print(f"Labels shape: {labels.shape}")  # (32,)