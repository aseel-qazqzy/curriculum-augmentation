
import os
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN        = (0.4914, 0.4822, 0.4465)
CIFAR10_STD         = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN       = (0.5071, 0.4867, 0.4408)
CIFAR100_STD        = (0.2675, 0.2565, 0.2761)
TINY_IMAGENET_MEAN  = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD   = (0.2770, 0.2691, 0.2821)

CIFAR_STATS = {
    "cifar10":        {"mean": CIFAR10_MEAN,       "std": CIFAR10_STD},
    "cifar100":       {"mean": CIFAR100_MEAN,      "std": CIFAR100_STD},
    "tiny_imagenet":  {"mean": TINY_IMAGENET_MEAN, "std": TINY_IMAGENET_STD},
}


# TRANSFORMS 
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
    num_workers:   int   = 0,
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
    full_val_dataset   = datasets.CIFAR10(root, train=True,  download=True, transform=test_transform)
    test_dataset       = datasets.CIFAR10(root, train=False, download=True, transform=test_transform)

    if val_split == 0.0:
        train_dataset = full_train_dataset
        val_dataset   = None
    else:
        val_size   = int(len(full_train_dataset) * val_split)
        if val_size < 1:
            raise ValueError(f"val_split={val_split} produces an empty validation set.")
        indices       = torch.randperm(len(full_train_dataset),
                                       generator=torch.Generator().manual_seed(42))
        train_dataset = torch.utils.data.Subset(full_train_dataset, indices[:len(full_train_dataset) - val_size])
        val_dataset   = torch.utils.data.Subset(full_val_dataset,   indices[len(full_train_dataset) - val_size:])

    if debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(512))
        val_dataset   = torch.utils.data.Subset(val_dataset,   range(128)) if val_dataset else None
        test_dataset  = torch.utils.data.Subset(test_dataset,  range(128))

    pin_memory   = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) if val_dataset else None
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    val_str = f"{len(val_dataset):,}" if val_dataset else "none (full-train mode)"
    print(f"CIFAR-10 loaded | Train: {len(train_dataset):,} | Val: {val_str} | Test: {len(test_dataset):,}")
    return train_loader, val_loader, test_loader

def get_cifar100_loaders(
    root:          str   = "./data/raw",
    batch_size:    int   = 128,
    val_split:     float = 0.1,
    train_transform = None,
    test_transform  = None,
    num_workers:   int   = 0,
    debug:         bool  = False,
):

    if train_transform is None or test_transform is None:
        train_transform, test_transform = get_static_transforms("cifar100")

    # Download datasets

    full_train_dataset = datasets.CIFAR100(root, train=True,  download=True, transform=train_transform)
    full_val_dataset   = datasets.CIFAR100(root, train=True,  download=True, transform=test_transform)
    test_dataset       = datasets.CIFAR100(root, train=False, download=True, transform=test_transform)

    if val_split == 0.0:
        train_dataset = full_train_dataset
        val_dataset   = None
    else:
        val_size   = int(len(full_train_dataset) * val_split)
        if val_size < 1:
            raise ValueError(f"val_split={val_split} produces an empty validation set.")
        indices       = torch.randperm(len(full_train_dataset),
                                       generator=torch.Generator().manual_seed(42))
        train_dataset = torch.utils.data.Subset(full_train_dataset, indices[:len(full_train_dataset) - val_size])
        val_dataset   = torch.utils.data.Subset(full_val_dataset,   indices[len(full_train_dataset) - val_size:])

    if debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(512))
        val_dataset   = torch.utils.data.Subset(val_dataset,   range(128)) if val_dataset else None
        test_dataset  = torch.utils.data.Subset(test_dataset,  range(128))

    pin_memory   = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) if val_dataset else None
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    val_str = f"{len(val_dataset):,}" if val_dataset else "none (full-train mode)"
    print(f"CIFAR-100 loaded | Train: {len(train_dataset):,} | Val: {val_str} | Test: {len(test_dataset):,}")
    return train_loader, val_loader, test_loader

def _reorganize_tiny_imagenet_val(val_dir):
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    img_dir  = os.path.join(val_dir, "images")

    if not os.path.exists(ann_file):
        return  # already reorganized

    print("  Tiny-ImageNet val/ into class subfolders ...")
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, class_id = parts[0], parts[1]
            src     = os.path.join(img_dir, fname)
            dst_dir = os.path.join(val_dir, class_id)
            os.makedirs(dst_dir, exist_ok=True)
            if os.path.exists(src):
                shutil.move(src, os.path.join(dst_dir, fname))

    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir)
    os.remove(ann_file)
    print("  val/ reorganized.")


def get_tiny_imagenet_loaders(
    root:            str   = "./data",
    batch_size:      int   = 128,
    val_split:       float = 0.1,
    train_transform        = None,
    test_transform         = None,
    num_workers:     int   = 0,
    debug:           bool  = False,
):
  
    base      = os.path.join(root, "tiny-imagenet-200")
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Tiny-ImageNet not found at {base}.\n"
            f"Run:  python data/download_tiny_imagenet.py"
        )

    _reorganize_tiny_imagenet_val(val_dir)

    if train_transform is None or test_transform is None:
        _norm = transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _norm,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            _norm,
        ])

    full_train = datasets.ImageFolder(train_dir, transform=train_transform)
    full_val   = datasets.ImageFolder(train_dir, transform=test_transform)
    # Official val/ — completely separate images, used only for final test
    test_ds    = datasets.ImageFolder(val_dir, transform=test_transform)

    if val_split == 0.0:
        train_ds = full_train
        val_ds   = None
    else:
        val_size = int(len(full_train) * val_split)
        indices  = torch.randperm(len(full_train),
                                  generator=torch.Generator().manual_seed(42))
        train_ds = torch.utils.data.Subset(full_train, indices[:len(full_train) - val_size])
        val_ds   = torch.utils.data.Subset(full_val,   indices[len(full_train) - val_size:])

    if debug:
        train_ds = torch.utils.data.Subset(train_ds, range(512))
        val_ds   = torch.utils.data.Subset(val_ds,   range(128)) if val_ds else None
        test_ds  = torch.utils.data.Subset(test_ds,  range(128))

    pin_memory   = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory) if val_ds else None
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    print(f"Tiny-ImageNet loaded | Train: {len(train_ds):,} | "
          f"Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar10_loaders(debug=True)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")   # (32, 3, 32, 32)
    print(f"Labels shape: {labels.shape}")  # (32,)