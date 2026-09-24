"""
experiments/madaug/make_split.py

Builds the fixed CIFAR-100 controlled split shared by all five methods:
    49,000 train / 1,000 validation (10 per class) / 10,000 test (official test set).

Method: sklearn StratifiedShuffleSplit(n_splits=1, random_state=0), the same splitter and
seed the official MADAug code uses for its reduced datasets (dataset.py::get_dataloaders).
The paper selects "1,000 images from the dataset as the validation set" for CIFAR-100 and
keeps the two sets disjoint (Sec. 3.3, Sec. 4). The split does not depend on the run seed.

Usage:
    python -m experiments.madaug.make_split            # writes experiments/madaug/splits/
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "splits" / "cifar100_val1000.json"


def indices_sha256(idx) -> str:
    return hashlib.sha256(
        ",".join(map(str, sorted(int(i) for i in idx))).encode()
    ).hexdigest()


def build_split(data_root: str, val_size: int = 1000, random_state: int = 0) -> dict:
    train_set = datasets.CIFAR100(data_root, train=True, download=True)
    targets = np.array(train_set.targets)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)

    per_class = Counter(targets[val_idx].tolist())
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(train_idx) + len(val_idx) == len(targets)
    assert set(per_class.values()) == {val_size // 100}, per_class

    return {
        "dataset": "cifar100",
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_per_class": val_size // 100,
        "method": f"sklearn StratifiedShuffleSplit(n_splits=1, test_size={val_size}, random_state={random_state})",
        "train_sha256": indices_sha256(train_idx),
        "val_sha256": indices_sha256(val_idx),
        "val_idx": val_idx.tolist(),
    }


def load_split(path) -> tuple[list, list]:
    """Returns (train_idx, val_idx); train = all 50k CIFAR-100 train indices not in val."""
    split = json.loads(Path(path).read_text())
    val_idx = sorted(split["val_idx"])
    val_set = set(val_idx)
    train_idx = [i for i in range(50000) if i not in val_set]
    assert indices_sha256(train_idx) == split["train_sha256"], "split file corrupted"
    assert indices_sha256(val_idx) == split["val_sha256"], "split file corrupted"
    return train_idx, val_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    split = build_split(args.data_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split))
    print(
        f"Wrote {out}: {split['n_train']} train / {split['n_val']} val "
        f"({split['val_per_class']} per class)\n  val_sha256={split['val_sha256']}"
    )
