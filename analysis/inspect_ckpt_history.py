"""
analysis/inspect_ckpt_history.py

Print what a run's checkpoint and history files contain (final accuracies,
epoch count, scalar metadata). Used to check which run produced a checkpoint
when files with the same name were overwritten.

    python analysis/inspect_ckpt_history.py
    python analysis/inspect_ckpt_history.py --pattern "wideresnet_static_mixing_*ep100*s42_p19"
"""

import argparse
import glob
from pathlib import Path

import torch

DEFAULT_PATTERN = (
    "wideresnet_tiered_[el][tp]s_mix_both_sgd_cosine_ep100_cifar100_s42_p19"
)


def _show(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n{path}")
    if not isinstance(obj, dict):
        print(f"  type: {type(obj)}")
        return
    print(f"  keys: {list(obj.keys())}")
    for k, v in obj.items():
        if isinstance(v, list) and v and not isinstance(v[0], (list, dict)):
            print(f"  {k}: last={v[-1]}  len={len(v)}")
        elif isinstance(v, (int, float, str, bool)) or v is None:
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    args = parser.parse_args()

    root = Path(args.checkpoint_dir)
    files = sorted(glob.glob(str(root / f"{args.pattern}_history.pt")))
    files += sorted(glob.glob(str(root / f"{args.pattern}_best.pth")))
    if not files:
        print(f"No files match {root / args.pattern}_{{history.pt,best.pth}}")
    for f in files:
        _show(f)


if __name__ == "__main__":
    main()
