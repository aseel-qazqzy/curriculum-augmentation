"""
Download and prepare the Tiny-ImageNet-200 dataset.

Run once before training:
    python data/download_tiny_imagenet.py

What this does:
  1. Downloads tiny-imagenet-200.zip (~236 MB)
  2. Extracts it to data/tiny-imagenet-200/
  3. Reorganizes the flat val/ directory into class subfolders

Final structure:
    data/tiny-imagenet-200/
        train/   <- 200 class folders, 500 images each  (100k total)
        val/     <- reorganized: 200 class folders, 50 images each (10k total)
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
ZIP_NAME = "tiny-imagenet-200.zip"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 / total_size)
    mb = downloaded / 1_000_000
    sys.stdout.write(f"\r  {pct:5.1f}%  ({mb:.1f} MB)")
    sys.stdout.flush()


def download(data_root="data"):
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / ZIP_NAME
    out_dir = root / "tiny-imagenet-200"

    # Skip if already extracted
    if out_dir.exists() and (out_dir / "train").exists():
        print(f"  Already extracted at {out_dir}")
    else:
        if not zip_path.exists():
            print(f"  Downloading Tiny-ImageNet ...")
            urllib.request.urlretrieve(URL, zip_path, _progress)
            print()
        else:
            print(f"  Found existing zip: {zip_path}")

        print(f"  Extracting ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)
        print("  Extraction complete.")

    _reorganize_val(out_dir / "val")

    n_train = sum(1 for _ in (out_dir / "train").rglob("*.JPEG"))
    n_val = sum(1 for _ in (out_dir / "val").rglob("*.JPEG"))
    print(f"\n  Done.  Dataset ready at: {out_dir}")
    print(f"  Train: {n_train:,} images  |  Val: {n_val:,} images")


def _reorganize_val(val_dir: Path):

    ann_file = val_dir / "val_annotations.txt"
    img_dir = val_dir / "images"

    if not ann_file.exists():
        print("  val/ already reorganized.")
        return

    print("  Reorganizing val/ into class subfolders ...")
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, class_id = parts[0], parts[1]
            src = img_dir / fname
            dst_dir = val_dir / class_id
            dst_dir.mkdir(exist_ok=True)
            if src.exists():
                shutil.move(str(src), str(dst_dir / fname))

    if img_dir.exists():
        shutil.rmtree(img_dir)
    ann_file.unlink()
    print("  val/ reorganized.")


if __name__ == "__main__":
    download()
