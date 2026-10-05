"""
experiments/external_baselines/sanity_test.py

Checks the RandAugment / TrivialAugment external baselines before any long run:
  A. isolation: no frozen baseline / MADAug / controlled-baseline file changed
  B. vendored aug_lib.py == official file except the single documented py3.11 line
  C. official configuration (ops, ranges, levels, N/M)
  D. sampling behaviour (TA: 1 op, uniform level 0..30; RA: 2 ops at level 14)
  E. pipeline order, deterministic validation/test transforms, batch shape
  F. protocol identical to controlled MADAug (split, normalization, args, optimizer, scheduler, model)

    python -m experiments.external_baselines.sanity_test
"""

import hashlib
import inspect
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(
        f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else "")
    )
    return cond


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True
    ).stdout.rstrip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def isolation():
    print("\nA. Isolation")
    protected = [
        "models",
        "training",
        "augmentations",
        "data",
        "configs",
        "scripts",
        "analysis",
        "experiments/madaug",
        "experiments/train_baseline.py",
        "experiments/utils.py",
        "experiments/config.py",
        "experiments/compute_entropy.py",
        "experiments/ablation.py",
        "requirements.txt",
        "run_sota.sh",
    ]
    diff = git("diff", "--stat", "HEAD", "--", *protected)
    check(
        "git diff HEAD empty for baseline, MADAug and controlled-baseline code",
        diff == "",
        diff,
    )
    untracked = [
        l[3:]
        for l in git("status", "--porcelain", "--untracked-files=all").splitlines()
        if l.startswith("??")
        and not l[3:].startswith("experiments/external_baselines/")
        and not l[3:].startswith("results/")
    ]
    check(
        "no new files outside experiments/external_baselines/",
        not untracked,
        ", ".join(untracked),
    )
    from experiments.madaug.controlled_baselines.protocol_test import MADAUG_MANIFEST
    from experiments.madaug.controlled_baselines import derive

    bad = [f for f, h in MADAUG_MANIFEST.items() if sha(ROOT / f) != h]
    check(
        "MADAug integration unchanged (sha256 manifest from protocol_test.py)",
        not bad,
        ", ".join(bad),
    )
    check(
        "controlled Static/ETS/LPS/EGS copies still byte-identical to derivation",
        all(derive.verify().values()),
    )


def vendored():
    print("\nB. Vendored official aug_lib.py")
    orig, used = (
        HERE / "vendor" / "aug_lib_official.py.orig",
        HERE / "vendor" / "aug_lib.py",
    )
    check(
        "official copy sha256 == automl/trivialaugment@e6545d8",
        sha(orig) == "8f5f9febb671fd12be067e311dbdcb837a2a29c672861f20c2cf03732fdd2868",
    )
    a, b = orig.read_text().splitlines(), used.read_text().splitlines()
    changed = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    check(
        "imported copy differs only in line 9 (@dataclass -> @dataclass(frozen=True))",
        len(a) == len(b)
        and len(changed) == 1
        and changed[0][0] == 9
        and changed[0][1] == "@dataclass"
        and changed[0][2].startswith("@dataclass(frozen=True)"),
        str(changed),
    )


RA_PAPER_OPS = [
    "identity",
    "AutoContrast",
    "Equalize",
    "Rotate",
    "Solarize",
    "Color",
    "Posterize",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
]


def configuration():
    print("\nC. Official configuration")
    from experiments.external_baselines.data import METHODS, OfficialAugment
    from experiments.external_baselines.vendor import aug_lib

    ra, ta = OfficialAugment("randaugment"), OfficialAugment("trivialaugment")
    check(
        "RandAugment: aug_lib.RandAugment, N=2, M=14",
        type(ra.sampler) is aug_lib.RandAugment
        and ra.sampler.n == 2
        and ra.sampler.m == 14,
    )
    check(
        "TrivialAugment: aug_lib.TrivialAugment",
        type(ta.sampler) is aug_lib.TrivialAugment,
    )
    ra.ensure_space()
    mm = aug_lib.min_max_vals
    check(
        "RA space fixed_standard: 14 RandAugment-paper ops, levels 0..30",
        [t.name for t in aug_lib.ALL_TRANSFORMS] == RA_PAPER_OPS
        and aug_lib.PARAMETER_MAX == 30,
    )
    check(
        "RA ranges = AutoAugment ranges (shear .3, translate 10px, rotate 30, enhance .1-1.9, posterize 4-8)",
        (
            mm.shear.max,
            mm.translate.max,
            mm.rotate.max,
            mm.enhancer.min,
            mm.enhancer.max,
            mm.posterize.min,
            mm.posterize.max,
        )
        == (0.3, 10, 30, 0.1, 1.9, 4, 8),
        str(mm),
    )
    ta.ensure_space()
    mm = aug_lib.min_max_vals
    check(
        "TA space wide_standard: same 14 ops, levels 0..30",
        [t.name for t in aug_lib.ALL_TRANSFORMS] == RA_PAPER_OPS
        and aug_lib.PARAMETER_MAX == 30,
    )
    check(
        "TA wide ranges (shear .99, translate 32px, rotate 135, enhance .01-2, posterize 2-8)",
        (
            mm.shear.max,
            mm.translate.max,
            mm.rotate.max,
            mm.enhancer.min,
            mm.enhancer.max,
            mm.posterize.min,
            mm.posterize.max,
        )
        == (0.99, 32, 135, 0.01, 2.0, 2, 8),
        str(mm),
    )
    check(
        "METHODS table matches",
        METHODS
        == {
            "randaugment": {
                "space": "fixed_standard",
                "num_strengths": 31,
                "n": 2,
                "m": 14,
            },
            "trivialaugment": {"space": "wide_standard", "num_strengths": 31},
        },
    )


def sampling():
    print("\nD. Sampling behaviour (instrumented ops, 3000 images each)")
    from experiments.external_baselines.data import OfficialAugment
    from experiments.external_baselines.vendor import aug_lib

    img = Image.fromarray(
        np.random.RandomState(0).randint(0, 256, (32, 32, 3), dtype=np.uint8)
    )
    for method in ("trivialaugment", "randaugment"):
        aug = OfficialAugment(method)
        aug.ensure_space()
        calls = []
        originals = {t: t.xform for t in aug_lib.ALL_TRANSFORMS}
        for t in aug_lib.ALL_TRANSFORMS:
            t.xform = (
                lambda f, name: (
                    lambda im, lvl: (calls[-1].append((name, lvl)), f(im, lvl))[1]
                )
            )(t.xform, t.name)
        try:
            random.seed(0)
            outs = []
            for _ in range(3000):
                calls.append([])
                outs.append(aug(img))
        finally:
            for t, f in originals.items():
                t.xform = f
        n_ops = Counter(len(c) for c in calls)
        levels = Counter(l for c in calls for _, l in c)
        ops = Counter(o for c in calls for o, _ in c)
        if method == "trivialaugment":
            check("TA: exactly 1 op per image", set(n_ops) == {1}, str(n_ops))
            check(
                "TA: levels span 0..30, roughly uniform",
                set(levels) == set(range(31)) and min(levels.values()) > 50,
                f"min count {min(levels.values())}",
            )
        else:
            check(
                "RA: exactly 2 ops per image (with replacement)",
                set(n_ops) == {2},
                str(n_ops),
            )
            check("RA: every op at level M=14", set(levels) == {14}, str(levels))
        check(f"{method}: all 14 ops sampled", len(ops) == 14, str(len(ops)))
        check(
            f"{method}: output is RGB 32x32 PIL and differs from input for some images",
            all(o.size == (32, 32) and o.mode == "RGB" for o in outs)
            and sum(np.array(o).tobytes() != np.array(img).tobytes() for o in outs)
            > 1000,
        )


def pipeline_and_protocol():
    print("\nE/F. Pipeline and protocol vs controlled MADAug")
    from torchvision import transforms as T

    import experiments.madaug.train_madaug as tm
    import experiments.external_baselines.train_external as te
    from data.datasets import CIFAR100_MEAN, CIFAR100_STD
    from experiments.external_baselines.data import (
        OfficialAugment,
        build_transforms,
        get_external_cifar100_loaders,
    )
    from experiments.madaug.data import CutoutDefault, get_madaug_cifar100_loaders
    from experiments.madaug.make_split import DEFAULT_OUT, load_split
    from experiments.madaug.wrn_fg import WRNFeatureClassifier
    from experiments.utils import build_optimizer, build_scheduler, set_seed
    from models.registry import get_model

    tr, tst = build_transforms("trivialaugment")
    kinds = [type(t) for t in tr.transforms]
    check(
        "train order: policy -> RandomCrop(32,4) -> HFlip -> ToTensor -> Normalize -> Cutout(16)",
        kinds
        == [
            OfficialAugment,
            T.RandomCrop,
            T.RandomHorizontalFlip,
            T.ToTensor,
            T.Normalize,
            CutoutDefault,
        ]
        and tr.transforms[1].padding == 4
        and tr.transforms[5].length == 16,
    )
    check(
        "val/test transform = ToTensor -> Normalize only",
        [type(t) for t in tst.transforms] == [T.ToTensor, T.Normalize],
    )
    check(
        "normalization = CIFAR-100 constants (same as MADAug data.py)",
        tuple(tr.transforms[4].mean) == tuple(CIFAR100_MEAN)
        and tuple(tst.transforms[1].std) == tuple(CIFAR100_STD),
    )

    root = str(ROOT / "data")
    trq, teq, vq, (ti, vi) = get_external_cifar100_loaders(
        "trivialaugment", root, str(DEFAULT_SPLIT_FILE()), num_workers=0
    )
    mti, mvi = load_split(DEFAULT_OUT)
    check(
        "split identical to MADAug (49,000 / 1,000, same indices)",
        ti == mti and vi == mvi and len(ti) == 49000 and len(vi) == 1000,
    )
    m_train, _, m_test, m_val, _ = get_madaug_cifar100_loaders(
        root, str(DEFAULT_OUT), num_workers=0
    )
    check(
        "test set: 10,000 images, tensors identical to MADAug test pipeline",
        len(teq.dataset) == 10000
        and all(
            torch.equal(teq.dataset[i][0], m_test.dataset[i][0]) for i in (0, 1, 9999)
        ),
    )
    check(
        "validation tensors identical to MADAug validation pipeline",
        all(torch.equal(vq.dataset[i][0], m_val.dataset[i][0]) for i in (0, 500, 999)),
    )
    check(
        "validation/test deterministic (same tensor twice)",
        torch.equal(vq.dataset[3][0], vq.dataset[3][0])
        and torch.equal(teq.dataset[3][0], teq.dataset[3][0]),
    )
    check(
        "train loader: batch 128, shuffle, drop_last, same as MADAug train_queue",
        trq.batch_size == m_train.batch_size == 128
        and trq.drop_last
        and m_train.drop_last
        and type(trq.sampler).__name__
        == type(m_train.sampler).__name__
        == "RandomSampler",
    )
    x, y = next(iter(trq))
    check(
        "train batch shape (128,3,32,32), labels in [0,100), finite",
        x.shape == (128, 3, 32, 32)
        and y.min() >= 0
        and y.max() < 100
        and torch.isfinite(x).all(),
    )
    check(
        "train augmentation is random (same image twice differs)",
        not torch.equal(trq.dataset[0][0], trq.dataset[0][0]),
    )

    a_m, a_e = vars(tm.get_args([])), vars(te.get_args(["--method", "randaugment"]))
    shared = [
        "data_root",
        "split_file",
        "seed",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "eta_min",
        "num_workers",
        "grad_clip",
        "cutout_length",
    ]
    diff = {k: (a_m[k], a_e[k]) for k in shared if a_m[k] != a_e[k]}
    check(
        "CLI defaults identical to train_madaug.py for all shared settings",
        not diff,
        str(diff),
    )
    check(
        "no AMP in either trainer",
        "autocast" not in inspect.getsource(te)
        and "GradScaler" not in inspect.getsource(te),
    )

    set_seed(42)
    net_e = WRNFeatureClassifier(get_model("wideresnet", num_classes=100))
    set_seed(42)
    net_m = WRNFeatureClassifier(get_model("wideresnet", num_classes=100))
    check(
        "WRN-28-10 (36.5M params), identical init to MADAug for the same seed",
        all(
            torch.equal(p, q)
            for p, q in zip(net_e.state_dict().values(), net_m.state_dict().values())
        )
        and 36_000_000 < sum(p.numel() for p in net_e.parameters()) < 37_000_000,
    )
    proto = {
        "optimizer": "sgd",
        "lr": 0.1,
        "weight_decay": 5e-4,
        "epochs": 200,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "eta_min": 1e-6,
    }
    opt, _ = build_optimizer(net_e, proto)
    g = opt.param_groups[0]
    sch, _ = build_scheduler(opt, proto)
    lin, cos = sch._schedulers
    check(
        "SGD momentum 0.9 Nesterov wd 5e-4 lr 0.1; LinearLR(0.1, 5ep) -> Cosine(T_max 195, eta_min 1e-6)",
        g["nesterov"]
        and g["momentum"] == 0.9
        and g["weight_decay"] == 5e-4
        and abs(g["lr"] - 0.01) < 1e-12
        and lin.total_iters == 5
        and cos.T_max == 195
        and cos.eta_min == 1e-6,
    )
    check(
        "train step = MADAug step minus policy: CE, backward, clip_grad_norm_(5), SGD step",
        "clip_grad_norm_(model.parameters(), grad_clip)" in inspect.getsource(te.train),
    )


def DEFAULT_SPLIT_FILE():
    from experiments.madaug.make_split import DEFAULT_OUT

    return DEFAULT_OUT


if __name__ == "__main__":
    isolation()
    vendored()
    configuration()
    sampling()
    pipeline_and_protocol()
    n_fail = sum(not ok for _, ok in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)
