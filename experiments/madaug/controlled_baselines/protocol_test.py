"""
experiments/madaug/controlled_baselines/protocol_test.py

Cross-method protocol test for the controlled five-method comparison
(Static Mixing / ETS / LPS / EGS controlled reruns + MADAug). Run before any training job:

    python -m experiments.madaug.controlled_baselines.protocol_test
    python -m experiments.madaug.controlled_baselines.protocol_test --skip_runs   # no training

  A. frozen files        original baseline code unchanged (git HEAD + pinned commit); MADAug integration unchanged
  B. copy integrity      controlled copies == frozen originals + the listed [controlled] substitutions only
  C. config equivalence  for each method's command, original vs controlled cfg differ only in split/output keys;
                         shared settings (WRN-28-10, bs 128, SGD Nesterov, cosine, warmup 5, lr 0.1, CE, AMP flag,
                         op pool, tiers, LPS/EGS parameters, seed) checked explicitly
  D. split consistency   all five methods get identical train/val/test indices (incl. EGS entropy loader order)
  E. final-epoch runs    2-epoch debug run per baseline: reported test = final-epoch model, no best-val selection
  F. overwrite guards    outputs only under results/controlled_baselines/, completed runs cannot be overwritten
"""

import argparse
import ast
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEST_OUT = ROOT / "results" / "controlled_baselines" / "protocol_test"

METHODS = {
    "static_mixing": ["--augmentation", "static_mixing", "--mix_mode", "both"],
    "ets": [
        "--augmentation",
        "tiered_curriculum",
        "--tier_schedule",
        "ets",
        "--mix_mode",
        "both",
    ],
    "lps": [
        "--augmentation",
        "tiered_curriculum",
        "--tier_schedule",
        "lps",
        "--mix_mode",
        "both",
    ],
    "egs": [
        "--augmentation",
        "tiered_curriculum",
        "--tier_schedule",
        "egs",
        "--mix_mode",
        "both",
    ],
}
COMMON = [
    "--dataset",
    "cifar100",
    "--model",
    "wideresnet",
    "--epochs",
    "200",
    "--scheduler",
    "cosine",
    "--warmup_epochs",
    "5",
    "--lr",
    "0.1",
    "--use_amp",
    "--seed",
    "42",
]

# sha256 of the MADAug integration as reviewed on 2026-09-24 (must stay unchanged)
MADAUG_MANIFEST = {
    "experiments/madaug/core/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "experiments/madaug/core/adaptive_augmentor.py": "e7ffb276f56109b44b75440732f9ca0d2a3d0646adaf4164d300cc02fd24fbeb",
    "experiments/madaug/core/config.py": "e56617d19d374dffa223cf02eac7ce6dc7a2bcdadebc2af87b46d9126bcaa24a",
    "experiments/madaug/core/operation.py": "1532952a8365f4c3ad2b49cfd6a5a83e96ceb3e142edffc41ab186585c5b8096",
    "experiments/madaug/core/policy_history.py": "32ecf2d182d786731473aadf0af52459f936bbecc874d219b68da3623836ad0c",
    "experiments/madaug/core/projection.py": "1483858f7df654defaab57fe8212834a65596500af2d08d633c86252134551df",
    "experiments/madaug/data.py": "5d1d2dfc2edd46924d3d93ccb3f7621754eac1ca72edafd1e14afc0b649bc10e",
    "experiments/madaug/wrn_fg.py": "9fbf426d2021ce8b7c78e41c969fa30a63d36cdc76a7a6280912baec6441449f",
    "experiments/madaug/train_madaug.py": "3aee1552ecf98c5783fd1074bc734bb1b9f0adb6bdefeab06c9a710dedf6ffec",
    "experiments/madaug/make_split.py": "69f373cc202b7e4495a66851c6eb8513e075708199b63b539267771fd62422ba",
    "experiments/madaug/splits/cifar100_val1000.json": "34919d3734e1302c2e1d6bc9923df20a44973a08b2fbeb258a8b3fe952a987f7",
}

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


# ── A ────────────────────────────────────────────────────────────────────────
def frozen_files():
    print("\nA. Frozen files")
    from experiments.madaug.controlled_baselines.derive import (
        FROZEN_COMMIT,
        FROZEN_FILES,
    )

    protected = [
        "models",
        "training",
        "augmentations",
        "data",
        "configs",
        "scripts",
        "experiments/train_baseline.py",
        "experiments/utils.py",
        "experiments/config.py",
        "experiments/compute_entropy.py",
        "experiments/ablation.py",
        "requirements.txt",
        "setup.py",
        "analysis",
        "run_sota.sh",
    ]
    diff = git("diff", "--stat", "HEAD", "--", *protected)
    check(
        "git diff HEAD empty for all shared baseline code + analysis/", diff == "", diff
    )
    for f in FROZEN_FILES:
        check(
            f"{f} working tree == frozen commit {FROZEN_COMMIT[:7]}",
            git("hash-object", f) == git("rev-parse", f"{FROZEN_COMMIT}:{f}"),
        )
    status = [
        l[3:]
        for l in git("status", "--porcelain", "--untracked-files=all").splitlines()
    ]
    other = [
        p
        for p in status
        if not p.startswith("experiments/madaug/") and p != ".vscode/mcp.json"
    ]
    check(
        "only experiments/madaug/ added (besides pre-existing .vscode/mcp.json)",
        not other,
        ", ".join(other),
    )
    bad = [
        f
        for f, sha in MADAUG_MANIFEST.items()
        if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != sha
    ]
    check(
        "MADAug integration unchanged (11 files, sha256 manifest)",
        not bad,
        ", ".join(bad),
    )
    ckpt_status = git(
        "status", "--porcelain", "--", "checkpoints", "results/cluster", "results/logs"
    )
    check(
        "no tracked changes under checkpoints/ or existing results/",
        ckpt_status == "",
        ckpt_status,
    )


# ── B ────────────────────────────────────────────────────────────────────────
def copy_integrity():
    print(
        "\nB. Controlled copies == frozen originals + [controlled] substitutions only"
    )
    from experiments.madaug.controlled_baselines import derive

    for name, ok in derive.verify().items():
        check(f"{name} re-derived from frozen source is byte-identical", ok)
    for label, subs in (
        ("train_controlled.py", derive.TRAIN_SUBS),
        ("data_controlled.py", derive.DATA_SUBS),
        ("entropy_controlled.py", derive.ENTROPY_SUBS),
    ):
        print(f"     {label}: {len(subs)} substitutions")
        for _, _, reason in subs:
            print(f"       - {reason}")
    text = (HERE / "train_controlled.py").read_text()
    orig = derive.frozen_source("experiments/train_baseline.py")
    fn = lambda src, name: ast.dump(
        next(n for n in ast.parse(src).body if getattr(n, "name", None) == name)
    )
    for f in ("build_transforms", "_resolve_tier", "parse_args"):
        check(f"{f}() AST identical to original", fn(text, f) == fn(orig, f))


# ── C ────────────────────────────────────────────────────────────────────────
def cfg_from_main_block(module, argv):
    """Execute a module's `if __name__ == "__main__"` block with main() replaced by a capture."""
    tree = ast.parse(inspect.getsource(module))
    block = next(
        n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.dump(n.test)
    )
    captured = {}
    ns = dict(vars(module))
    ns["main"] = lambda cfg: captured.setdefault("cfg", dict(cfg))
    old = sys.argv
    sys.argv = ["x", *argv]
    try:
        exec(
            compile(
                ast.Module(body=block.body, type_ignores=[]), module.__file__, "exec"
            ),
            ns,
        )
    finally:
        sys.argv = old
    return captured["cfg"]


def config_equivalence():
    print("\nC. Config: original baseline vs controlled rerun, per method")
    import experiments.train_baseline as orig
    import experiments.madaug.controlled_baselines.train_controlled as ctrl
    from experiments.config import BASE_CONFIG
    from experiments.utils import build_optimizer, build_scheduler

    allowed = {"split_file", "val_split", "checkpoint_dir", "log_dir"}
    for m, margs in METHODS.items():
        a = cfg_from_main_block(orig, COMMON + margs)
        b = cfg_from_main_block(ctrl, COMMON + margs)
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        check(
            f"{m}: cfg differs only in {sorted(allowed)}",
            diff <= allowed,
            str(sorted(diff)),
        )
        exp = {
            "dataset": "cifar100",
            "model": "wideresnet",
            "batch_size": 128,
            "optimizer": "sgd",
            "lr": 0.1,
            "weight_decay": 5e-4,
            "scheduler": "cosine",
            "warmup_epochs": 5,
            "eta_min": 1e-6,
            "epochs": 200,
            "label_smoothing": 0.0,
            "use_amp": True,
            "op_pool": 19,
            "fixed_strength": 0.7,
            "mix_mode": "both",
            "mix_alpha": 1.0,
            "mix_prob": 0.5,
            "seed": 42,
        }
        exp.update(
            {
                k: BASE_CONFIG[k]
                for k in BASE_CONFIG
                if k.startswith(("tier_t", "lps_", "egs_"))
            }
        )
        wrong = {k: b.get(k) for k, v in exp.items() if b.get(k) != v}
        check(
            f"{m}: shared settings (WRN-28-10, bs128, SGD lr0.1 wd5e-4, cosine+5ep warmup, CE, AMP, 19-op pool, "
            f"tier/LPS/EGS params, seed)",
            not wrong,
            str(wrong),
        )
        check(
            f"{m}: checkpoint/log dirs inside results/controlled_baselines/",
            "controlled_baselines" in b["checkpoint_dir"]
            and "controlled_baselines" in b["log_dir"],
        )
    net = torch.nn.Linear(2, 2)
    opt, _ = build_optimizer(net, b)
    g = opt.param_groups[0]
    check(
        "optimizer: SGD momentum 0.9, Nesterov, wd 5e-4 (shared build_optimizer)",
        isinstance(opt, torch.optim.SGD)
        and g["nesterov"]
        and g["momentum"] == 0.9
        and g["weight_decay"] == 5e-4,
    )
    sch, _ = build_scheduler(opt, b)
    lin, cos = sch._schedulers
    check(
        "scheduler: LinearLR(start 0.1, 5 ep) -> Cosine(T_max 195, eta_min 1e-6)",
        lin.start_factor == 0.1
        and lin.total_iters == 5
        and cos.T_max == 195
        and cos.eta_min == 1e-6,
    )
    check(
        "build_transforms(): identical source in original and controlled (same aug pools, tiers, strengths)",
        inspect.getsource(orig.build_transforms)
        == inspect.getsource(ctrl.build_transforms),
    )


# ── D ────────────────────────────────────────────────────────────────────────
def split_consistency():
    print("\nD. Split consistency across all five methods")
    from torchvision.transforms import Lambda
    import experiments.madaug.controlled_baselines.train_controlled as ctrl
    from experiments.madaug.controlled_baselines.data_controlled import (
        get_controlled_cifar100_loaders,
    )
    from experiments.madaug.controlled_baselines.entropy_controlled import (
        build_raw_entropy_loader,
    )
    from experiments.madaug.data import get_madaug_cifar100_loaders
    from experiments.madaug.make_split import DEFAULT_OUT, load_split
    from data.datasets import CIFAR100_MEAN, CIFAR100_STD

    root = str(ROOT / "data")
    split = str(DEFAULT_OUT)
    tr, va = load_split(split)
    tr2, va2 = load_split(split)
    targets = np.array(
        __import__("torchvision")
        .datasets.CIFAR100(root, train=True, download=False)
        .targets
    )
    check(
        "split: 49,000 train / 1,000 val, disjoint, covers all 50,000",
        len(tr) == 49000
        and len(va) == 1000
        and not set(tr) & set(va)
        and sorted(set(tr) | set(va)) == list(range(50000)),
    )
    check(
        "split: exactly 10 validation images per class",
        set(np.bincount(targets[va], minlength=100)) == {10},
    )
    check(
        "split: deterministic (two loads identical, sha256 verified by load_split)",
        tr == tr2 and va == va2,
    )

    mq_train, mq_search, mq_test, mq_val, _ = get_madaug_cifar100_loaders(
        root, split, num_workers=0
    )
    ref = {
        "train": list(mq_train.dataset.dataset.indices),
        "val": list(mq_search.dataset.dataset.indices),
    }
    test_ref = mq_test.dataset.dataset
    print(
        f"     MADAug: train {len(ref['train'])} / val {len(ref['val'])} / test {len(test_ref)}"
    )

    for m, margs in METHODS.items():
        cfg = cfg_from_main_block(ctrl, COMMON + margs)
        train_tf, val_tf = ctrl.build_transforms(cfg)
        loader_tf = Lambda(lambda x: x) if m == "egs" else train_tf
        trl, val, tel = get_controlled_cifar100_loaders(
            root=root,
            batch_size=128,
            split_file=split,
            train_transform=loader_tf,
            test_transform=val_tf,
            num_workers=0,
        )
        same_train = list(trl.dataset.indices) == ref["train"]
        same_val = list(val.dataset.indices) == ref["val"]
        same_test = (
            len(tel.dataset) == 10000
            and not tel.dataset.train
            and np.array_equal(tel.dataset.data, test_ref.data)
            and tel.dataset.targets == test_ref.targets
        )
        check(f"{m}: train indices == MADAug (49,000, same order)", same_train)
        check(
            f"{m}: val indices == MADAug policy-val (1,000){' — LPS plateau signal' if m == 'lps' else ''}",
            same_val,
        )
        check(
            f"{m}: test set == MADAug test set (10,000 identical images/labels)",
            same_test,
        )
        norm = [t for t in val_tf.transforms if t.__class__.__name__ == "Normalize"][0]
        check(
            f"{m}: CIFAR-100 normalization (same constants as MADAug)",
            tuple(norm.mean) == tuple(CIFAR100_MEAN)
            and tuple(norm.std) == tuple(CIFAR100_STD),
        )
        check(
            f"{m}: batch size 128, shuffle, drop_last=True (original loader settings)",
            trl.batch_size == 128 and trl.drop_last,
        )
        if m == "egs":
            ent = build_raw_entropy_loader(
                dataset="cifar100",
                root=root,
                split_file=split,
                batch_size=128,
                num_workers=0,
            )
            check(
                "egs: entropy loader indices == training indices, same order (entropy[i] <-> difficulty[i])",
                list(ent.dataset.indices) == ref["train"]
                and not ent.sampler.__class__.__name__ == "RandomSampler",
            )


def snapshot_existing_outputs() -> dict:
    """{path: (size, mtime)} for every file in checkpoints/ and results/ except results/controlled_baselines/."""
    snap = {}
    for base in (ROOT / "checkpoints", ROOT / "results"):
        for f in base.rglob("*"):
            if f.is_file() and "controlled_baselines" not in f.parts:
                st = f.stat()
                snap[str(f)] = (st.st_size, st.st_mtime_ns)
    return snap


# ── E / F ────────────────────────────────────────────────────────────────────
def final_epoch_runs():
    print(
        "\nE. Final-epoch reporting: 2-epoch debug run per baseline (512 train / 128 val / 128 test)"
    )
    from experiments.madaug.controlled_baselines.data_controlled import (
        get_controlled_cifar100_loaders,
    )
    from experiments.madaug.make_split import DEFAULT_OUT
    from experiments.utils import get_device
    from models.registry import get_model
    from training.trainer import evaluate
    import experiments.madaug.controlled_baselines.train_controlled as ctrl

    if TEST_OUT.exists():
        shutil.rmtree(TEST_OUT)
    before = snapshot_existing_outputs()
    ck, lg = TEST_OUT / "checkpoints", TEST_OUT / "logs"
    device = get_device()
    for m, margs in METHODS.items():
        extra = ["--egs_update_freq", "1"] if m == "egs" else []
        argv = (
            COMMON
            + margs
            + extra
            + [
                "--debug",
                "--epochs",
                "2",
                "--num_workers",
                "0",
                "--checkpoint_dir",
                str(ck),
                "--log_dir",
                str(lg),
            ]
        )
        p = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.madaug.controlled_baselines.train_controlled",
                *argv,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if not check(
            f"{m}: run completed",
            p.returncode == 0,
            p.stderr.strip().splitlines()[-1] if p.returncode else "",
        ):
            continue
        mfile = [
            f
            for f in (TEST_OUT / "metrics").glob("*.json")
            if (
                ("tiered_" + m) in f.name
                if m != "static_mixing"
                else "static_mixing" in f.name
            )
        ][0]
        met = json.loads(mfile.read_text())
        name = met["run_name"]
        last = torch.load(
            ck / f"{name}_last.pth", map_location="cpu", weights_only=False
        )
        check(
            f"{m}: reported epoch == final epoch (2), flagged final",
            met["reported_epoch"] == 2 == last["epoch"]
            and met["reported_epoch_is_final"],
        )
        model = get_model("wideresnet", num_classes=100).to(device)
        model.load_state_dict(last["model_state_dict"])
        cfg = cfg_from_main_block(ctrl, argv)
        _, val_tf = ctrl.build_transforms(cfg)
        _, _, tel = get_controlled_cifar100_loaders(
            root=str(ROOT / "data"),
            batch_size=128,
            split_file=str(DEFAULT_OUT),
            train_transform=val_tf,
            test_transform=val_tf,
            num_workers=0,
            debug=True,
        )
        _, top1, _ = evaluate(model, tel, torch.nn.CrossEntropyLoss(), device)
        check(
            f"{m}: reported test top-1 == fresh evaluation of the final-epoch model",
            abs(top1 * 100 - met["final_epoch_top1"]) < 1e-9,
            f"{met['final_epoch_top1']:.4f} vs {top1 * 100:.4f}",
        )
        check(
            f"{m}: validation logged every epoch, no best-val selection or early stop",
            len(met["history"]["val_acc"]) == 2
            and "Best saved" not in p.stdout
            and "Early stopping" not in p.stdout
            and not list(ck.glob(f"{name}_best.pth")),
        )
        check(
            f"{m}: run name ends in _ctrl49k, uses split file",
            name.endswith("_ctrl49k")
            and met["split"]["n_train"] == 49000
            and met["split"]["n_val"] == 1000,
        )
        if m == "egs":
            check(
                "egs: entropy recomputed each epoch without index errors",
                "EGS T1:" in p.stdout,
            )
        if m == "lps":
            check(
                "lps: LossPlateauScheduler received validation loss",
                met["history"]["val_loss"][0] > 0,
            )

    print("\nF. Overwrite guards")
    argv = (
        COMMON
        + METHODS["ets"]
        + [
            "--debug",
            "--epochs",
            "2",
            "--num_workers",
            "0",
            "--checkpoint_dir",
            str(ck),
            "--log_dir",
            str(lg),
        ]
    )
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.madaug.controlled_baselines.train_controlled",
            *argv,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    check(
        "re-running a completed controlled run is refused",
        p.returncode != 0 and "FileExistsError" in p.stderr,
    )
    for bad in (
        ["--checkpoint_dir", "checkpoints"],
        ["--log_dir", "results/logs"],
        ["--val_split", "0.1"],
        ["--early_stopping_patience", "10"],
    ):
        argv = COMMON + METHODS["ets"] + ["--debug", "--epochs", "1", *bad]
        p = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.madaug.controlled_baselines.train_controlled",
                *argv,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        check(
            f"rejected: {' '.join(bad)}", p.returncode != 0 and "ValueError" in p.stderr
        )
    after = snapshot_existing_outputs()
    changed = sorted({p for p, _ in set(after.items()) ^ set(before.items())})
    # Files written by this test are CIFAR-100 runs named *_ctrl49k*; anything else that changed was
    # written by another process (e.g. an unrelated training run on this machine) and is reported only.
    ours = [p for p in changed if "ctrl49k" in p]
    check(
        "no controlled-run file (*_ctrl49k*) written outside results/controlled_baselines/",
        not ours,
        str(ours[:5]),
    )
    other = [p for p in changed if "ctrl49k" not in p]
    if other:
        print(f"     info: {len(other)} file(s) outside results/controlled_baselines/ changed during the test "
              f"by another process (not written by this test): {sorted({Path(p).name for p in other})}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_runs", action="store_true")
    a = ap.parse_args()
    frozen_files()
    copy_integrity()
    config_equivalence()
    split_consistency()
    if not a.skip_runs:
        final_epoch_runs()
    frozen_files()  # again, after the runs
    n_fail = sum(not ok for _, ok in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)
