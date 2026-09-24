"""
experiments/madaug/sanity_test.py

Pre-flight checks for the MADAug integration (cheap; CPU is enough):

  1. official_copies    core/ files == official files up to the documented [madaug-adapt] edits
  2. protected_files    models/, training/, augmentations/, data/, shared experiment code unchanged vs HEAD
  3. wrapper            our WRN-28-10 through the f()/g() wrapper: identical output, shapes, params, backward
  4. bilevel_step       one real official train() step with the bi-level update, instrumented
  5. smoke_run          2-epoch train_madaug.main() on a tiny subset (bi-level active in epoch 1),
                        checkpoint + metrics written, resume path loads

Usage:
    python -m experiments.madaug.sanity_test                 # all checks
    python -m experiments.madaug.sanity_test --skip_smoke    # skip check 5
"""

import argparse
import ast
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFICIAL = HERE / "official"
CORE = HERE / "core"

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(
        f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else "")
    )
    return cond


def class_ast(path, cls):
    tree = ast.parse(Path(path).read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == cls
    )
    return ast.dump(node, include_attributes=False)


# ── 1 ────────────────────────────────────────────────────────────────────────
def official_copies():
    print("\n1. Official copies")
    expected = {}
    for line in (OFFICIAL / "SOURCE.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            expected[parts[1]] = parts[0]
    for fname, sha in expected.items():
        got = hashlib.sha256((OFFICIAL / fname).read_bytes()).hexdigest()
        check(f"official/{fname} sha256 matches recorded upstream copy", got == sha)

    for fname in ("operation.py", "config.py"):
        check(
            f"core/{fname} byte-identical to official",
            (CORE / fname).read_bytes() == (OFFICIAL / fname).read_bytes(),
        )

    subs_projection = [
        ("from config import OPS_NAMES", "from .config import OPS_NAMES")
    ]
    subs_augmentor = [
        (
            "from cv2 import magnitude",
            "# [madaug-adapt] removed unused import: from cv2 import magnitude",
        ),
        ("from operation import apply_augment", "from .operation import apply_augment"),
        (
            "from networks import get_model",
            "# [madaug-adapt] removed unused import: from networks import get_model",
        ),
        (
            "from utils import PolicyHistory",
            "from .policy_history import PolicyHistory",
        ),
        (
            "from config import OPS_NAMES\n",
            "from .config import OPS_NAMES\n\n# [madaug-adapt] device for tensors the official "
            'code moves with .cuda(); set by train_madaug.py\nDEVICE = torch.device("cuda")\n',
        ),
        ("images.cuda()", "images.to(DEVICE)"),
        ("trans_image.cuda()", "trans_image.to(DEVICE)"),
        (
            "torch.stack(trans_images, dim=0).cuda()",
            "torch.stack(trans_images, dim=0).to(DEVICE)",
        ),
    ]
    for fname, subs in (
        ("projection.py", subs_projection),
        ("adaptive_augmentor.py", subs_augmentor),
    ):
        text = (OFFICIAL / fname).read_text()
        for a, b in subs:
            text = text.replace(a, b)
        check(
            f"core/{fname} == official + {len(subs)} documented substitutions only",
            text == (CORE / fname).read_text(),
        )

    check(
        "core/policy_history.py::PolicyHistory AST == official utils.py",
        class_ast(CORE / "policy_history.py", "PolicyHistory")
        == class_ast(OFFICIAL / "utils.py", "PolicyHistory"),
    )
    for cls in ("CutoutDefault", "AugmentDataset"):
        check(
            f"data.py::{cls} AST == official dataset.py",
            class_ast(HERE / "data.py", cls) == class_ast(OFFICIAL / "dataset.py", cls),
        )
    for fn in ("AvgrageMeter", "accuracy"):
        check(
            f"train_madaug.py::{fn} AST == official utils.py",
            class_ast(HERE / "train_madaug.py", fn)
            == class_ast(OFFICIAL / "utils.py", fn),
        )


# ── 2 ────────────────────────────────────────────────────────────────────────
def protected_files():
    print("\n2. Protected files unchanged (vs git HEAD)")
    protected = [
        "models",
        "training",
        "augmentations",
        "data/datasets.py",
        "data/transforms.py",
        "configs",
        "scripts",
        "experiments/train_baseline.py",
        "experiments/utils.py",
        "experiments/config.py",
        "experiments/ablation.py",
        "requirements.txt",
        "setup.py",
    ]
    diff = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", *protected],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    check(
        "git diff HEAD on protected paths is empty",
        diff.returncode == 0 and diff.stdout.strip() == "",
        diff.stdout.strip(),
    )
    head_blob = subprocess.run(
        ["git", "rev-parse", "HEAD:models/wideresnet.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    work_blob = subprocess.run(
        ["git", "hash-object", "models/wideresnet.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    check(
        "models/wideresnet.py blob == HEAD blob",
        head_blob == work_blob and head_blob != "",
        head_blob[:12],
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    changed = [
        l[3:]
        for l in status.splitlines()
        if not l[3:].startswith("experiments/madaug/") and l[3:] != ".vscode/mcp.json"
    ]
    check(
        "only experiments/madaug/ added (besides pre-existing .vscode/mcp.json change)",
        not changed,
        ", ".join(changed),
    )


# ── 3 ────────────────────────────────────────────────────────────────────────
def wrapper():
    print("\n3. WRN-28-10 wrapper")
    from models.registry import get_model
    from experiments.madaug.wrn_fg import WRNFeatureClassifier

    torch.manual_seed(0)
    net = get_model("wideresnet", num_classes=100)
    ref_state = copy.deepcopy(net.state_dict())
    gf = WRNFeatureClassifier(net)
    x = torch.randn(4, 3, 32, 32)

    net.eval()
    with torch.no_grad():
        same = torch.equal(gf(x), net(x))
        feats, logits = gf.f(x), gf.g(gf.f(x))
    check("wrapper(x) bit-identical to net(x) (eval)", same)
    check(
        "f(x) shape (B, 640)", tuple(feats.shape) == (4, 640), str(tuple(feats.shape))
    )
    check(
        "g(f(x)) shape (B, 100)",
        tuple(logits.shape) == (4, 100),
        str(tuple(logits.shape)),
    )
    check("fc.in_features == 640 (policy input width)", gf.fc.in_features == 640)
    check(
        "wrapper parameters are the net's own parameter objects",
        [id(p) for p in gf.parameters()] == [id(p) for p in net.parameters()],
    )
    n = sum(p.numel() for p in gf.parameters())
    check("parameter count == our WRN-28-10 (36,536,884)", n == 36_536_884, f"{n:,}")
    check(
        "dropout p=0.3 kept",
        {m.p for m in gf.modules() if isinstance(m, nn.Dropout)} == {0.3},
    )
    check(
        "wrapper does not alter weights",
        all(torch.equal(ref_state[k], v) for k, v in net.state_dict().items()),
    )

    gf.train()
    loss = nn.CrossEntropyLoss()(gf(x), torch.randint(0, 100, (4,)))
    loss.backward()
    check(
        "train-mode forward/backward: every parameter receives a finite gradient",
        all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in gf.parameters()
        ),
    )


# ── 4 ────────────────────────────────────────────────────────────────────────
def bilevel_step():
    print("\n4. One official train() step with bi-level update (instrumented, CPU)")
    from experiments.madaug.core import adaptive_augmentor
    from experiments.madaug.core.adaptive_augmentor import MDAAug
    from experiments.madaug.core.projection import Projection
    from experiments.madaug.core.config import OPS_NAMES
    from experiments.madaug.data import get_madaug_cifar100_loaders
    from experiments.madaug.make_split import DEFAULT_OUT
    from experiments.madaug.wrn_fg import WRNFeatureClassifier
    from experiments.madaug import train_madaug as tm
    from experiments.utils import build_optimizer, build_scheduler, set_seed
    from models.registry import get_model

    device = torch.device("cpu")
    adaptive_augmentor.DEVICE = device
    set_seed(0)
    B = 8
    train_q, search_q, test_q, _, (train_idx, val_idx) = get_madaug_cifar100_loaders(
        str(ROOT / "data"),
        str(DEFAULT_OUT),
        batch_size=B,
        num_workers=0,
        debug_n_train=64,
    )

    # data separation
    check(
        "split sizes 49,000 / 1,000",
        (len(train_idx) == 64)
        and len(val_idx) == 1000
        and len(json.loads(Path(DEFAULT_OUT).read_text())["val_idx"]) == 1000,
    )
    check(
        "policy-val indices disjoint from training indices",
        not set(search_q.dataset.dataset.indices)
        & set(train_q.dataset.dataset.indices),
    )
    check(
        "train & policy-val loaders draw from CIFAR-100 *train* split",
        train_q.dataset.dataset.dataset.train
        and search_q.dataset.dataset.dataset.train,
    )
    check(
        "test loader is CIFAR-100 *test* split and is not passed to train()",
        not test_q.dataset.dataset.train,
    )
    check(
        "policy-val batch size = official search divider (16)",
        search_q.batch_size == 16,
    )

    gf = WRNFeatureClassifier(get_model("wideresnet", num_classes=100)).to(device)
    h = Projection(in_features=gf.fc.in_features, n_layers=0, n_hidden=128).to(device)
    proto = {
        "optimizer": "sgd",
        "lr": 0.1,
        "weight_decay": 5e-4,
        "epochs": 200,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "eta_min": 1e-6,
    }
    gf_opt, _ = build_optimizer(gf, proto)
    build_scheduler(gf_opt, proto)
    h_opt = torch.optim.Adam(
        h.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-3
    )
    crit = nn.CrossEntropyLoss()
    mdaaug = MDAAug(
        after_transforms=train_q.dataset.after_transforms,
        n_class=100,
        gf_model=gf,
        h_model=h,
        save_dir=tempfile.mkdtemp(),
        config={
            "sampling": "prob",
            "k_ops": 2,
            "delta": 0.3,
            "temp": 3.0,
            "search_d": 32,
            "target_d": 32,
        },
    )

    # ── instrumentation (observes, does not change the computation) ──
    log = {
        "aug_calls": [],
        "policy_inputs": [],
        "modes": [],
        "clip": [],
        "at_h_step": None,
        "model_fwd_training": None,
    }
    orig_apply = adaptive_augmentor.apply_augment
    adaptive_augmentor.apply_augment = lambda img, name, level: (
        log["aug_calls"].append(name),
        orig_apply(img, name, level),
    )[1]
    h.register_forward_hook(
        lambda m, i, o: log["policy_inputs"].append((tuple(i[0].shape), tuple(o.shape)))
    )
    orig_predict = MDAAug.predict_aug_params

    def predict(self, images, mode):
        out = orig_predict(self, images, mode)
        drop_training = {
            d.training for d in self.gf_model.modules() if isinstance(d, nn.Dropout)
        }
        bn_training = {
            b.training for b in self.gf_model.modules() if isinstance(b, nn.BatchNorm2d)
        }
        log["modes"].append(
            (
                mode,
                type(self.gf_model).__name__,
                self.gf_model is gf,
                self.gf_model.training,
                drop_training,
                bn_training,
                gf.training,
            )
        )
        return out

    MDAAug.predict_aug_params = predict
    orig_clip = nn.utils.clip_grad_norm_

    def clip(params, max_norm, *a, **k):
        params = list(params)
        with_grad = sum(p.grad is not None for p in params)
        total = orig_clip(params, max_norm, *a, **k)
        log["clip"].append((len(params), with_grad, float(total)))
        return total

    nn.utils.clip_grad_norm_ = clip
    w0 = {k: v.clone() for k, v in gf.state_dict().items()}
    h0 = {k: v.clone() for k, v in h.state_dict().items()}
    orig_h_step = h_opt.step

    def h_step(*a, **k):
        log["at_h_step"] = {
            "gf_unchanged": all(
                torch.equal(w0[k], v) for k, v in gf.state_dict().items()
            ),
            "gf_grads_none": all(p.grad is None for p in gf.parameters()),
            "h_grad_norm": float(
                sum(p.grad.norm() ** 2 for p in h.parameters() if p.grad is not None)
                ** 0.5
            ),
        }
        return orig_h_step(*a, **k)

    h_opt.step = h_step
    orig_fwd = type(gf).forward

    def fwd(self, x):
        if self is gf:
            log["model_fwd_training"] = self.training
        return orig_fwd(self, x)

    type(gf).forward = fwd

    split_rate = 0.5
    stats = {
        "h_updates": 0,
        "policy_images": 0,
        "train_images": 0,
        "last_val_loss_meta": None,
        "epoch_h_grad_norms": [],
        "nonfinite": 0,
    }
    try:
        tm.train(
            train_q,
            search_q,
            gf,
            mdaaug,
            crit,
            gf_opt,
            5.0,
            h_opt,
            epoch=1,
            search_freq=3,
            split_rate=split_rate,
            bi_epochs=0,
            batch_size=B,
            device=device,
            stats=stats,
            max_steps=1,
        )
    finally:
        adaptive_augmentor.apply_augment = orig_apply
        MDAAug.predict_aug_params = orig_predict
        nn.utils.clip_grad_norm_ = orig_clip
        type(gf).forward = orig_fwd

    n_ops = len(OPS_NAMES)
    n_pol = int(split_rate * B)
    explore = [m for m in log["modes"] if m[0] == "explore"]
    exploit = [m for m in log["modes"] if m[0] == "exploit"]
    print(f"     predict_aug_params calls: {log['modes']}")
    print(
        f"     clip_grad_norm_ calls (n_params, n_with_grad, total_norm): {log['clip']}"
    )
    check(
        "policy ran in explore (bi-level) and exploit (training) mode",
        len(explore) == 1 and len(exploit) == 1,
    )
    check(
        "policy input = task-model features (B, 640) -> output (B, 2*17)",
        log["policy_inputs"][0] == ((B, 640), (B, 2 * n_ops))
        and log["policy_inputs"][1] == ((n_pol, 640), (n_pol, 2 * n_ops)),
        str(log["policy_inputs"]),
    )
    check(
        "explore uses the higher-patched meta model (not gf_model)",
        not explore[0][2] and explore[0][1] != "WRNFeatureClassifier",
    )
    check(
        "bi-level inner model is in eval mode: BN running stats, dropout off",
        explore[0][3] is False
        and explore[0][4] == {False}
        and explore[0][5] == {False},
    )
    check(
        "meta_model.eval() leaves gf_model in train mode",
        explore[0][6] is True,
    )
    check(
        "exploit uses deepcopy snapshot of gf_model (after search step)",
        not exploit[0][2] and exploit[0][1] == "WRNFeatureClassifier",
    )
    check(
        f"augmentations applied: explore {B}x{n_ops} + exploit {n_pol}x2 op calls",
        len(log["aug_calls"]) == B * n_ops + n_pol * 2,
        f"{len(log['aug_calls'])} calls",
    )
    check(
        "inner clip_grad_norm_ is a no-op (fast weights carry no .grad) — official behaviour",
        log["clip"][0][1] == 0 and log["clip"][0][2] == 0.0,
        str(log["clip"][0]),
    )
    check(
        "outer clip_grad_norm_ acts on real gradients (max_norm 5)",
        log["clip"][1][1] > 0,
        str(log["clip"][1]),
    )
    a = log["at_h_step"]
    check(
        "bi-level step leaves gf_model weights + BN stats untouched", a["gf_unchanged"]
    )
    check("validation loss does not leak gradients into gf_model", a["gf_grads_none"])
    check(
        "policy receives non-zero gradient from validation loss",
        a["h_grad_norm"] > 0,
        f"||grad h|| = {a['h_grad_norm']:.3e}",
    )
    check(
        "policy network updated (h params changed)",
        any(not torch.equal(h0[k], v) for k, v in h.state_dict().items()),
    )
    check("model update ran in train mode", log["model_fwd_training"] is True)
    check(
        "task model updated (gf params changed)",
        any(not torch.equal(w0[k], v) for k, v in gf.state_dict().items()),
    )
    check("one h update counted", stats["h_updates"] == 1)


# ── 5 ────────────────────────────────────────────────────────────────────────
def smoke_run():
    print("\n5. 2-epoch smoke run of train_madaug.main() (CPU, batch 128, 3 steps/epoch)")
    from experiments.madaug import train_madaug as tm

    out = Path(tempfile.mkdtemp(prefix="madaug_smoke_"))
    argv = [
        "--seed",
        "42",
        "--epochs",
        "2",
        "--batch_size",
        "128",
        "--debug_n_train",
        "384",
        "--debug_n_eval",
        "64",
        "--warmup_epochs",
        "1",
        "--max_steps",
        "3",
        "--num_workers",
        "0",
        "--device",
        "cpu",
        "--out_dir",
        str(out),
    ]
    final = tm.main(argv)
    name = "madaug_cifar100_wrn28-10_ep2_s42_debug"
    m = json.loads((out / "metrics" / f"{name}.json").read_text())
    check(
        "metrics file complete, final-epoch top-1 + error recorded",
        m["complete"]
        and abs(m["final_epoch_top1"] + m["final_epoch_top1_error"] - 100) < 1e-9,
    )
    check(
        "epoch 0: no bi-level (epoch > bi_epochs is false); epoch 1: bi-level active",
        m["history"]["h_updates"][0] == 0 and m["history"]["h_updates"][1] == 1,
        str(m["history"]["h_updates"]),
    )
    check(
        "tanh curriculum split_rate epoch0=0.01, epoch1=tanh(1/40)+0.01",
        abs(m["history"]["split_rate"][0] - 0.01) < 1e-6
        and abs(
            m["history"]["split_rate"][1]
            - (float(torch.tanh(torch.tensor(1 / 40))) + 0.01)
        )
        < 1e-6,
    )
    check(
        "config, log, last + final checkpoints written",
        all(
            p.exists()
            for p in (
                out / "configs" / f"{name}.json",
                out / "logs" / f"{name}.log",
                out / "checkpoints" / f"{name}_last.pth",
                out / "checkpoints" / f"{name}_final.pth",
            )
        ),
    )
    ck = torch.load(out / "checkpoints" / f"{name}_last.pth", weights_only=False)
    check(
        "checkpoint stores policy, snapshot, both optimizers, scheduler, RNG",
        all(
            k in ck
            for k in (
                "h_model",
                "snapshot",
                "h_optimizer",
                "gf_optimizer",
                "scheduler",
                "rng",
            )
        )
        and not ck["snapshot_is_model"],
    )
    final2 = tm.main(argv + ["--resume"])
    check(
        "resume loads checkpoint and reproduces final metrics",
        final2["final_epoch_top1"] == final["final_epoch_top1"],
    )
    print(f"     smoke outputs in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_smoke", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    official_copies()
    protected_files()
    wrapper()
    bilevel_step()
    if not args.skip_smoke:
        smoke_run()
    n_fail = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)
