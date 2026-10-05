"""
analysis/madaug_policy_analysis.py

Post-hoc analysis of the five controlled MADAug runs (CIFAR-100, WRN-28-10, 200 epochs).
Read-only: loads the saved configs, metrics and *_final.pth checkpoints; trains nothing and
writes only to --out_dir.

Questions answered (one section of the output each):
  1. Config identity  — did all seeds run with identical settings (only the seed differs)?
  2. Clean accuracy   — is the low final train accuracy of some seeds (measured on
                        policy-augmented images) due to harder augmentation or to underfitting?
                        -> accuracy of each final model on the CLEAN 49k training images.
  3. Learned policy   — did the seeds converge to different policies? -> each seed's policy
                        network on the 1,000 validation images: op-selection probabilities,
                        magnitudes, expected magnitude, selection entropy.
  4. Policy hardness  — 5x5 matrix: accuracy drop on the validation images when the policy of
                        seed i augments the inputs of the model of seed j. A harder policy
                        shows up as a column effect across all models.
  5. Dynamics         — per-epoch test accuracy, augmented train accuracy, policy gradient
                        norm, and the across-seed SD of test accuracy.
  6. Variance test    — Brown-Forsythe (median-centred Levene) MADAug vs each controlled
                        baseline found in results/controlled_baselines/metrics/ (n=5 each),
                        plus Welch's t-test on the means.

The policy is queried exactly as in training (core/adaptive_augmentor.py::predict_aug_params,
'exploit' mode): input = un-normalised ToTensor image (the official search=True path), output
magnitudes = sigmoid, weights = softmax(w / temperature).

Usage (on the machine that holds the checkpoints, GPU recommended):
    python analysis/madaug_policy_analysis.py
    python analysis/madaug_policy_analysis.py --results_dir results/madaug --out_dir results/madaug_analysis
"""

import argparse
import csv
import glob
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import CIFAR100_MEAN, CIFAR100_STD  # noqa: E402
from experiments.madaug.core import adaptive_augmentor  # noqa: E402
from experiments.madaug.core.adaptive_augmentor import MDAAug  # noqa: E402
from experiments.madaug.core.config import OPS_NAMES  # noqa: E402
from experiments.madaug.core.projection import Projection  # noqa: E402
from experiments.madaug.make_split import load_split  # noqa: E402
from experiments.madaug.wrn_fg import WRNFeatureClassifier  # noqa: E402
from models.registry import get_model  # noqa: E402

RUN_RE = re.compile(r"^madaug_cifar100_wrn28-10_ep200_s(\d+)(_sb\d+)?$")
# args that legitimately differ between seeds of the same configuration
PER_RUN_ARGS = {"seed", "resume"}
# validated categorical palette (dataviz reference palette, slots 1-5, fixed order by seed)
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/madaug")
    p.add_argument("--ctrl_dir", default="results/controlled_baselines")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="results/madaug_analysis")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--hardness_seed",
        type=int,
        default=0,
        help="RNG seed for policy sampling in section 4",
    )
    p.add_argument(
        "--n_train_eval",
        type=int,
        default=0,
        help="smoke tests only: clean train acc on the first N images (0 = all 49k)",
    )
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── 1. runs + config identity ────────────────────────────────────────────────
def discover_runs(results_dir: Path):
    runs = {}
    for f in sorted(
        (results_dir / "configs").glob("madaug_cifar100_wrn28-10_ep200_s*.json")
    ):
        m = RUN_RE.match(f.stem)
        if not m:  # skips _debug / _smoke runs
            continue
        runs[int(m.group(1))] = f.stem
    if not runs:
        sys.exit(f"No final MADAug configs found in {results_dir / 'configs'}")
    return dict(sorted(runs.items()))


def config_identity(results_dir, runs):
    cfgs = {
        s: json.loads((results_dir / "configs" / f"{n}.json").read_text())
        for s, n in runs.items()
    }
    ref_seed = next(iter(cfgs))
    ref = cfgs[ref_seed]
    diffs = {}
    for s, c in cfgs.items():
        d = {}
        keys = set(ref["args"]) | set(c["args"])
        for k in sorted(keys - PER_RUN_ARGS):
            if ref["args"].get(k) != c["args"].get(k):
                d[f"args.{k}"] = (ref["args"].get(k), c["args"].get(k))
        for k in ("madaug", "split", "optimizer", "scheduler", "loss", "amp", "model"):
            if json.dumps(ref.get(k), sort_keys=True) != json.dumps(
                c.get(k), sort_keys=True
            ):
                d[k] = "differs"
        if ref["git"].get("commit") != c["git"].get("commit"):
            d["git.commit"] = (ref["git"].get("commit"), c["git"].get("commit"))
        diffs[s] = d
    info = {
        s: {
            "run_name": runs[s],
            "git_commit": c["git"].get("commit"),
            "git_dirty": c["git"].get("dirty"),
            "gpu": c["versions"].get("gpu"),
            "torch": c["versions"].get("torch"),
            "search_batch_size": c["madaug"].get("search_batch_size"),
        }
        for s, c in cfgs.items()
    }
    return cfgs, diffs, info


# ── models / data ────────────────────────────────────────────────────────────
def load_models(results_dir, run, cfg, device):
    ck = torch.load(
        results_dir / "checkpoints" / f"{run}_final.pth",
        map_location=device,
        weights_only=False,
    )
    gf = WRNFeatureClassifier(get_model("wideresnet", num_classes=100)).to(device)
    gf.load_state_dict(ck["gf_model"])
    h = Projection(
        in_features=gf.fc.in_features,
        n_layers=cfg["args"]["n_proj_layer"],
        n_hidden=128,
    ).to(device)
    h.load_state_dict(ck["h_model"])
    gf.eval()
    h.eval()
    return gf, h


@torch.no_grad()
def accuracy(model, loader, device):
    correct = total = 0
    for x, y in loader:
        correct += model(x.to(device)).argmax(1).eq(y.to(device)).sum().item()
        total += y.numel()
    return 100.0 * correct / total


normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)


# ── 3. learned policy ────────────────────────────────────────────────────────
@torch.no_grad()
def policy_stats(gf, h, cfg, raw_val, device):
    """Exactly predict_aug_params('exploit'): raw ToTensor input, sigmoid / softmax(w/T)."""
    T = cfg["madaug"]["temp"]
    n_ops = len(OPS_NAMES)
    mags, ws = [], []
    for x, _ in DataLoader(raw_val, batch_size=256):
        a = h(gf.f(x.to(device)))
        m, w = torch.split(a, n_ops, dim=1)
        mags.append(torch.sigmoid(m).cpu())
        ws.append(torch.softmax(w / T, dim=-1).cpu())
    m, w = torch.cat(mags), torch.cat(ws)
    ent = -(w * w.clamp_min(1e-12).log()).sum(1) / np.log(n_ops)
    return {
        "op_prob": w.mean(0).tolist(),
        "op_magnitude": m.mean(0).tolist(),
        "expected_magnitude": float((w * m).sum(1).mean()),
        "mean_magnitude": float(m.mean()),
        "selection_entropy_norm": float(ent.mean()),
        "top_op": OPS_NAMES[int(w.mean(0).argmax())],
        "top_op_prob": float(w.mean(0).max()),
    }


# ── 4. policy hardness matrix ────────────────────────────────────────────────
@torch.no_grad()
def augmented_accuracy(policy_gf, policy_h, eval_model, cfg, raw_val, device, seed):
    """Val accuracy of eval_model on images augmented by (policy_gf, policy_h) as in training
    ('exploit': multinomial k_ops, magnitude perturbation delta), after_transforms without Cutout
    so only the learned policy is measured."""
    after = transforms.Compose([transforms.ToTensor(), normalize])
    aug = MDAAug(
        after_transforms=after,
        n_class=100,
        gf_model=policy_gf,
        h_model=policy_h,
        config=cfg["madaug"],
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    correct = total = 0
    for x, y in DataLoader(raw_val, batch_size=128):
        xa = aug(x, mode="exploit")
        correct += eval_model(xa).argmax(1).eq(y.to(device)).sum().item()
        total += y.numel()
    policy_h.eval()  # predict_aug_params switches h to train(); projection has no BN/dropout
    return 100.0 * correct / total


# ── 6. variance test ─────────────────────────────────────────────────────────
def controlled_baselines(ctrl_dir: Path):
    out = {}
    for f in glob.glob(str(ctrl_dir / "metrics" / "*_ctrl49k.json")):
        d = json.loads(Path(f).read_text())
        if (
            d.get("complete")
            and d.get("epochs") == 200
            and "debug" not in d["run_name"]
        ):
            out.setdefault(d["method"], {})[d["seed"]] = d["final_epoch_top1"]
    return out


def tests(madaug_acc, baselines):
    from scipy import stats

    res = {}
    for method, accs in sorted(baselines.items()):
        b = list(accs.values())
        if len(b) < 3:
            continue
        bf = stats.levene(madaug_acc, b, center="median")
        welch = stats.ttest_ind(madaug_acc, b, equal_var=False)
        res[method] = {
            "n": len(b),
            "mean": float(np.mean(b)),
            "sd": float(np.std(b, ddof=1)),
            "brown_forsythe_W": float(bf.statistic),
            "brown_forsythe_p": float(bf.pvalue),
            "welch_t": float(welch.statistic),
            "welch_p": float(welch.pvalue),
        }
    return res


# ── figures / tables ─────────────────────────────────────────────────────────
def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(which="both", colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def plot_dynamics(hist, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = list(hist)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
    panels = [
        ("test_acc", "Test top-1 (%)", "(a) Test accuracy", False),
        (
            "train_acc",
            "Train top-1 on augmented images (%)",
            "(b) Augmented train accuracy",
            False,
        ),
        (
            "h_grad_norm_mean",
            "Policy gradient norm (log)",
            "(c) Policy-network gradient norm",
            True,
        ),
    ]
    for ax, (key, ylab, title, log) in zip(axes, panels):
        for i, s in enumerate(seeds):
            ep = np.array(hist[s]["epoch"])
            y = np.array(
                [np.nan if v is None else v for v in hist[s][key]], dtype=float
            )
            ax.plot(ep, y, color=SEED_COLORS[i], linewidth=1.6, label=f"seed {s}")
            last = np.where(np.isfinite(y))[0]
            if len(last):
                ax.plot(
                    ep[last[-1]],
                    y[last[-1]],
                    "o",
                    ms=5,
                    color=SEED_COLORS[i],
                    markeredgecolor="white",
                    markeredgewidth=1.2,
                )
                if not log:
                    ax.annotate(
                        f"s{s}",
                        (ep[last[-1]], y[last[-1]]),
                        xytext=(4, 0),
                        textcoords="offset points",
                        fontsize=7,
                        color=TEXT,
                        va="center",
                    )
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=9, color=TEXT, loc="left")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        style(ax)
    # (d) across-seed SD of test accuracy
    ax = axes[3]
    n = min(len(hist[s]["test_acc"]) for s in seeds)
    sd = np.std(np.array([hist[s]["test_acc"][:n] for s in seeds]), axis=0, ddof=1)
    ax.plot(np.arange(n), sd, color=TEXT, linewidth=1.6)
    ax.set_title(
        "(d) Across-seed SD of test accuracy", fontsize=9, color=TEXT, loc="left"
    )
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("SD (pp)", fontsize=8)
    style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(seeds),
        fontsize=8,
        frameon=False,
    )
    for ext in ("pdf", "png"):
        fig.savefig(out / f"madaug_seed_dynamics.{ext}", dpi=200)
    plt.close(fig)


def plot_policy(policies, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    seeds = list(policies)
    cmap = LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), constrained_layout=True)
    for ax, key, title, fmt in (
        (
            axes[0],
            "op_prob",
            "Mean op-selection probability (validation images)",
            "{:.2f}",
        ),
        (
            axes[1],
            "op_magnitude",
            "Mean op magnitude (0-1, validation images)",
            "{:.2f}",
        ),
    ):
        M = np.array([policies[s][key] for s in seeds])
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=M.max())
        ax.set_yticks(
            range(len(seeds)), [f"seed {s}" for s in seeds], fontsize=8, color=TEXT
        )
        ax.set_xticks(
            range(len(OPS_NAMES)),
            OPS_NAMES,
            fontsize=7,
            rotation=45,
            ha="right",
            color=TEXT,
        )
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(
                    j,
                    i,
                    fmt.format(M[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if M[i, j] > 0.55 * M.max() else TEXT,
                )
        ax.set_title(title, fontsize=9, color=TEXT, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"madaug_policy_per_seed.{ext}", dpi=200)
    plt.close(fig)


def latex_table(rows, out):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Controlled MADAug runs per seed (CIFAR-100, WRN-28-10, 200 epochs). "
        r"Augmented train accuracy is measured on policy-augmented images during the final epoch; "
        r"clean accuracies use unaugmented images. Expected magnitude $=\sum_k w_k m_k$ of the "
        r"learned policy on the validation images; hardness = validation accuracy drop when the "
        r"seed's own policy augments the inputs.}",
        r"\label{tab:madaug_seed_analysis}",
        r"\begin{tabular}{rcccccc}",
        r"\toprule",
        r"\textbf{Seed} & \textbf{Test} & \textbf{Train (aug.)} & \textbf{Train (clean)} & "
        r"\textbf{Exp.\ magnitude} & \textbf{Sel.\ entropy} & \textbf{Hardness (pp)} \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['seed']} & {r['test_acc']:.2f} & {r['train_acc_augmented']:.2f} & "
            f"{r['train_acc_clean']:.2f} & {r['expected_magnitude']:.3f} & "
            f"{r['selection_entropy_norm']:.3f} & {r['own_policy_drop']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "madaug_seed_analysis_table.tex").write_text("\n".join(lines) + "\n")


def main():
    a = get_args()
    device = pick_device(a.device)
    adaptive_augmentor.DEVICE = device
    rdir, out = Path(a.results_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(rdir)
    print(f"Runs: {runs}  | device {device}")

    # 1. config identity
    cfgs, diffs, info = config_identity(rdir, runs)
    identical = all(not d for d in diffs.values())
    print(
        "\n1. CONFIG IDENTITY:",
        "IDENTICAL apart from the seed" if identical else "DIFFERENCES FOUND",
    )
    for s, d in diffs.items():
        print(
            f"   s{s}: {info[s]['run_name']} commit {str(info[s]['git_commit'])[:7]} "
            f"dirty={info[s]['git_dirty']} gpu={info[s]['gpu']} search_batch={info[s]['search_batch_size']}"
            + (f"  DIFF {d}" if d else "")
        )

    # metrics
    metrics = {
        s: json.loads((rdir / "metrics" / f"{n}.json").read_text())
        for s, n in runs.items()
    }
    incomplete = [s for s, m in metrics.items() if not m.get("complete")]
    if incomplete:
        print(f"   WARNING: incomplete runs {incomplete}")

    # data (split from the runs' own config)
    split_file = cfgs[next(iter(cfgs))]["split"]["file"]
    if not Path(
        split_file
    ).exists():  # absolute path from the cluster; fall back to repo copy
        split_file = str(
            Path(__file__).resolve().parents[1]
            / "experiments/madaug/splits/cifar100_val1000.json"
        )
    train_idx, val_idx = load_split(split_file)
    clean_tf = transforms.Compose([transforms.ToTensor(), normalize])
    full_clean = torchvision.datasets.CIFAR100(
        a.data_root, train=True, download=True, transform=clean_tf
    )
    full_raw = torchvision.datasets.CIFAR100(
        a.data_root, train=True, download=True, transform=transforms.ToTensor()
    )
    eval_idx = train_idx[: a.n_train_eval] if a.n_train_eval else train_idx
    train_clean = DataLoader(
        Subset(full_clean, eval_idx), batch_size=a.batch_size, num_workers=4
    )
    val_clean = DataLoader(Subset(full_clean, val_idx), batch_size=a.batch_size)
    raw_val = Subset(full_raw, val_idx)

    models, rows, policies = {}, [], {}
    print("\n2./3. CLEAN ACCURACY AND LEARNED POLICY")
    for s, n in runs.items():
        gf, h = load_models(rdir, n, cfgs[s], device)
        models[s] = (gf, h)
        hist = metrics[s]["history"]
        pol = policy_stats(gf, h, cfgs[s], raw_val, device)
        policies[s] = pol
        row = {
            "seed": s,
            "test_acc": hist["test_acc"][-1],
            "val_acc": hist["val_acc"][-1],
            "train_acc_augmented": hist["train_acc"][-1],
            "train_loss_augmented": hist["train_loss"][-1],
            "train_acc_clean": accuracy(gf, train_clean, device),
            "val_acc_clean_recomputed": accuracy(gf, val_clean, device),
            "h_grad_norm_last": hist["h_grad_norm_mean"][-1],
            **{k: v for k, v in pol.items() if not isinstance(v, list)},
        }
        rows.append(row)
        print(
            f"   s{s}: test {row['test_acc']:.2f} | train aug {row['train_acc_augmented']:.2f} "
            f"clean {row['train_acc_clean']:.2f} | E[mag] {pol['expected_magnitude']:.3f} "
            f"entropy {pol['selection_entropy_norm']:.3f} top {pol['top_op']} ({pol['top_op_prob']:.2f})"
        )

    # 4. hardness matrix: rows = model seed, cols = policy seed
    print("\n4. POLICY HARDNESS (val acc drop, pp; rows = model, cols = policy)")
    seeds = list(runs)
    clean_val = {s: r["val_acc_clean_recomputed"] for s, r in zip(seeds, rows)}
    H = np.zeros((len(seeds), len(seeds)))
    for i, sm in enumerate(seeds):
        for j, sp in enumerate(seeds):
            acc = augmented_accuracy(
                models[sp][0],
                models[sp][1],
                models[sm][0],
                cfgs[sp],
                raw_val,
                device,
                a.hardness_seed,
            )
            H[i, j] = clean_val[sm] - acc
        print(f"   model s{sm}: " + "  ".join(f"{v:6.2f}" for v in H[i]))
    col_mean = H.mean(0)
    print(
        "   policy column means: "
        + "  ".join(f"s{s}={v:.2f}" for s, v in zip(seeds, col_mean))
    )
    for r, i in zip(rows, range(len(seeds))):
        r["own_policy_drop"] = float(H[i, i])
        r["policy_drop_mean_over_models"] = float(col_mean[i])

    # 5./6.
    test_accs = [r["test_acc"] for r in rows]
    print(
        f"\n5. MADAug final test: {np.mean(test_accs):.2f} +- {np.std(test_accs, ddof=1):.2f} (n={len(test_accs)})"
    )
    corr = float(np.corrcoef([r["train_acc_clean"] for r in rows], test_accs)[0, 1])
    corr_aug = float(
        np.corrcoef([r["train_acc_augmented"] for r in rows], test_accs)[0, 1]
    )
    corr_mag = float(
        np.corrcoef([r["expected_magnitude"] for r in rows], test_accs)[0, 1]
    )
    print(
        f"   Pearson r (n={len(rows)}, descriptive only): test vs train-aug {corr_aug:.2f}, "
        f"test vs train-clean {corr:.2f}, test vs E[mag] {corr_mag:.2f}"
    )
    baselines = controlled_baselines(Path(a.ctrl_dir))
    t = tests(test_accs, baselines) if baselines else {}
    print(
        "\n6. VARIANCE / MEAN TESTS vs controlled baselines"
        + ("" if t else ": no controlled metrics found")
    )
    for m, r in t.items():
        print(
            f"   {m}: {r['mean']:.2f} +- {r['sd']:.2f} (n={r['n']}) | Brown-Forsythe W={r['brown_forsythe_W']:.2f} "
            f"p={r['brown_forsythe_p']:.3f} | Welch t={r['welch_t']:.2f} p={r['welch_p']:.3f}"
        )

    # outputs
    with open(out / "per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "policy_per_op.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "op", "selection_prob", "magnitude"])
        for s, p in policies.items():
            for op, pr, mg in zip(OPS_NAMES, p["op_prob"], p["op_magnitude"]):
                w.writerow([s, op, f"{pr:.5f}", f"{mg:.5f}"])
    np.savetxt(
        out / "hardness_matrix.csv",
        H,
        delimiter=",",
        fmt="%.3f",
        header="rows=model seed, cols=policy seed: " + ",".join(map(str, seeds)),
    )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "runs": runs,
                "config_identical": identical,
                "config_diffs": diffs,
                "run_info": info,
                "per_seed": rows,
                "policies": policies,
                "hardness_matrix": H.tolist(),
                "seeds": seeds,
                "madaug_mean": float(np.mean(test_accs)),
                "madaug_sd": float(np.std(test_accs, ddof=1)),
                "pearson_test_vs_train_aug": corr_aug,
                "pearson_test_vs_train_clean": corr,
                "pearson_test_vs_expected_magnitude": corr_mag,
                "controlled_baselines": baselines,
                "tests": t,
            },
            indent=2,
            default=str,
        )
    )
    plot_dynamics({s: metrics[s]["history"] for s in runs}, out)
    plot_policy(policies, out)
    latex_table(rows, out)
    print(
        f"\nWrote {out}/: summary.json, per_seed.csv, policy_per_op.csv, hardness_matrix.csv, "
        "madaug_seed_dynamics.{pdf,png}, madaug_policy_per_seed.{pdf,png}, madaug_seed_analysis_table.tex"
    )


if __name__ == "__main__":
    main()
