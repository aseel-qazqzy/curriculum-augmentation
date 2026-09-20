"""
analysis/seed_band_plot.py
Multi-seed mean±std band plot — the primary thesis committee figure.

Produces three publication-quality figures:
  fig_seed_bands_<model>.png       — val accuracy mean ± 1σ bands + best-acc bar chart
  fig_tier_zoom_<model>.png        — zoomed view of tier-transition dip/recovery (ep 10-60)
  fig_seed_distribution_<model>.png — box plot of best acc per seed (consistency check)

Data source: tries checkpoint `*_history.pt` files first (results/cluster/checkpoints),
falls back to parsing the raw cluster `.log` files (results/cluster/logs/<model>/) when
no checkpoint history is found — log epoch resolution is sparse (epoch 1 + every 10th),
so curves are linearly interpolated to per-epoch resolution before smoothing.

Usage:
    python analysis/seed_band_plot.py --model wideresnet
    python analysis/seed_band_plot.py --model resnet50
    python analysis/seed_band_plot.py --model resnet50 --checkpoint_dir /path/to/checkpoints
"""

import os
import re
import sys
import argparse
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    from scipy import stats as sstats

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

CHECKPOINT_DIR = str(_ROOT / "results" / "cluster" / "checkpoints")
LOG_DIR_ROOT = _ROOT / "results" / "cluster" / "logs"
FIGURES_DIR = str(_ROOT / "results" / "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── experiment config ─────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 3407, 1024]

# ETS fixed boundaries: t1=0.20, t2=0.45 at 100 epochs (model-independent)
TIER_EPOCHS = [20, 45]

MODEL_DISPLAY_MAP = {"wideresnet": "WideResNet-28-10", "resnet50": "ResNet-50"}
MODEL_LOG_SUBDIR = {"wideresnet": "wideresnet", "resnet50": "resnet"}
MODEL_TOKEN = {"wideresnet": "wideresnet", "resnet50": "resnet50"}
MODEL_DISPLAY = MODEL_DISPLAY_MAP["wideresnet"]  # overwritten in main() from --model

# One entry per method. `token` = unique lowercase substring to match in a filename.
METHODS = {
    "Static Mixing": {
        "token": "static",
        "color": "#0072B2",  # blue
        "ls": "-",
        "lw": 1.8,
        "zorder": 2,
    },
    "ETS (ours)": {
        "token": "ets",
        "color": "#009E73",  # green
        "ls": "--",
        "lw": 2.2,
        "zorder": 4,
    },
    "LPS (ours)": {
        "token": "lps",
        "color": "#E69F00",  # orange
        "ls": "--",
        "lw": 2.2,
        "zorder": 3,
    },
    "EGS v2 (ours)": {
        "token": "egs",
        "color": "#CC79A7",  # pink
        "ls": ":",
        "lw": 2.0,
        "zorder": 3,
    },
}

# ── matplotlib style ──────────────────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.titlepad": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "lines.linewidth": 2.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.12,
    }
)


# ── data loading ──────────────────────────────────────────────────────────────

# Dev-mode epoch line, e.g.:
#   Epoch [ 10/100] Train: 3.5359 / 18.34% | Val: 2.8590 / 28.48% | Top-5: 60.04% | LR: ...
LOG_EPOCH_RE = re.compile(
    r"Epoch \[\s*(\d+)/\d+\]"
    r"\s+Train:\s*[\d.]+\s*/\s*[\d.]+%"
    r"\s+\|\s+Val:\s*[\d.]+\s*/\s*([\d.]+)%"
)
# FINAL RESULTS block: "  Test Top-1                  77.79%"
LOG_TEST1_RE = re.compile(r"Test\s+Top-1\s*:?\s*([\d.]+)%")


def _find_by_token(directory, token, seed, ext_suffix, model_token):
    """First file in `directory` whose name contains both `token` (method) and
    `model_token` (model), restricted to this seed via the `_s{seed}_p19` marker
    every run filename carries."""
    if not directory or not os.path.isdir(directory):
        return None
    candidates = sorted(Path(directory).glob(f"*_s{seed}_p19*{ext_suffix}"))
    for c in candidates:
        low = c.name.lower()
        if token in low and model_token in low:
            return c
    return None


def _load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"    WARNING: {os.path.basename(path)}: {e}")
        return None


def _load_log_as_history(path):
    """Parse a raw cluster .log file into (history dict, test_top1). Log epoch
    rows are sparse (epoch 1 + every 10th), so the val_acc curve is linearly
    interpolated to full per-epoch resolution. test_top1 comes from the
    FINAL RESULTS block (single held-out evaluation, not used for model
    selection) — that's the unbiased number, unlike best-val-so-far."""
    text = path.read_text(errors="replace")
    rows = [(int(m.group(1)), float(m.group(2))) for m in LOG_EPOCH_RE.finditer(text)]
    if len(rows) < 2:
        return None, None
    xs = np.array([e for e, _ in rows], dtype=float)
    ys = np.array([v for _, v in rows], dtype=float)
    max_ep = int(xs.max())
    full_x = np.arange(1, max_ep + 1)
    full_y = np.interp(full_x, xs, ys)
    t1_m = LOG_TEST1_RE.search(text)
    test_top1 = float(t1_m.group(1)) if t1_m else None
    return {"val_acc": full_y.tolist()}, test_top1


def _load_checkpoint_test_top1(history_path):
    """Mirror compare_methods.py: test_top1 lives in the sibling `_best.pth`,
    not in the `_history.pt` file itself."""
    best_path = Path(str(history_path).replace("_history.pt", "_best.pth"))
    if not best_path.exists():
        return None
    try:
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        t1 = ckpt.get("test_top1")
        if t1 is None:
            return None
        return t1 * 100 if t1 <= 1.0 else t1
    except Exception:
        return None


def gather_seeds(method_name, cfg, seeds, checkpoint_dir, log_dir, model_token):
    histories, found, test_top1s = [], [], []
    for seed in seeds:
        h = None
        src = "checkpoint"
        t1 = None
        path = _find_by_token(
            checkpoint_dir, cfg["token"], seed, "history.pt", model_token
        )
        if path is not None:
            h = _load_pt(path)
            if h is not None:
                t1 = _load_checkpoint_test_top1(path)
        if h is None:
            path = _find_by_token(log_dir, cfg["token"], seed, ".log", model_token)
            if path is not None:
                h, t1 = _load_log_as_history(path)
                src = "log"
        if h is None:
            print(
                f"    [{method_name}] seed {seed}: not found (checked checkpoints + logs)"
            )
            continue
        histories.append(h)
        found.append(seed)
        test_top1s.append(t1)
        n = len(h["val_acc"])
        bv = max(h["val_acc"])
        bv = bv * 100 if bv <= 1.0 else bv
        t1_str = f"{t1:.2f}%" if t1 is not None else "—"
        print(
            f"    [{method_name}] seed {seed}: {n} ep, best_val={bv:.2f}%  test={t1_str}  [{src}] ← {path.name if hasattr(path, 'name') else os.path.basename(path)}"
        )
    if not histories:
        print(f"    [{method_name}] — no seeds found, method skipped")
    else:
        print(
            f"    [{method_name}] {len(histories)}/{len(seeds)} seeds loaded  {found}"
        )
    return histories, found, test_top1s


def compute_stats_vs_static(md, key="test_top1_by_seed"):
    """Paired (by seed) comparison of each method vs Static Mixing on `key`
    (default: Test Top-1 — the unbiased, held-out metric) — computed
    dynamically from the loaded seeds so it's correct for any model."""
    if "Static Mixing" not in md:
        return {}
    static_by_seed = md["Static Mixing"][key]

    out = {}
    for name, m in md.items():
        if name == "Static Mixing":
            continue
        by_seed = m[key]
        pairs = [(s, v) for s, v in by_seed.items() if s in static_by_seed]
        if len(pairs) < 2:
            continue
        a = np.array([static_by_seed[s] for s, _ in pairs])
        b = np.array([v for _, v in pairs])
        delta = float(b.mean() - a.mean())
        entry = {"delta": f"{delta:+.2f} pp"}
        if HAVE_SCIPY and len(pairs) >= 3:
            diff = b - a
            _, p = sstats.ttest_rel(b, a)
            sd = diff.std(ddof=1)
            d = float(diff.mean() / sd) if sd > 0 else float("nan")
            entry["p"] = "<0.001" if p < 0.001 else f"={p:.3g}"
            entry["d"] = f"{d:.2f}"
        out[name] = entry
    return out


def compute_bands(histories, metric="val_acc", smooth_w=5):
    """Return (epochs_1d, mean_1d, std_1d, list_of_smoothed_curves)."""
    arrays = []
    for h in histories:
        v = np.array(h[metric], dtype=float)
        if v.max() <= 1.0:
            v *= 100
        arrays.append(v)
    n = min(len(a) for a in arrays)
    arrays = [uniform_filter1d(a[:n], size=smooth_w) for a in arrays]
    stacked = np.stack(arrays, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=1) if len(arrays) > 1 else np.zeros(n)
    return np.arange(1, n + 1), mean, std, arrays


# ── drawing helpers ───────────────────────────────────────────────────────────

TIER_BG = ["#EBF5FB", "#E9F7EF", "#FEF9E7"]  # blue / green / yellow
TIER_LABELS = ["Tier 1\n(flip, crop)", "Tier 2\n(+jitter, rot.)", "Tier 3\n(+cutout)"]
TIER_VCOLORS = ["#27AE60", "#E67E22"]


def draw_tiers(ax, ylo, yhi, tier_epochs=TIER_EPOCHS):
    boundaries = [1] + tier_epochs + [100]
    for j in range(3):
        ax.axvspan(
            boundaries[j], boundaries[j + 1], alpha=0.18, color=TIER_BG[j], zorder=0
        )
        mid = (boundaries[j] + boundaries[j + 1]) / 2
        ax.text(
            mid,
            ylo + (yhi - ylo) * 0.015,
            TIER_LABELS[j],
            ha="center",
            fontsize=6,
            color="#999999",
            va="bottom",
            zorder=1,
        )
    for ep_t, tc in zip(tier_epochs, TIER_VCOLORS):
        ax.axvline(ep_t, color=tc, lw=1.0, ls=":", alpha=0.80, zorder=5)


def save_fig(fig, fname):
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path)
    print(f"  → saved: {fname}")


# ── Figure 1: main band plot ──────────────────────────────────────────────────


def fig_seed_bands(md, stats=None, fname="fig_seed_bands.png"):
    """
    Left  — val accuracy curves: mean line + ±1σ shaded band + ghost seed lines.
    Right — bar chart of best val accuracy with ±1σ error bars + stat annotations.
    """
    stats = stats or {}
    if not md:
        print("  fig_seed_bands: no data — skipping.")
        return

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(
        f"Validation Accuracy — Mean ± 1σ  ·  5 Seeds  ·  {MODEL_DISPLAY}  ·  CIFAR-100  ·  19-op Pool  ·  100 Epochs",
        fontsize=9,
        fontweight="bold",
    )

    # ── left: learning curves ─────────────────────────────────────────────────
    all_lo, all_hi = [], []
    for m in md.values():
        all_lo.append(float((m["mean"] - m["std"]).min()))
        all_hi.append(float((m["mean"] + m["std"]).max()))
    ylo = max(0, min(all_lo) - 2.0)
    yhi = min(100, max(all_hi) + 3.5)

    draw_tiers(ax_left, ylo, yhi)

    # tier-boundary labels above plot
    for ep_t, tc, label in zip(
        TIER_EPOCHS, TIER_VCOLORS, ["↑ Tier 2 unlocked", "↑ Tier 3 unlocked"]
    ):
        ax_left.text(
            ep_t + 0.6,
            yhi - 0.4,
            label,
            fontsize=6.5,
            color=tc,
            va="top",
            ha="left",
            zorder=6,
        )

    for method_name, m in md.items():
        cfg = m["cfg"]
        x, mean, std = m["epochs"], m["mean"], m["std"]

        # ghost individual seed lines
        for curve in m["curves"]:
            n = min(len(x), len(curve))
            ax_left.plot(
                x[:n],
                curve[:n],
                color=cfg["color"],
                lw=0.5,
                alpha=0.15,
                zorder=cfg["zorder"],
            )

        # ±1σ band
        ax_left.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.17,
            color=cfg["color"],
            zorder=cfg["zorder"],
        )

        # mean line
        n_s = m["n_seeds"]
        ax_left.plot(
            x,
            mean,
            color=cfg["color"],
            lw=cfg["lw"],
            ls=cfg["ls"],
            label=f"{method_name}  (n={n_s})",
            zorder=cfg["zorder"] + 1,
        )

        # best-epoch dot
        bi = int(np.argmax(mean))
        ax_left.scatter(
            [x[bi]],
            [mean[bi]],
            color=cfg["color"],
            s=40,
            zorder=10,
            edgecolors="white",
            linewidths=0.8,
        )

    ax_left.set_xlim(0, 102)
    ax_left.set_ylim(ylo, yhi)
    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel("Validation Accuracy (%)")
    ax_left.set_title("(a)  Validation Accuracy — Mean ± 1σ  (shaded = ±1 std)")
    ax_left.legend(loc="upper left", fontsize=7.5)
    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    # ── right: bar chart — Test Top-1 (unbiased, held-out; not the val metric
    #    used for model selection / tier scheduling) ───────────────────────────
    names = [n for n in md if md[n]["test_top1_by_seed"]]
    colors = [md[n]["cfg"]["color"] for n in names]

    test_means, test_stds = [], []
    for n in names:
        vals = np.array(list(md[n]["test_top1_by_seed"].values()))
        test_means.append(float(vals.mean()))
        test_stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

    x_pos = np.arange(len(names))
    bars = ax_right.bar(
        x_pos,
        test_means,
        yerr=test_stds,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        width=0.55,
        capsize=5,
        error_kw={"linewidth": 1.3, "ecolor": "black"},
        zorder=3,
    )

    # stagger label height whenever neighbouring bars sit within 1pp of each other,
    # so the 3-line stat annotations don't collide (common with ETS vs LPS)
    label_lift = [0.0] * len(names)
    for i in range(1, len(names)):
        if abs(test_means[i] - test_means[i - 1]) < 1.0:
            label_lift[i] = label_lift[i - 1] + 1.6

    for bar, bm, bs, name, lift in zip(bars, test_means, test_stds, names, label_lift):
        stat_line = ""
        if name in stats:
            s = stats[name]
            if "p" in s:
                stat_line = f"\np{s['p']}  d={s['d']}  {s['delta']}"
            else:
                stat_line = f"\n{s['delta']}"
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            bm + bs + 0.25 + lift,
            f"{bm:.2f}% ±{bs:.2f}{stat_line}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    short = [n.replace(" (ours)", "").replace(" v2", "") for n in names]
    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(short, rotation=12, ha="right")
    ax_right.set_ylabel("Test Top-1 Accuracy (%)")
    ax_right.set_title(
        "(b)  Test Top-1 Accuracy ± 1σ  (vs Static Mixing, paired t-test)"
    )
    y_bot = max(0, min(test_means) - 5)
    y_top = min(100, max(test_means) + max(test_stds) + max(label_lift) + 4)
    ax_right.set_ylim(y_bot, y_top)

    plt.tight_layout()
    save_fig(fig, fname)
    plt.show()


# ── Figure 2: tier-transition zoom ────────────────────────────────────────────


def fig_tier_zoom(md, zoom=(10, 62), fname="fig_tier_zoom.png"):
    """
    Zoomed view of epochs 10-62: shows the dip immediately after each tier
    unlock in curriculum methods — direct mechanistic evidence.
    """
    if not md:
        print("  fig_tier_zoom: no data — skipping.")
        return

    z0, z1 = zoom
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.suptitle(
        f"Tier-Transition Detail  ·  Epochs 10 – 62  ·  CIFAR-100  ·  {MODEL_DISPLAY}\n"
        "Each tier unlock causes a brief accuracy dip as the model adapts — "
        "then recovers above the static baseline",
        fontsize=9,
        fontweight="bold",
    )

    all_lo, all_hi = [], []
    for m in md.values():
        mask = (m["epochs"] >= z0) & (m["epochs"] <= z1)
        lo_v = float((m["mean"] - m["std"])[mask].min())
        hi_v = float((m["mean"] + m["std"])[mask].max())
        all_lo.append(lo_v)
        all_hi.append(hi_v)

    ylo = max(0, min(all_lo) - 1.5)
    yhi = min(100, max(all_hi) + 2.5)

    draw_tiers(ax, ylo, yhi)

    annotated_dip = False
    for method_name, m in md.items():
        cfg = m["cfg"]
        x, mean, std = m["epochs"], m["mean"], m["std"]
        mask = (x >= z0) & (x <= z1)
        xz, mz, sz = x[mask], mean[mask], std[mask]
        if len(xz) == 0:
            continue

        ax.fill_between(xz, mz - sz, mz + sz, alpha=0.35, color=cfg["color"])
        ax.plot(
            xz, mz, color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"], label=method_name
        )

        # annotate the tier-2 dip on the first CL method found
        if not annotated_dip and "(ours)" in method_name:
            # find the local minimum in a 10-epoch window after tier-2 transition
            t2 = TIER_EPOCHS[0]
            win_mask = (xz >= t2) & (xz <= t2 + 12)
            if win_mask.any():
                dip_rel = int(np.argmin(mz[win_mask]))
                dip_x = float(xz[win_mask][dip_rel])
                dip_y = float(mz[win_mask][dip_rel])
                ax.annotate(
                    "Accuracy dip:\nmodel adapts\nto harder ops",
                    xy=(dip_x, dip_y),
                    xytext=(dip_x + 6, dip_y - (yhi - ylo) * 0.12),
                    fontsize=7.5,
                    color=cfg["color"],
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=cfg["color"], lw=1.0),
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc="white",
                        ec=cfg["color"],
                        alpha=0.92,
                    ),
                    zorder=10,
                )
                annotated_dip = True

    # tier-boundary labels
    for ep_t, tc, label in zip(
        TIER_EPOCHS,
        TIER_VCOLORS,
        [
            "Tier 2 unlocked\n(+jitter, rotation)",
            "Tier 3 unlocked\n(+cutout, grayscale)",
        ],
    ):
        if z0 <= ep_t <= z1:
            ax.text(
                ep_t + 0.5,
                yhi - 0.3,
                label,
                fontsize=7,
                color=tc,
                va="top",
                ha="left",
                fontweight="bold",
            )

    ax.set_xlim(z0, z1)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.legend(loc="lower right", fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    plt.tight_layout()
    save_fig(fig, fname)
    plt.show()


# ── Figure 3: seed-distribution box plot ─────────────────────────────────────


def fig_seed_distribution(md, fname="fig_seed_distribution.png"):
    """
    Bar + seed-dot overlay per method.
    Bar = mean (low alpha fill).  Bold hline = mean.  Dots = individual seeds.
    Thin error bar (no caps) = ±1σ.
    """
    multi = {k: v for k, v in md.items() if v["n_seeds"] > 1}
    if not multi:
        print("  fig_seed_distribution: need ≥2 seeds per method — skipping.")
        return

    rng = np.random.default_rng(0)
    names = list(multi.keys())
    colors = [multi[n]["cfg"]["color"] for n in names]
    data = [[float(np.max(c)) for c in multi[n]["curves"]] for n in names]
    means = [float(np.mean(d)) for d in data]
    stds = [float(np.std(d, ddof=1)) if len(d) > 1 else 0.0 for d in data]

    all_vals = [v for d in data for v in d]
    ylo = min(all_vals) - 0.8
    yhi = max(all_vals) + 3.0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle(
        f"Seed Consistency — Best Val Accuracy  ·  5 Seeds  ·  CIFAR-100  ·  {MODEL_DISPLAY}",
        fontsize=10,
        fontweight="bold",
    )

    x_pos = np.arange(len(names))
    bar_w = 0.55
    static_mu = means[0]

    for xi, (d, mu, std, color) in enumerate(zip(data, means, stds, colors)):
        # filled bar from ylo to mean (low alpha for background effect)
        ax.bar(
            xi,
            mu - ylo,
            bottom=ylo,
            width=bar_w,
            color=color,
            alpha=0.20,
            edgecolor=color,
            linewidth=1.5,
            zorder=1,
        )

        # bold mean line spanning the bar width
        ax.hlines(
            mu,
            xi - bar_w / 2 + 0.02,
            xi + bar_w / 2 - 0.02,
            color=color,
            linewidth=3.0,
            zorder=4,
        )

        # thin ±1σ line, no caps
        ax.vlines(xi, mu - std, mu + std, color="black", linewidth=1.8, zorder=3)

        # individual seed dots, jittered
        jitter = rng.uniform(-0.13, 0.13, len(d))
        ax.scatter(
            x_pos[xi] + jitter,
            d,
            color=color,
            s=90,
            zorder=5,
            edgecolors="white",
            linewidths=1.2,
        )

        # label above the ±1σ top
        delta_str = f"\nvs Static +{mu - static_mu:.2f} pp" if xi > 0 else ""
        ax.text(
            xi,
            mu + std + 0.45,
            f"{mu:.2f}% ± {std:.2f}{delta_str}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )

    short = [n.replace(" (ours)", "").replace(" v2", "") for n in names]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short, fontsize=12)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_ylabel("Best Validation Accuracy (%)", fontsize=10)
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.text(
        0.99,
        0.02,
        "● individual seed   — mean   | ±1σ",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#777777",
        style="italic",
    )

    plt.tight_layout()
    save_fig(fig, fname)
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    global MODEL_DISPLAY

    parser = argparse.ArgumentParser(description="Multi-seed band plot")
    parser.add_argument(
        "--model",
        choices=list(MODEL_DISPLAY_MAP),
        default="wideresnet",
        help="Which model's runs to plot (selects results/cluster/logs/<model> as the log fallback)",
    )
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument(
        "--log_dir", default=None, help="Override the log fallback directory"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Gaussian smoothing window in epochs (default 5)",
    )
    args = parser.parse_args()

    if not HAVE_TORCH:
        print("ERROR: torch not found — activate your venv first.")
        sys.exit(1)

    MODEL_DISPLAY = MODEL_DISPLAY_MAP[args.model]
    model_token = MODEL_TOKEN[args.model]
    log_dir = args.log_dir or str(LOG_DIR_ROOT / MODEL_LOG_SUBDIR[args.model])

    print(f"\n  model          : {MODEL_DISPLAY}")
    print(f"  checkpoint_dir : {args.checkpoint_dir}")
    print(f"  log_dir        : {log_dir}")
    print(f"  seeds          : {args.seeds}")
    print(f"  smooth window  : {args.smooth}")
    print(f"  figures → {FIGURES_DIR}\n")

    md = {}
    for method_name, cfg in METHODS.items():
        print(f"  ── {method_name}")
        histories, found, test_top1s = gather_seeds(
            method_name, cfg, args.seeds, args.checkpoint_dir, log_dir, model_token
        )
        if not histories:
            print()
            continue

        epochs, mean, std, curves = compute_bands(histories, "val_acc", args.smooth)

        md[method_name] = {
            "epochs": epochs,
            "mean": mean,
            "std": std,
            "curves": curves,
            "seeds": found,
            "n_seeds": len(histories),
            "test_top1_by_seed": {
                s: t for s, t in zip(found, test_top1s) if t is not None
            },
            "cfg": cfg,
        }
        print()

    if not md:
        print("  No history files or logs found locally for this model.")
        print("  Pull checkpoint histories from the cluster with:")
        print(
            "    rsync -av user@cluster:/path/checkpoints/*_p19*_history.pt checkpoints/"
        )
        print(f"  ...or make sure {log_dir} contains the raw .log files.")
        sys.exit(0)

    print(f"  Methods loaded: {list(md.keys())}\n")
    stats = compute_stats_vs_static(md)
    print("  Generating figures...\n")

    fig_seed_bands(md, stats, fname=f"fig_seed_bands_{args.model}.png")
    fig_tier_zoom(md, fname=f"fig_tier_zoom_{args.model}.png")
    fig_seed_distribution(md, fname=f"fig_seed_distribution_{args.model}.png")

    print(f"\n  Done. Figures saved to: {os.path.abspath(FIGURES_DIR)}\n")


if __name__ == "__main__":
    main()
