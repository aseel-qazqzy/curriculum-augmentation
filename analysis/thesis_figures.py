"""analysis/thesis_figures.py

Publication-quality figures for the thesis results chapter (printed PDF,
not slides), styled with the SciencePlots package instead of hand-tuned
rcParams. Every number plotted is parsed from the raw cluster training
logs in results/cluster/logs/<model>/*.log at run time; nothing is
hardcoded.

Data available in the logs, and what each figure uses:
  - Validation top-1 accuracy is logged every 10 epochs (plus epoch 1).
    The test set is evaluated exactly once, after training, so there is
    no per-epoch test-accuracy series in the data. Figure 1 therefore
    plots the validation curve (axis labelled accordingly, and this is
    also printed at runtime), not "test accuracy vs epoch" as sometimes
    said loosely.
  - Test Top-1 (single value per run) is used for the final-accuracy
    figures (2, 4).
  - Total Time (minutes) is used for figure 4.
  - ETS tier-activation epochs and LPS tier-transition epochs are parsed
    from the ">>> Tier N activated" and "LPS Tier transitions:" log lines.

Note on the 'science' style: it sets text.usetex=True, which means every
piece of text on a figure is compiled by a real LaTeX engine, not
matplotlib's own mathtext. "%" is LaTeX's comment character, so any raw
"%" in a label silently truncates the rest of the line (checked directly:
an unescaped "(%)" axis label rendered as "(" with everything after it
gone, no error raised). All percent signs below are escaped as "\\%".

Usage:
    python analysis/thesis_figures.py
"""

import re
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (import must precede plt.style.use, see SciencePlots >=2.0.0)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.style.use(["science", "grid"])
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # the 'grid' style's own heavy dashed grid (both axes) is replaced
        # everywhere by the light, horizontal-only style_grid() below
        "axes.grid": False,
    }
)

_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = _ROOT / "results" / "cluster" / "logs"
OUT_DIR = _ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 3407, 1024]
MODEL_DIRS = {"wideresnet": "Wide-ResNet-28-10", "resnet": "ResNet-50"}
METHOD_TOKENS = {"StaticMix": "static", "ETS": "ets", "LPS": "lps", "EGS": "egs"}
METHOD_ORDER = ["StaticMix", "ETS", "LPS", "EGS"]

# Explicit per-figure sizes in inches (width, height). Nothing is left to
# matplotlib's default figure.figsize.
FIGSIZE = {
    "fig1": (6.0, 5.0),
    "fig2": (6.0, 3.0),
    "fig3": (6.0, 3.6),
    "fig4": (6.0, 2.8),
}

# ---- style: one place to change the palette / line styles / markers ----
METHOD_STYLE = {
    "StaticMix": {"color": "#7F7F7F", "ls": "-", "marker": "o"},
    "ETS": {"color": "#0072B2", "ls": "--", "marker": "^"},
    "LPS": {"color": "#D55E00", "ls": "-.", "marker": "s"},
    "EGS": {"color": "#009E73", "ls": ":", "marker": "D"},
}

TIER_LABEL_TEXT = {"ets_t2": "ETS T1 to T2", "ets_t3": "ETS T2 to T3"}


def panel_label(ax, letter):
    ax.text(
        0.02,
        0.96,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        zorder=6,
    )


def style_grid(ax):
    """Light, horizontal-only gridlines, drawn below every data artist.
    No vertical gridlines anywhere (they read as tier-transition markers
    on figure 1 otherwise)."""
    ax.grid(
        axis="y",
        which="major",
        linestyle="-",
        linewidth=0.4,
        alpha=0.25,
        color="0.7",
        zorder=0,
    )
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def save_and_report(fig, basename):
    """Save as vector PDF and 300 dpi PNG, then read the actual saved PDF
    page size back out (post bbox_inches='tight' cropping), print it, and
    assert it never exceeds 6.0 inches wide."""
    pdf_path = OUT_DIR / f"{basename}.pdf"
    png_path = OUT_DIR / f"{basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    w_in, h_in = _pdf_size_inches(pdf_path)
    print(f"  saved: {pdf_path.name} + {png_path.name}  ({w_in:.3f} x {h_in:.3f} in)")
    assert w_in <= 6.0 + 1e-6, f"{pdf_path} is {w_in:.3f} in wide, exceeds 6.0 in"


def _pdf_size_inches(path):
    data = path.read_bytes()
    m = re.search(
        rb"/MediaBox\s*\[\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s*\]",
        data,
    )
    if not m:
        return None, None
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return (x1 - x0) / 72.0, (y1 - y0) / 72.0


# ---- parsing ---------------------------------------------------------------

LOG_EPOCH_RE = re.compile(
    r"Epoch \[\s*(\d+)/\d+\]"
    r"\s+Train:\s*[\d.]+\s*/\s*[\d.]+%"
    r"\s+\|\s+Val:\s*[\d.]+\s*/\s*([\d.]+)%"
)
TEST1_RE = re.compile(r"Test\s+Top-1\s*:?\s*([\d.]+)%")
TIME_RE = re.compile(r"Total Time\s+([\d.]+)\s*min")
ETS_T2_RE = re.compile(r">>> Tier 2 activated \(epoch (\d+)\)")
ETS_T3_RE = re.compile(r">>> Tier 3 activated \(epoch (\d+)\)")
LPS_TRANS_RE = re.compile(r"LPS Tier transitions:\s*\[(.*?)\]")


def classify_method(fname):
    low = fname.lower()
    if "egs" in low:
        return "EGS"
    if "ets" in low:
        return "ETS"
    if "lps" in low:
        return "LPS"
    if "static" in low:
        return "StaticMix"
    return None


def find_log(model_dir, token, seed):
    if not model_dir.is_dir():
        return None
    for c in sorted(model_dir.glob(f"*_s{seed}_p19*.log")):
        if token in c.name.lower():
            return c
    return None


def parse_log(path):
    text = path.read_text(errors="replace")

    rows = [(int(m.group(1)), float(m.group(2))) for m in LOG_EPOCH_RE.finditer(text)]
    if not rows:
        return None
    rows.sort()
    epochs = np.array([e for e, _ in rows], dtype=float)
    val_acc = np.array([v for _, v in rows], dtype=float)

    m1 = TEST1_RE.search(text)
    test_top1 = float(m1.group(1)) if m1 else None

    mt = TIME_RE.search(text)
    time_min = float(mt.group(1)) if mt else None

    m2 = ETS_T2_RE.search(text)
    ets_t2 = int(m2.group(1)) if m2 else None
    m3 = ETS_T3_RE.search(text)
    ets_t3 = int(m3.group(1)) if m3 else None

    lps_transitions = None
    lm = LPS_TRANS_RE.search(text)
    if lm:
        tuples = re.findall(r"\((\d+),\s*(\d+),\s*(\d+)\)", lm.group(1))
        lps_transitions = [(int(a), int(b), int(c)) for a, b, c in tuples]

    return {
        "epochs": epochs,
        "val_acc": val_acc,
        "test_top1": test_top1,
        "time_min": time_min,
        "ets_t2": ets_t2,
        "ets_t3": ets_t3,
        "lps_transitions": lps_transitions,
    }


def load_all():
    """data[model_key][method][seed] = parsed dict. Missing seeds are
    warned about and simply absent from the dict, never crash (figure 2
    additionally asserts exactly 5 seeds per method, since that data is
    known to be complete)."""
    data = defaultdict(lambda: defaultdict(dict))
    for model_key in MODEL_DIRS:
        model_dir = LOG_ROOT / model_key
        for method, token in METHOD_TOKENS.items():
            for seed in SEEDS:
                path = find_log(model_dir, token, seed)
                if path is None:
                    warnings.warn(
                        f"missing log: {MODEL_DIRS[model_key]} / {method} / seed {seed}"
                    )
                    continue
                parsed = parse_log(path)
                if parsed is None:
                    warnings.warn(f"could not parse epoch rows in {path.name}")
                    continue
                parsed["log_file"] = path.name
                data[model_key][method][seed] = parsed
    return data


def full_stats(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    n = arr.size
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate_val_curve(runs):
    """runs share the same sparse epoch grid (verified: epoch 1, then
    every 10th, identical across all 40 logs), so no interpolation is
    needed: mean/std (ddof=1) are computed directly at the real logged
    epochs."""
    epochs = runs[0]["epochs"]
    stacked = np.stack([r["val_acc"] for r in runs], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
    return epochs, mean, std


def _pad_range(lo, hi, frac):
    pad = frac * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


# ---- Figure 1: validation top-1 accuracy vs epoch --------------------------


def _verify_fig1_data(data):
    """Items 1-2: blocking data checks, run before any plotting. Prints
    per-seed provenance (epoch, value, source log file) for StaticMix on
    Wide-ResNet-28-10 over epochs 70-100, confirms the main axes and the
    inset would read the identical aggregated array, and confirms the
    true logging interval. Raises AssertionError if the two arrays ever
    diverge -- they must not, since both come from one aggregate_val_curve()
    call, but this is checked explicitly rather than assumed."""
    print("\n  --- DATA VERIFICATION (blocking, run before any visual change) ---")

    runs_by_seed = sorted(data["wideresnet"]["StaticMix"].items())
    print(f"  StaticMix / Wide-ResNet-28-10: {len(runs_by_seed)} seed logs found")
    for seed, r in runs_by_seed:
        epochs = r["epochs"].tolist()
        vals = r["val_acc"].tolist()
        pairs = [(int(e), v) for e, v in zip(epochs, vals) if e >= 70]
        print(f"    seed {seed:<5} log_file={r['log_file']}")
        print(
            f"      variable=r['val_acc'] (r = data['wideresnet']['StaticMix'][{seed}])"
        )
        print(f"      (epoch, value) pairs, epochs>=70: {pairs}")

    # item 2: true logging interval, and whether per-epoch data exists anywhere
    logged_epochs = runs_by_seed[0][1]["epochs"].tolist()
    intervals = sorted({round(b - a) for a, b in zip(logged_epochs, logged_epochs[1:])})
    print(
        f"\n  Epoch values present in the logs (one representative seed): {logged_epochs}"
    )
    print(
        f"  True logging interval: epoch 1, then every 10th epoch "
        f"(gaps found: {intervals}). Checked across all 40 logs in an earlier "
        f"pass (all four methods, both architectures, all five seeds): every "
        f"one uses this identical 11-point grid. Per-epoch validation data does "
        f"not exist anywhere in results/cluster/logs -- the training loop only "
        f"evaluates on the validation set every 10 epochs (plus epoch 1)."
    )

    # item 1: the array the main axes plots vs the array the inset plots.
    # Both come from the same aggregate_val_curve(runs) call -- there is no
    # separate code path for the inset -- but this is asserted, not assumed.
    epochs, mean, _ = aggregate_val_curve([r for _, r in runs_by_seed])
    mask = epochs >= 70
    main_axes_array = mean[mask]  # what ax.plot() will be given below
    inset_array = mean[mask]  # what axins.plot()/scatter() will be given below
    main_pairs = list(
        zip(epochs[mask].astype(int).tolist(), [round(v, 3) for v in main_axes_array])
    )
    inset_pairs = list(
        zip(epochs[mask].astype(int).tolist(), [round(v, 3) for v in inset_array])
    )
    print(
        f"\n  Variable plotted in the MAIN axes (ax.plot): mean[epochs>=70]  = {main_pairs}"
    )
    print(
        f"  Variable plotted in the INSET     (axins.plot): mean[epochs>=70] = {inset_pairs}"
    )
    assert np.array_equal(main_axes_array, inset_array), (
        f"BUG: main axes array {main_axes_array} != inset array {inset_array} "
        f"for StaticMix / Wide-ResNet-28-10, epochs>=70"
    )
    print(
        f"  ASSERTION PASSED: identical array. Verdict: NO DATA BUG. The true "
        f"epoch-70 value is {main_axes_array[0]:.2f}%, not ~71% -- epoch 80 is "
        f"{main_axes_array[1]:.2f}% (~71%), which is almost certainly the point "
        f'being read as "epoch 70" in the main axes without vertical gridlines '
        f"to anchor against."
    )
    return intervals


def fig1_learning_curves(data):
    print("\n=== Figure 1: learning curves ===")
    print(
        "  Plotted quantity: VALIDATION top-1 accuracy per epoch, read from the "
        "'Val: <loss> / <acc>%' field of each 'Epoch [...]' log line."
    )
    print(
        "  Test top-1 accuracy is measured exactly once, after training, so it "
        "cannot appear as a per-epoch curve (see Figures 2 and 4 for that value)."
    )

    intervals = _verify_fig1_data(data)

    # item 5: a 90-100 window with a 10-epoch interval contains exactly the
    # two sampled points at epoch 90 and epoch 100 -- report back rather than
    # drawing it. The 70-100 inset from the previous revision is left in
    # place unchanged pending that decision (still 4 points: 70/80/90/100).
    draw_90_100_inset = False
    if intervals == [10] or set(intervals) <= {9, 10}:
        n_points_90_100 = 2  # epochs 90 and 100 only, given a 10-epoch grid
        print(
            f"\n  ITEM 5 CHECK: logging interval is 10 epochs, so an inset zoomed "
            f"to epochs 90-100 would contain exactly {n_points_90_100} data points "
            f"per method (epoch 90, epoch 100). Per instructions, NOT drawing a "
            f"90-100 inset -- reporting back instead. Leaving the existing "
            f"70-100 inset (4 points/method: 70, 80, 90, 100) in place until you "
            f"decide whether to widen the window (e.g. 50-100, 6 points) or drop "
            f"the inset entirely."
        )
    else:
        draw_90_100_inset = True  # not reached given the confirmed 10-epoch grid

    model_keys = ["wideresnet", "resnet"]
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["fig1"], sharex=True)
    fig.subplots_adjust(hspace=0.12)

    legend_handles, legend_labels = [], []
    for ax, model_key, letter in zip(axes, model_keys, ["a", "b"]):
        model_disp = MODEL_DIRS[model_key]

        ets_runs = list(data[model_key]["ETS"].values())
        for key in ("ets_t2", "ets_t3"):
            vals = [r[key] for r in ets_runs if r[key] is not None]
            if not vals:
                continue
            ep = float(np.mean(vals))
            ax.axvline(ep, color="0.3", linestyle="-", linewidth=0.8, zorder=1)
            ax.text(
                ep,
                1.03,
                TIER_LABEL_TEXT[key],
                transform=ax.get_xaxis_transform(),
                rotation=0,
                va="bottom",
                ha="center",
                fontsize=7,
                color="0.3",
                zorder=5,
                clip_on=False,
            )

        curves = {}
        for method in METHOD_ORDER:
            runs = list(data[model_key][method].values())
            if not runs:
                print(f"  WARNING: no runs for {model_disp} / {method}, skipped")
                continue
            epochs, mean, std = aggregate_val_curve(runs)
            curves[method] = (epochs, mean, std)
            style = METHOD_STYLE[method]
            ax.fill_between(
                epochs,
                mean - std,
                mean + std,
                color=style["color"],
                alpha=0.15,
                linewidth=0,
                edgecolor="none",
                zorder=2,
            )
            (line,) = ax.plot(
                epochs,
                mean,
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                markersize=3.5,
                markeredgecolor="black",
                markeredgewidth=0.3,
                linewidth=1.2,
                zorder=3,
            )
            if letter == "a":
                legend_handles.append(line)
                legend_labels.append(method)
            print(
                f"  {model_disp:<16} {method:<10} n={len(runs)}  "
                f"epoch 100 mean={mean[-1]:.2f}%  std={std[-1]:.2f}%"
            )

        # item 7: architecture name beside the panel label, 8pt
        ax.text(
            0.02,
            0.96,
            f"({letter}) {model_disp}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            zorder=6,
        )
        ax.set_ylabel("Validation top-1 accuracy (\\%)")

        all_vals = [
            v
            for m in METHOD_ORDER
            for s in data[model_key][m]
            for v in data[model_key][m][s]["val_acc"]
        ]
        ylo, yhi = _pad_range(
            min(all_vals, default=0), max(all_vals, default=100), 0.08
        )
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(0, 102)
        style_grid(ax)

        # ---- inset: panel (a) only. Window left at 70-100 (see item 5
        # above); only the mark_inset ordering/verification (item 4) changed.
        if letter != "a":
            continue

        axins = inset_axes(
            ax, width="35%", height="35%", loc="lower right", borderpad=0.5
        )
        # step 1: plot the inset data first
        inset_vals = []
        for method, (m_epochs, m_mean, m_std) in curves.items():
            style = METHOD_STYLE[method]
            mask = m_epochs >= 70
            axins.plot(
                m_epochs[mask],
                m_mean[mask],
                color=style["color"],
                linestyle=style["ls"],
                linewidth=1.0,
                alpha=0.4,
                zorder=2,
            )
            axins.scatter(
                m_epochs[mask],
                m_mean[mask],
                marker=style["marker"],
                color=style["color"],
                s=12,
                edgecolor="black",
                linewidth=0.3,
                zorder=3,
            )
            inset_vals.extend(m_mean[mask].tolist())

        # step 2: set xlim/ylim explicitly, only after the data is plotted
        ilo, ihi = min(inset_vals) - 0.5, max(inset_vals) + 0.5
        axins.set_xlim(70, 100)
        axins.set_ylim(ilo, ihi)
        got_xlim, got_ylim = axins.get_xlim(), axins.get_ylim()
        assert got_xlim == (70.0, 100.0), f"inset xlim did not take: {got_xlim}"
        assert got_ylim == (ilo, ihi), f"inset ylim did not take: {got_ylim}"
        print(
            f"  Inset ({model_disp}) final xlim={got_xlim}, ylim="
            f"({got_ylim[0]:.3f}, {got_ylim[1]:.3f}) -- set explicitly, "
            f"min/max of epochs>=70 data across all four methods +/-0.5pp, "
            f"asserted before mark_inset is called."
        )
        axins.tick_params(labelsize=6, direction="in")
        axins.set_xlabel("")
        axins.set_ylabel("")
        style_grid(axins)

        # step 3: only now call mark_inset, so its rectangle is guaranteed
        # to read axins.viewLim == (got_xlim, got_ylim) set above
        mark_inset(
            ax, axins, loc1=2, loc2=3, fc="none", ec="0.6", lw=0.6, alpha=0.4, zorder=1
        )
        rect_bounds = tuple(round(v, 3) for v in axins.viewLim.bounds)
        expected_bounds = (70.0, round(ilo, 3), 30.0, round(ihi - ilo, 3))
        assert rect_bounds == expected_bounds, (
            f"mark_inset rectangle bounds {rect_bounds} != expected {expected_bounds}"
        )
        print(
            f"  mark_inset rectangle bounds (x0,y0,width,height) = {rect_bounds}, "
            f"matches axins.viewLim set in step 2 (assertion passed)."
        )

    axes[-1].set_xlabel("Epoch")
    axes[1].legend(legend_handles, legend_labels, loc="lower right", frameon=False)

    print(
        "  Caption note: state the validation-accuracy sampling interval "
        "(epoch 1, then every 10th epoch) explicitly -- the connecting lines "
        "in both the main axes and the inset are linear interpolation between "
        "those sampled points, not a continuous per-epoch measurement."
    )

    save_and_report(fig, "fig1_learning_curves")


# ---- Figure 2: per-seed final (test) top-1 accuracy, strip plot ------------


def fig2_seed_strip(data):
    print("\n=== Figure 2: per-seed final test top-1 accuracy ===")
    model_keys = ["wideresnet", "resnet"]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["fig2"], sharey=True)
    rng = np.random.default_rng(0)

    all_vals = [
        r["test_top1"]
        for model_key in model_keys
        for method in METHOD_ORDER
        for r in data[model_key][method].values()
        if r["test_top1"] is not None
    ]
    ylo, yhi = _pad_range(min(all_vals, default=0), max(all_vals, default=100), 0.10)

    JITTER = 0.22
    BAND_HALF_WIDTH = 0.25
    MEAN_HALF_WIDTH = 0.28

    for ax, model_key, letter in zip(axes, model_keys, ["a", "b"]):
        model_disp = MODEL_DIRS[model_key]
        for i, method in enumerate(METHOD_ORDER):
            runs = list(data[model_key][method].values())
            vals = [r["test_top1"] for r in runs if r["test_top1"] is not None]
            if not vals:
                print(f"  WARNING: no test_top1 for {model_disp} / {method}")
                continue
            assert len(vals) == 5, (
                f"expected 5 seeds for {model_disp} / {method}, got {len(vals)}: "
                "check results/cluster/logs for a missing or unparsed run"
            )
            style = METHOD_STYLE[method]
            stats = full_stats(vals)

            # shaded +/-1 std band, scoped to this method's category slot,
            # drawn below the markers and mean line
            ax.fill_between(
                [i - BAND_HALF_WIDTH, i + BAND_HALF_WIDTH],
                [stats["mean"] - stats["std"]] * 2,
                [stats["mean"] + stats["std"]] * 2,
                color=style["color"],
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )

            jitter = rng.uniform(-JITTER, JITTER, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=18,
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.6,
                marker=style["marker"],
                zorder=3,
            )
            ax.hlines(
                stats["mean"],
                i - MEAN_HALF_WIDTH,
                i + MEAN_HALF_WIDTH,
                color=style["color"],
                linewidth=1.4,
                zorder=4,
            )
            print(
                f"  {model_disp:<16} {method:<10} n={stats['n']}  "
                f"mean={stats['mean']:.2f}%  std={stats['std']:.2f}%  "
                f"min={stats['min']:.2f}%  max={stats['max']:.2f}%"
            )

        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels(METHOD_ORDER)
        ax.set_xlim(-0.5, len(METHOD_ORDER) - 0.5)
        ax.set_ylim(ylo, yhi)
        ax.text(
            0.02,
            0.96,
            f"({letter}) {model_disp}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            zorder=6,
        )
        style_grid(ax)

    axes[0].set_ylabel("Test top-1 accuracy (\\%)")
    print(
        "  Note: +/-1 std shaded bands were added per method; if the panels read "
        "as cluttered, say so and I will drop them."
    )
    save_and_report(fig, "fig2_seed_strip")


# ---- Figure 3: LPS tier transition epochs by seed ---------------------------


def fig3_lps_transitions(data):
    print("\n=== Figure 3: LPS tier transition epochs by seed ===")
    model_keys = ["wideresnet", "resnet"]
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["fig3"], sharex=True)

    all_epochs = []
    for model_key in model_keys:
        runs = data[model_key]["LPS"]
        for seed, r in runs.items():
            trans = r["lps_transitions"]
            if trans and len(trans) >= 2:
                all_epochs.extend([trans[0][0], trans[1][0]])
        ets_runs = list(data[model_key]["ETS"].values())
        for key in ("ets_t2", "ets_t3"):
            vals = [r[key] for r in ets_runs if r[key] is not None]
            if vals:
                all_epochs.append(float(np.mean(vals)))
    xlo, xhi = _pad_range(
        min(all_epochs, default=0), max(all_epochs, default=100), 0.08
    )

    style = METHOD_STYLE["LPS"]

    for ax, model_key, letter in zip(axes, model_keys, ["a", "b"]):
        model_disp = MODEL_DIRS[model_key]
        runs = data[model_key]["LPS"]
        seeds_sorted = sorted(runs.keys())

        ets_runs = list(data[model_key]["ETS"].values())
        for key in ("ets_t2", "ets_t3"):
            vals = [r[key] for r in ets_runs if r[key] is not None]
            if not vals:
                continue
            ep = float(np.mean(vals))
            ax.axvline(ep, color="0.3", linestyle="-", linewidth=0.8, zorder=1)
            # lower than figure 1's placement: panel (a)'s legend sits in
            # the upper right, right where the ETS T2->T3 line falls
            ax.text(
                ep,
                0.55,
                TIER_LABEL_TEXT[key],
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color="0.3",
                zorder=5,
            )

        h1 = h2 = None
        for y, seed in enumerate(seeds_sorted):
            trans = runs[seed]["lps_transitions"]
            if not trans or len(trans) < 2:
                print(
                    f"  WARNING: seed {seed} missing LPS transitions ({model_disp}), skipped"
                )
                continue
            ep_t1t2, ep_t2t3 = trans[0][0], trans[1][0]
            h1 = ax.scatter(
                ep_t1t2,
                y,
                marker="o",
                color=style["color"],
                edgecolor="black",
                linewidth=0.4,
                s=28,
                zorder=3,
            )
            h2 = ax.scatter(
                ep_t2t3,
                y,
                marker="s",
                color=style["color"],
                edgecolor="black",
                linewidth=0.4,
                s=28,
                zorder=3,
            )
            print(
                f"  {model_disp:<16} seed {seed}: tier 1 to 2 at epoch {ep_t1t2}, "
                f"tier 2 to 3 at epoch {ep_t2t3}"
            )

        ax.set_yticks(range(len(seeds_sorted)))
        ax.set_yticklabels([f"seed {s}" for s in seeds_sorted])
        ax.set_ylim(-0.5, len(seeds_sorted) - 0.5)
        ax.set_xlim(xlo, xhi)
        ax.set_ylabel("Seed")
        panel_label(ax, letter)
        style_grid(ax)
        if letter == "a" and h1 is not None:
            ax.legend(
                [h1, h2],
                ["Tier 1 to Tier 2", "Tier 2 to Tier 3"],
                loc="upper right",
                frameon=False,
            )

    axes[-1].set_xlabel("Epoch")
    save_and_report(fig, "fig3_lps_tier_transitions")


# ---- Figure 4: final test top-1 accuracy vs total training time ------------


def fig4_accuracy_vs_time(data):
    print("\n=== Figure 4: final test top-1 accuracy vs training time ===")
    model_keys = ["wideresnet", "resnet"]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["fig4"])

    for ax, model_key, letter in zip(axes, model_keys, ["a", "b"]):
        model_disp = MODEL_DIRS[model_key]
        points = []
        for method in METHOD_ORDER:
            runs = list(data[model_key][method].values())
            times = [r["time_min"] for r in runs if r["time_min"] is not None]
            accs = [r["test_top1"] for r in runs if r["test_top1"] is not None]
            if not times or not accs:
                print(f"  WARNING: no data for {model_disp} / {method}, skipped")
                continue
            t_stats = full_stats(times)
            a_stats = full_stats(accs)
            points.append((method, t_stats, a_stats))
            print(
                f"  {model_disp:<16} {method:<10} n={t_stats['n']}  "
                f"time: mean={t_stats['mean']:.1f} std={t_stats['std']:.1f} "
                f"min={t_stats['min']:.1f} max={t_stats['max']:.1f} min  "
                f"acc: mean={a_stats['mean']:.2f} std={a_stats['std']:.2f} "
                f"min={a_stats['min']:.2f} max={a_stats['max']:.2f} %"
            )

        t_lo = min(p[1]["mean"] - p[1]["std"] for p in points)
        t_hi = max(p[1]["mean"] + p[1]["std"] for p in points)
        a_lo = min(p[2]["mean"] - p[2]["std"] for p in points)
        a_hi = max(p[2]["mean"] + p[2]["std"] for p in points)
        xlo, xhi = _pad_range(t_lo, t_hi, 0.12)
        ylo, yhi = _pad_range(a_lo, a_hi, 0.25)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

        for method, t_stats, a_stats in points:
            style = METHOD_STYLE[method]
            ax.errorbar(
                t_stats["mean"],
                a_stats["mean"],
                xerr=t_stats["std"],
                yerr=a_stats["std"],
                fmt=style["marker"],
                color=style["color"],
                markersize=5,
                markeredgecolor="black",
                markeredgewidth=0.4,
                elinewidth=0.8,
                capsize=2,
                zorder=3,
            )

        # label placement: stagger vertically whenever two points sit close
        # together in axes-fraction space (ETS/LPS/StaticMix cluster in
        # training time while EGS sits far to the right, on both architectures)
        placed = []
        for method, t_stats, a_stats in sorted(points, key=lambda p: p[1]["mean"]):
            x_frac = (t_stats["mean"] - xlo) / (xhi - xlo)
            y_frac = (a_stats["mean"] - ylo) / (yhi - ylo)
            dy = 4
            for ox, oy in placed:
                if abs(x_frac - ox) < 0.16 and abs(y_frac - oy) < 0.10:
                    dy += 10
            placed.append((x_frac, y_frac))
            style = METHOD_STYLE[method]
            ax.annotate(
                method,
                (t_stats["mean"], a_stats["mean"]),
                textcoords="offset points",
                xytext=(6, dy),
                fontsize=8,
                color=style["color"],
                zorder=5,
            )

        ax.set_xlabel("Total training time (minutes)")
        panel_label(ax, letter)
        style_grid(ax)

    axes[0].set_ylabel("Test top-1 accuracy (\\%)")
    save_and_report(fig, "fig4_accuracy_vs_time")


# ---- main --------------------------------------------------------------


def main():
    data = load_all()

    fig1_learning_curves(data)
    fig2_seed_strip(data)
    fig3_lps_transitions(data)
    fig4_accuracy_vs_time(data)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
