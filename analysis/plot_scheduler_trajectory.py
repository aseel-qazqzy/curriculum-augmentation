"""analysis/plot_scheduler_trajectory.py

Scheduler pacing/trajectory plot for the three dynamic schedulers (ETS, LPS,
EGS) — Static+Mixing is intentionally excluded since it has no tier schedule
(it samples from the full Tier-3 op pool from epoch 1 onward, i.e. it is not
"dynamic" in the sense this figure is about).

What is plotted: an "effective tier" curve on a 1-3 scale, per epoch, mean +/-
std over the 5 seeds, for each model. This unifies three structurally
different logging schemes onto one comparable axis:

  - ETS  : tier(epoch) is a DETERMINISTIC step function, reconstructed
           EXACTLY at every epoch from the fixed boundaries printed once in
           each log's header ("Tier 1 (ep   1-20)", "Tier 2 (ep 21-45)" ->
           t1_end=20, t2_end=45; identical across all 10 ETS logs here).
  - LPS  : tier(epoch) is a DATA-DRIVEN step function, reconstructed EXACTLY
           at every epoch from the "LPS Tier transitions: [(epoch, from, to),
           ...]" list each log prints once at the end (transition epochs
           differ per seed, which is exactly the point of LPS).
  - EGS  : there is no single scalar "tier" (each of the 45,000 training
           samples has its own tier). Instead, every logged epoch line
           reports population counts "EGS T1:a T2:b T3:c". The population-
           level effective tier = (1*a + 2*b + 3*c) / (a+b+c) is plotted as
           the closest analogue to ETS/LPS's tier(epoch) — a continuous
           curriculum-progress signal. Because these EGS logs are only
           written every 10 epochs, this curve is only defined at those ~11
           sampled points per seed (unlike ETS/LPS, which are reconstructed
           analytically at all 100 epochs) — plotted with markers to make
           that resolution difference visible rather than implying a false
           precision.

Usage:
    python analysis/plot_scheduler_trajectory.py
    python analysis/plot_scheduler_trajectory.py --band ci95
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from plot_thesis_convergence import (
    LOG_ROOT,
    MODEL_DIRS,
    OUT_DIR,
    _tex,
    configure_style,
    parse_method,
    parse_seed,
)

METHOD_ORDER = ["ETS", "LPS", "EGS"]
METHOD_STYLE = {
    "ETS": {"color": "#0072B2", "ls": "--", "marker": "s"},
    "LPS": {"color": "#009E73", "ls": "-.", "marker": "^"},
    "EGS": {"color": "#D55E00", "ls": (0, (1, 1)), "marker": "D"},
}

_TOTAL_EPOCHS_RE = re.compile(r"Epochs\s*:\s*(\d+)")
_ETS_T1_RE = re.compile(r"Tier 1 \(ep\s*\d+-\s*(\d+)\)")
_ETS_T2_RE = re.compile(r"Tier 2 \(ep\s*\d+-\s*(\d+)\)")
_LPS_TRANSITIONS_RE = re.compile(r"LPS Tier transitions:\s*(\[.*\])")
_EGS_EPOCH_RE = re.compile(
    r"Epoch\s*\[\s*(\d+)\s*/\s*\d+\].*?EGS T1:(\d+) T2:(\d+) T3:(\d+)"
)


def _load_total_epochs(text: str) -> int:
    m = _TOTAL_EPOCHS_RE.search(text)
    if not m:
        raise ValueError("Could not find 'Epochs      : N' header line")
    return int(m.group(1))


def _trajectory_ets(text: str, total_epochs: int) -> pd.DataFrame:
    t1_m, t2_m = _ETS_T1_RE.search(text), _ETS_T2_RE.search(text)
    if not (t1_m and t2_m):
        raise ValueError(
            "Could not find ETS 'Tier 1 (ep a-b)'/'Tier 2 (ep a-b)' header lines"
        )
    t1_end, t2_end = int(t1_m.group(1)), int(t2_m.group(1))
    epochs = np.arange(1, total_epochs + 1)
    tier = np.where(epochs <= t1_end, 1, np.where(epochs <= t2_end, 2, 3))
    return pd.DataFrame({"epoch": epochs, "effective_tier": tier})


def _trajectory_lps(text: str, total_epochs: int) -> pd.DataFrame:
    m = _LPS_TRANSITIONS_RE.search(text)
    if not m:
        raise ValueError("Could not find 'LPS Tier transitions: [...]' line")
    transitions = ast.literal_eval(m.group(1))  # list of (epoch, from_tier, to_tier)
    transitions = sorted(transitions, key=lambda t: t[0])
    epochs = np.arange(1, total_epochs + 1)
    tier = np.ones_like(epochs)
    current = 1
    t_idx = 0
    for i, ep in enumerate(epochs):
        while t_idx < len(transitions) and ep >= transitions[t_idx][0]:
            current = transitions[t_idx][2]
            t_idx += 1
        tier[i] = current
    return pd.DataFrame({"epoch": epochs, "effective_tier": tier})


def _trajectory_egs(text: str) -> pd.DataFrame:
    rows = []
    for m in _EGS_EPOCH_RE.finditer(text):
        epoch, t1, t2, t3 = (int(m.group(i)) for i in range(1, 5))
        total = t1 + t2 + t3
        if total == 0:
            continue
        eff = (1 * t1 + 2 * t2 + 3 * t3) / total
        rows.append({"epoch": epoch, "effective_tier": eff})
    if not rows:
        raise ValueError("Could not find any 'EGS T1:.. T2:.. T3:..' epoch lines")
    return pd.DataFrame(rows)


def load_trajectory(file_path: Path) -> pd.DataFrame:
    model_dir = file_path.parent.name
    model_label = MODEL_DIRS.get(model_dir, model_dir)
    method = parse_method(file_path.name)
    seed = parse_seed(file_path.name)
    if method not in METHOD_ORDER or seed is None:
        raise ValueError(
            f"Not a dynamic-scheduler log (or unparsable): {file_path.name}"
        )

    text = file_path.read_text()
    total_epochs = _load_total_epochs(text)

    if method == "ETS":
        traj = _trajectory_ets(text, total_epochs)
    elif method == "LPS":
        traj = _trajectory_lps(text, total_epochs)
    else:  # EGS
        traj = _trajectory_egs(text)

    traj["model"] = model_label
    traj["method"] = method
    traj["seed"] = seed
    return traj


def discover_trajectories(log_root: Path) -> pd.DataFrame:
    frames, skipped = [], []
    for model_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        for log_file in sorted(model_dir.glob("*.log")):
            method = parse_method(log_file.name)
            if method not in METHOD_ORDER:
                continue  # Static+Mixing (or anything unrecognized) — not a scheduler trajectory
            try:
                frames.append(load_trajectory(log_file))
            except ValueError as e:
                skipped.append(str(e))
    if skipped:
        print(f"WARNING: skipped {len(skipped)} file(s):")
        for s in skipped:
            print(f"  - {s}")
    if not frames:
        raise RuntimeError(
            f"No parsable scheduler-trajectory logs found under {log_root}"
        )
    df = pd.concat(frames, ignore_index=True)
    n_runs = df.drop_duplicates(subset=["model", "method", "seed"]).shape[0]
    print(f"Parsed trajectories from {n_runs} unique (model, method, seed) runs.")
    return df


def aggregate_trajectory(df: pd.DataFrame, band: str) -> pd.DataFrame:
    g = df.groupby(["model", "method", "epoch"])["effective_tier"]
    agg = g.agg(mean="mean", std="std", n="count").reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    agg["sem"] = agg["std"] / np.sqrt(agg["n"])
    if band == "ci95":
        agg["band"] = agg.apply(
            lambda r: r["sem"] * stats.t.ppf(0.975, r["n"] - 1) if r["n"] > 1 else 0.0,
            axis=1,
        )
    else:
        agg["band"] = agg["std"]
    return agg


def plot_scheduler_trajectory(
    agg: pd.DataFrame, use_tex: bool, name: str = "scheduler_trajectory"
) -> None:
    model_labels = list(MODEL_DIRS.values())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    for ax, model_label in zip(axes, model_labels):
        sub_model = agg[agg["model"] == model_label]
        for method in METHOD_ORDER:
            sub = sub_model[sub_model["method"] == method].sort_values("epoch")
            if sub.empty:
                continue
            style = METHOD_STYLE[method]
            # EGS is only sampled every ~10 epochs (see module docstring) — show
            # markers without connecting every point as a smooth line to avoid
            # implying resolution that isn't there; ETS/LPS are exact at every
            # epoch, so draw as continuous step-like lines with sparse markers.
            markevery = 1 if method == "EGS" else 10
            ax.plot(
                sub["epoch"],
                sub["mean"],
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                markersize=5,
                markevery=markevery,
                label=method,
                zorder=3,
            )
            ax.fill_between(
                sub["epoch"],
                sub["mean"] - sub["band"],
                sub["mean"] + sub["band"],
                color=style["color"],
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
        ax.set_title(model_label)
        ax.set_xlabel(_tex("Epoch", use_tex))
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["Tier 1", "Tier 2", "Tier 3"])
        ax.margins(x=0.02)

    axes[0].set_ylabel(_tex("Effective tier (curriculum pacing)", use_tex))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHOD_ORDER),
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  Saved -> {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scheduler pacing/trajectory plot for ETS/LPS/EGS"
    )
    parser.add_argument("--log_root", default=str(LOG_ROOT))
    parser.add_argument("--band", choices=["sd", "ci95"], default="sd")
    args = parser.parse_args()

    df = discover_trajectories(Path(args.log_root))
    use_tex = configure_style()

    agg = aggregate_trajectory(df, band=args.band)
    plot_scheduler_trajectory(agg, use_tex=use_tex)

    print("\nDone.")


if __name__ == "__main__":
    main()
