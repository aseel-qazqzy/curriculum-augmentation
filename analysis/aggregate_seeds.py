"""
analysis/aggregate_seeds.py
Aggregate multi-seed runs and produce the main thesis results table.

Detects all checkpoints matching *_s{seed}_best.pth, groups them by their
base name (everything before _s{seed}), then computes mean ± std over seeds.
Also runs a paired t-test (ETS vs Static) for statistical significance.

Usage:
    python analysis/aggregate_seeds.py                     # all methods
    python analysis/aggregate_seeds.py --model wideresnet  # filter by model
    python analysis/aggregate_seeds.py --csv               # also write CSV
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    print("torch not found — cannot load checkpoints", file=sys.stderr)
    sys.exit(1)

try:
    from scipy import stats as _stats

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

# seed suffix pattern: _s42, _s123, _s456, …
_SEED_RE = re.compile(r"_s(\d+)_best\.pth$")


def _discover(checkpoint_dir: Path, model_filter: str = None) -> dict:
    """Return {base_name: [(seed, test_top1, val_acc, best_epoch), ...]}"""
    groups = defaultdict(list)

    for p in sorted(checkpoint_dir.glob("*_s*_best.pth")):
        m = _SEED_RE.search(p.name)
        if not m:
            continue
        seed = int(m.group(1))
        base = p.name[: m.start()]  # strip _s{seed}_best.pth

        if model_filter and not base.startswith(model_filter):
            continue

        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  WARNING: could not load {p.name}: {e}")
            continue

        test_top1 = ckpt.get("test_top1")
        val_acc = ckpt.get("val_acc", 0.0)
        epoch = ckpt.get("epoch", 0)

        if test_top1 is None:
            print(f"  WARNING: {p.name} has no test_top1 — run not fully complete")
            continue

        groups[base].append(
            (seed, float(test_top1) * 100, float(val_acc) * 100, int(epoch))
        )

    return dict(groups)


def _stats_row(values: list[float]) -> tuple:
    """Return (mean, std, n) — std=0 when n=1."""
    arr = np.array(values)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, len(arr)


def _fmt(mean, std, n) -> str:
    if n == 1:
        return f"{mean:.2f}%       "
    return f"{mean:.2f}% ± {std:.2f}%"


def print_table(groups: dict) -> list[dict]:
    rows = []
    print(f"\n{'═' * 90}")
    print("  RESULTS TABLE — Multi-Seed Aggregation")
    print(
        f"  {'Method':<52} {'Seeds':>5}  {'Test Top-1':>18}  {'Best Val':>12}  {'BestEp':>6}"
    )
    print(f"  {'─' * 52}  {'─' * 5}  {'─' * 18}  {'─' * 12}  {'─' * 6}")

    for base in sorted(groups):
        entries = sorted(groups[base], key=lambda x: x[0])
        test_vals = [e[1] for e in entries]
        val_vals = [e[2] for e in entries]
        seeds = [e[0] for e in entries]

        t_mean, t_std, n = _stats_row(test_vals)
        v_mean, v_std, _ = _stats_row(val_vals)

        seed_str = "+".join(str(s) for s in seeds)
        print(
            f"  {base:<52}  s={seed_str:<4}  {_fmt(t_mean, t_std, n):<18}  "
            f"{_fmt(v_mean, v_std, n):<12}"
        )

        rows.append(
            {
                "base": base,
                "seeds": seeds,
                "n": n,
                "test_mean": t_mean,
                "test_std": t_std,
                "val_mean": v_mean,
                "val_std": v_std,
                "test_values": test_vals,
            }
        )

    print(f"{'═' * 90}\n")
    return rows


def paired_ttest(rows: list[dict], method_a: str, method_b: str) -> None:
    """Paired t-test between two methods identified by substring in base name."""
    a = next((r for r in rows if method_a in r["base"]), None)
    b = next((r for r in rows if method_b in r["base"]), None)

    if a is None or b is None:
        print(f"  t-test: could not find both '{method_a}' and '{method_b}' — skipping")
        return
    if a["n"] != b["n"]:
        print(f"  t-test: seed count mismatch ({a['n']} vs {b['n']}) — skipping")
        return
    if a["n"] < 2:
        print(f"  t-test: only {a['n']} seed — need ≥2 for significance test")
        return

    if not HAVE_SCIPY:
        print("  t-test: scipy not installed — pip install scipy")
        return

    t, p = _stats.ttest_rel(a["test_values"], b["test_values"])
    delta = a["test_mean"] - b["test_mean"]
    sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant (p≥0.05)"
    print(f"  Paired t-test  |  {method_a} vs {method_b}")
    print(f"    Δ mean = {delta:+.2f}pp   t = {t:.3f}   p = {p:.4f}   → {sig}")
    print()


def write_csv(rows: list[dict], path: Path) -> None:
    import csv

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "base",
                "n",
                "seeds",
                "test_mean",
                "test_std",
                "val_mean",
                "val_std",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "base": r["base"],
                    "n": r["n"],
                    "seeds": "+".join(str(s) for s in r["seeds"]),
                    "test_mean": f"{r['test_mean']:.4f}",
                    "test_std": f"{r['test_std']:.4f}",
                    "val_mean": f"{r['val_mean']:.4f}",
                    "val_std": f"{r['val_std']:.4f}",
                }
            )
    print(f"  CSV saved -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=str(CHECKPOINT_DIR))
    parser.add_argument(
        "--model",
        default=None,
        help="Filter by model prefix, e.g. wideresnet or resnet50",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write results to results/tables/aggregated.csv",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("METHOD_A", "METHOD_B"),
        default=["ets", "static"],
        help="Two method substrings for paired t-test (default: ets static)",
    )
    args = parser.parse_args()

    groups = _discover(Path(args.checkpoint_dir), model_filter=args.model)
    if not groups:
        print("No completed multi-seed checkpoints found.")
        print("Checkpoints must follow the naming: <base>_s<seed>_best.pth")
        print("Example: wideresnet_static_sgd_cosine_ep100_cifar100_s42_best.pth")
        sys.exit(0)

    rows = print_table(groups)

    print("  Statistical significance:")
    paired_ttest(rows, args.compare[0], args.compare[1])

    if args.csv:
        write_csv(rows, RESULTS_DIR / "aggregated.csv")


if __name__ == "__main__":
    main()
