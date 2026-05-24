"""
analysis/ttest_significance.py

Welch's t-test + Wilcoxon signed-rank test + Cohen's d for all pairwise
method comparisons. Uses 5-seed results from WideResNet-28-10 · CIFAR-100
· 19-op pool · 100 epochs.

Run:
    python analysis/ttest_significance.py
"""

import numpy as np
from scipy import stats

# ── Results ───────────────────────────────────────────────────────────────────
# Seeds: 42 / 123 / 456 / 3407 / 1024  (same 5 seeds for all methods → paired)
RESULTS = {
    "Static Mixing": [77.79, 76.82, 77.69, 78.17, 77.54],
    "ETS": [81.35, 81.25, 81.35, 81.79, 81.23],
    "LPS": [81.36, 81.43, 81.27, 81.34, 81.79],
    "EGS v2": [79.83, 80.39, 79.81, 79.21, 79.49],
}

# ── Pairs to compare ──────────────────────────────────────────────────────────
PAIRS = [
    ("ETS", "Static Mixing"),
    ("LPS", "Static Mixing"),
    ("EGS v2", "Static Mixing"),
    ("ETS", "EGS v2"),
    ("LPS", "EGS v2"),
    ("ETS", "LPS"),
]


def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else float("inf")


def significance_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    header = (
        "WideResNet-28-10 · CIFAR-100 · 19-op · 100ep · "
        "seeds 42/123/456/3407/1024 (n=5, paired)"
    )
    print(f"\n{header}\n")

    # ── Per-method summary ────────────────────────────────────────────────────
    print(f"{'Method':<16} {'Mean':>7} {'Std':>6}  Seeds")
    print("─" * 60)
    for name, vals in RESULTS.items():
        print(
            f"{name:<16} {np.mean(vals):>6.2f}%  ±{np.std(vals, ddof=1):.2f}%  {vals}"
        )

    # ── Welch's t-test (unpaired) ─────────────────────────────────────────────
    print(f"\n── Welch's t-test (unpaired, unequal variance) ──")
    print(
        f"{'Comparison':<28} {'Δ Acc':>7} {'t':>7} {'p':>8} {'Sig':>4} {'Cohen d':>8}"
    )
    print("─" * 72)
    for a_name, b_name in PAIRS:
        a = np.array(RESULTS[a_name])
        b = np.array(RESULTS[b_name])
        t, p = stats.ttest_ind(a, b, equal_var=False)
        delta = np.mean(a) - np.mean(b)
        d = cohens_d(a, b)
        stars = significance_stars(p)
        label = f"{a_name} vs {b_name}"
        print(
            f"{label:<28} {delta:>+6.2f}pp  {t:>6.3f}  {p:>8.4f}  {stars:>4}  {d:>7.2f}"
        )

    # ── Wilcoxon signed-rank test (paired) ────────────────────────────────────
    print(f"\n── Wilcoxon signed-rank test (paired, non-parametric) ──")
    print(f"  Note: n=5 → minimum achievable two-tailed p ≈ 0.0625")
    print(f"{'Comparison':<28} {'Δ Acc':>7} {'W':>7} {'p':>8} {'Sig':>4}")
    print("─" * 58)
    for a_name, b_name in PAIRS:
        a = np.array(RESULTS[a_name])
        b = np.array(RESULTS[b_name])
        delta = np.mean(a) - np.mean(b)
        label = f"{a_name} vs {b_name}"
        try:
            w, p = stats.wilcoxon(a, b, alternative="two-sided")
            stars = significance_stars(p)
            print(f"{label:<28} {delta:>+6.2f}pp  {w:>6.1f}  {p:>8.4f}  {stars:>4}")
        except ValueError as e:
            print(f"{label:<28} {delta:>+6.2f}pp  — skipped ({e})")

    print("\nSignificance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    print("Cohen's d:    small ≥0.2  medium ≥0.5  large ≥0.8")
    print("\nWilcoxon uses paired differences (same 5 seeds across all methods).")
    print("Welch's t-test is the primary test; Wilcoxon confirms non-parametrically.")


if __name__ == "__main__":
    main()
