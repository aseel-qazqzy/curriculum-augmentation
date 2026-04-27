"""analysis/extract_logs.py — extract per-epoch metrics from training log files.

Usage:
    python analysis/extract_logs.py --report_mode dev        # ablations / method comparison
    python analysis/extract_logs.py --report_mode final      # thesis Table 1 vs published papers
    python analysis/extract_logs.py --report_mode ablation   # compact ablation table
    python analysis/extract_logs.py --report_mode curves     # export per-epoch CSV for figures
    python analysis/extract_logs.py --pattern "resnet50_tiered*"  # filter by glob
"""

import re
import csv
import sys
import argparse
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "results" / "logs"
TABL_DIR = Path(__file__).parent.parent / "results" / "tables"

# ── Regex patterns ────────────────────────────────────────────────────────────

# Dev mode epoch line: includes Val: fields
EPOCH_DEV_RE = re.compile(
    r"Epoch \[\s*(\d+)/\d+\]"
    r"\s+Train:\s*([\d.]+)\s*/\s*([\d.]+)%"
    r"\s+\|\s+Val:\s*([\d.]+)\s*/\s*([\d.]+)%"
    r"\s+\|\s+Top-5:\s*([\d.]+)%"
    r"\s+\|\s+LR:\s*([\S]+)"
)

# Full-train mode epoch line: no Val: fields
EPOCH_FULL_RE = re.compile(
    r"Epoch \[\s*(\d+)/\d+\]"
    r"\s+Train:\s*([\d.]+)\s*/\s*([\d.]+)%"
    r"\s+\|\s+LR:\s*([\S]+)"
)

EXP_RE = re.compile(r"Experiment\s*:\s*(.+)")
MODE_RE = re.compile(r"Mode:\s*(full-train|dev)")
BEST_RE = re.compile(r"Best saved \(epoch=(\d+),\s*val_acc=([\d.]+)%\)")
TEST1_RE = re.compile(r"Test\s+Top-1\s*:\s*([\d.]+)%")
TEST5_RE = re.compile(r"Test\s+Top-5\s*:\s*([\d.]+)%")
GAP_RE = re.compile(r"Val.Test Gap\s*:\s*([\d.]+)%")


# ── Parser ────────────────────────────────────────────────────────────────────


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")

    exp_match = EXP_RE.search(text)
    exp_name = exp_match.group(1).strip() if exp_match else path.stem

    mode_match = MODE_RE.search(text)
    full_train = (
        mode_match and "full-train" in mode_match.group(1) if mode_match else False
    )

    # Parse epoch rows
    rows = []
    for m in EPOCH_DEV_RE.finditer(text):
        rows.append(
            {
                "experiment": exp_name,
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "train_acc": float(m.group(3)),
                "val_loss": float(m.group(4)),
                "val_acc": float(m.group(5)),
                "val_top5": float(m.group(6)),
                "lr": m.group(7),
            }
        )

    if not rows:  # full-train mode — no val columns
        for m in EPOCH_FULL_RE.finditer(text):
            rows.append(
                {
                    "experiment": exp_name,
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "train_acc": float(m.group(3)),
                    "val_loss": None,
                    "val_acc": None,
                    "val_top5": None,
                    "lr": m.group(4),
                }
            )

    best_matches = BEST_RE.findall(text)
    best_val_acc = max((float(v) for _, v in best_matches), default=None)
    best_epoch = int(best_matches[-1][0]) if best_matches else None

    last = rows[-1] if rows else {}
    m1 = TEST1_RE.search(text)
    m5 = TEST5_RE.search(text)
    mg = GAP_RE.search(text)

    return {
        "exp_name": exp_name,
        "log_file": path.name,
        "full_train_mode": full_train,
        "epochs": rows,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "last_train_loss": last.get("train_loss"),
        "last_train_acc": last.get("train_acc"),
        "last_val_loss": last.get("val_loss"),
        "last_val_acc": last.get("val_acc"),
        "last_val_top5": last.get("val_top5"),
        "test_top1": float(m1.group(1)) if m1 else None,
        "test_top5": float(m5.group(1)) if m5 else None,
        "val_test_gap": float(mg.group(1)) if mg else None,
    }


# ── Formatters ────────────────────────────────────────────────────────────────


def _fmt(val, decimals=2) -> str:
    return f"{val:.{decimals}f}" if val is not None else "—"


# ── Report modes ──────────────────────────────────────────────────────────────


def report_dev(runs: list[dict]) -> None:
    """Development mode — Best Val Top-1, Best Epoch, Val-Test Gap.
    Use for: method comparison and ablation tables."""
    col = [42, 10, 10, 10, 10, 8]
    hdr = ["Experiment", "BestVal%", "BestEpoch", "Val-Test%", "TestTop1%", "Epochs"]
    sep = "  ".join("-" * c for c in col)
    fmt = "  ".join(f"{{:<{c}}}" for c in col)

    print(f"\n{'=' * sum(col)}")
    print("DEVELOPMENT REPORT  (val_split=0.1 | use for ablations & method comparison)")
    print(fmt.format(*hdr))
    print(sep)
    for r in sorted(runs, key=lambda x: x["best_val_acc"] or 0, reverse=True):
        if r["full_train_mode"]:
            continue
        print(
            fmt.format(
                r["exp_name"][: col[0]],
                _fmt(r["best_val_acc"]),
                str(r["best_epoch"]) if r["best_epoch"] else "—",
                _fmt(r["val_test_gap"]),
                _fmt(r["test_top1"]),
                str(len(r["epochs"])),
            )
        )
    print("=" * sum(col) + "\n")

    out = TABL_DIR / "dev_results.csv"
    fields = [
        "exp_name",
        "best_val_acc",
        "best_epoch",
        "val_test_gap",
        "test_top1",
        "test_top5",
        "last_train_loss",
        "last_train_acc",
    ]
    _write_summary_csv([r for r in runs if not r["full_train_mode"]], fields, out)


def report_final(runs: list[dict]) -> None:
    """Full-train mode — Test Top-1, Test Top-5.
    Use for: main thesis comparison table vs published papers."""
    col = [42, 11, 11, 12, 12]
    hdr = ["Experiment", "TestTop1%", "TestTop5%", "TrainLoss", "TrainAcc%"]
    sep = "  ".join("-" * c for c in col)
    fmt = "  ".join(f"{{:<{c}}}" for c in col)

    print(f"\n{'=' * sum(col)}")
    print("FINAL REPORT  (val_split=0.0 | use for thesis Table 1 vs published papers)")
    print(fmt.format(*hdr))
    print(sep)
    for r in sorted(runs, key=lambda x: x["test_top1"] or 0, reverse=True):
        if not r["full_train_mode"]:
            continue
        print(
            fmt.format(
                r["exp_name"][: col[0]],
                _fmt(r["test_top1"]),
                _fmt(r["test_top5"]),
                _fmt(r["last_train_loss"], 4),
                _fmt(r["last_train_acc"]),
            )
        )
    print("=" * sum(col) + "\n")

    out = TABL_DIR / "final_results.csv"
    fields = ["exp_name", "test_top1", "test_top5", "last_train_loss", "last_train_acc"]
    _write_summary_csv([r for r in runs if r["full_train_mode"]], fields, out)


def report_ablation(runs: list[dict]) -> None:
    """Compact ablation table sorted by Best Val Top-1.
    Use for: thesis Table 2 (ablation study)."""
    col = [42, 10, 10, 10]
    hdr = ["Experiment", "BestVal%", "BestEpoch", "Val-Test%"]
    sep = "  ".join("-" * c for c in col)
    fmt = "  ".join(f"{{:<{c}}}" for c in col)

    print(f"\n{'=' * sum(col)}")
    print("ABLATION REPORT  (val_split=0.1 | sorted by Best Val Top-1)")
    print(fmt.format(*hdr))
    print(sep)
    for r in sorted(runs, key=lambda x: x["best_val_acc"] or 0, reverse=True):
        if r["full_train_mode"]:
            continue
        print(
            fmt.format(
                r["exp_name"][: col[0]],
                _fmt(r["best_val_acc"]),
                str(r["best_epoch"]) if r["best_epoch"] else "—",
                _fmt(r["val_test_gap"]),
            )
        )
    print("=" * sum(col) + "\n")

    out = TABL_DIR / "ablation_results.csv"
    fields = ["exp_name", "best_val_acc", "best_epoch", "val_test_gap"]
    _write_summary_csv([r for r in runs if not r["full_train_mode"]], fields, out)


def report_curves(runs: list[dict]) -> None:
    """Export per-epoch CSV for learning curve figures."""
    out = TABL_DIR / "curves.csv"
    fields = [
        "experiment",
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_top5",
        "lr",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            writer.writerows(r["epochs"])
    print(
        f"Curves CSV saved → {out}  ({sum(len(r['epochs']) for r in runs)} epoch rows)"
    )


def _write_summary_csv(runs: list[dict], fields: list, out: Path) -> None:
    TABL_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    print(f"CSV saved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

REPORT_MODES = {
    "dev": report_dev,
    "final": report_final,
    "ablation": report_ablation,
    "curves": report_curves,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract metrics from training logs")
    parser.add_argument(
        "--report_mode",
        choices=list(REPORT_MODES),
        default="dev",
        help="dev=ablations/comparison | final=vs published papers | "
        "ablation=compact | curves=per-epoch CSV",
    )
    parser.add_argument(
        "--pattern", default="*.log", help="Glob pattern for log files (default: *.log)"
    )
    args = parser.parse_args()

    log_files = sorted(LOG_DIR.glob(args.pattern))
    if not log_files:
        print(f"No log files matching '{args.pattern}' in {LOG_DIR}", file=sys.stderr)
        sys.exit(1)

    runs = [parse_log(p) for p in log_files]
    print(f"Parsed {len(runs)} log file(s) from {LOG_DIR}\n")

    REPORT_MODES[args.report_mode](runs)


if __name__ == "__main__":
    main()
