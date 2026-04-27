"""
experiments/tune_lps.py
Grid-search tau x window x min_epochs for the val-loss LossPlateauScheduler.

Replays the scheduler on a saved val_loss history (no GPU needed), scores
every combination, and prints a ranked top-N list.

Scoring (higher = better):
  +2  if Tier 3 is reached at all
  +1  for every epoch Tier 3 starts before ETS (beats ETS)
  -1  for every epoch Tier 3 starts after  ETS (lags  ETS)
  +0.1 * (Tier-3 epochs)  — reward longer Tier-3 exposure
  -5  if T1->T2 happens before min_t1t2 (too aggressive in Tier 1)

Usage:
    python experiments/tune_lps.py
    python experiments/tune_lps.py --checkpoint checkpoints/my_run_best.pth
    python experiments/tune_lps.py --top 10
    python experiments/tune_lps.py --taus 0.01 0.02 0.05 --windows 3 5 7 --min_epochs_list 5 10 15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def simulate_lps(val_losses, tau, window, min_epochs, milestones):
    """
    Replay LossPlateauScheduler on a val_loss sequence.
    Returns list of (epoch, old_tier, new_tier).
    """
    history = []
    tier = 1
    epochs_in_tier = 0
    lr_drop_grace = 0
    tier_changes = []

    for epoch, vl in enumerate(val_losses, start=1):
        history.append(vl)
        epochs_in_tier += 1
        if lr_drop_grace > 0:
            lr_drop_grace -= 1

        if epoch in milestones:
            lr_drop_grace = window  # grace period — history preserved

        if tier == 3:
            continue
        if epochs_in_tier < min_epochs:
            continue
        if lr_drop_grace > 0:
            continue
        if len(history) < window * 2:
            continue

        recent_avg = sum(history[-window:]) / window
        previous_avg = sum(history[-window * 2 : -window]) / window
        denom = max(abs(previous_avg), 1e-8)
        improvement = (previous_avg - recent_avg) / denom  # val_loss: down = better

        if improvement < tau:
            tier_changes.append((epoch, tier, tier + 1))
            tier += 1
            epochs_in_tier = 0
            history = []
            lr_drop_grace = 0

    return tier_changes


def score(t1t2, t2t3, t3_epochs, n_epochs, ets_t3, min_t1t2):
    """
    Composite score for a (tau, window, min_epochs) combination.

    Penalises:
      - never reaching Tier 3            (-999 sentinel)
      - T1->T2 too early (< min_t1t2)    (-5)
      - Tier 3 starting late vs ETS      (-1 per epoch lag)

    Rewards:
      - reaching Tier 3 at all           (+2)
      - beating ETS                      (+1 per epoch lead)
      - more Tier-3 epochs               (+0.1 per epoch)
    """
    if t2t3 is None:
        return -999

    s = 2.0
    s += ets_t3 - t2t3  # positive if beats ETS, negative if lags
    s += 0.1 * t3_epochs

    if t1t2 is not None and t1t2 < min_t1t2:
        s -= 5.0  # penalise leaving Tier 1 before the model has stabilised

    return round(s, 2)


def main():
    parser = argparse.ArgumentParser(
        description="Grid-search tau x window x min_epochs for val-loss LPS"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to _best.pth or _history.pt (default: lps_nomix best.pth)",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[33, 66, 83],
        help="MultiStepLR milestone epochs for grace-period simulation",
    )
    parser.add_argument(
        "--ets_t3",
        type=int,
        default=46,
        help="ETS Tier-3 start epoch — used as reference (default: 46)",
    )
    parser.add_argument(
        "--min_t1t2",
        type=int,
        default=15,
        help="Earliest acceptable T1->T2 epoch; earlier is penalised (default: 15)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top-ranked combos to print (default: 10)",
    )
    parser.add_argument(
        "--taus", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.10]
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[3, 5, 7, 10])
    parser.add_argument(
        "--min_epochs_list",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="Values of min_epochs_per_tier to try",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    ckpt_path = args.checkpoint or str(
        root
        / "checkpoints"
        / "resnet50_tiered_lps_nomix_sgd_multistep_ep100_cifar100_best.pth"
    )

    print(f"Loading : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history", ckpt)
    val_losses = history["val_loss"]
    n = len(val_losses)
    print(
        f"Epochs  : {n}  |  milestones: {args.milestones}  |  ETS Tier-3 ref: ep {args.ets_t3}"
    )
    print(
        f"Search  : {len(args.taus)} taus x {len(args.windows)} windows x "
        f"{len(args.min_epochs_list)} min_epochs = "
        f"{len(args.taus) * len(args.windows) * len(args.min_epochs_list)} combinations\n"
    )

    results = []
    for tau in args.taus:
        for window in args.windows:
            for min_ep in args.min_epochs_list:
                changes = simulate_lps(
                    val_losses,
                    tau=tau,
                    window=window,
                    min_epochs=min_ep,
                    milestones=args.milestones,
                )
                t1t2 = next((ep for ep, o, _ in changes if o == 1), None)
                t2t3 = next((ep for ep, o, _ in changes if o == 2), None)
                t3_ep = n - t2t3 + 1 if t2t3 else 0
                s = score(t1t2, t2t3, t3_ep, n, args.ets_t3, args.min_t1t2)
                results.append((s, tau, window, min_ep, t1t2, t2t3, t3_ep))

    results.sort(reverse=True)

    # ── full grid table ──────────────────────────────────────────────────────
    col = 10
    hdr = (
        f"{'tau':>6}  {'win':>4}  {'min_ep':>6}  "
        f"{'T1->T2':>{col}}  {'T2->T3':>{col}}  {'T3 eps':>{col}}  {'score':>7}  note"
    )
    print(hdr)
    print("─" * (len(hdr) + 4))

    for s, tau, window, min_ep, t1t2, t2t3, t3_ep in results:
        t1t2_s = f"ep {t1t2:>3}" if t1t2 else "never"
        t2t3_s = f"ep {t2t3:>3}" if t2t3 else "never"
        t3ep_s = f"{t3_ep} ep" if t2t3 else "—"
        if t2t3 is None:
            note = "STUCK"
        elif t2t3 < args.ets_t3:
            note = f"beats ETS +{args.ets_t3 - t2t3}"
        elif t2t3 == args.ets_t3:
            note = "ties ETS"
        else:
            note = f"lags ETS -{t2t3 - args.ets_t3}"
        print(
            f"{tau:>6.3f}  {window:>4}  {min_ep:>6}  "
            f"{t1t2_s:>{col}}  {t2t3_s:>{col}}  {t3ep_s:>{col}}  {s:>7.2f}  {note}"
        )

    # ── top-N ranked summary ─────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Top {args.top} combinations by score")
    print(f"{'─' * 60}")
    print(
        f"  {'Rank':>4}  {'tau':>6}  {'win':>4}  {'min_ep':>6}  "
        f"{'T2->T3':>8}  {'T3 eps':>7}  {'score':>7}"
    )
    for rank, (s, tau, window, min_ep, t1t2, t2t3, t3_ep) in enumerate(
        results[: args.top], 1
    ):
        t2t3_s = f"ep {t2t3}" if t2t3 else "never"
        t3ep_s = f"{t3_ep}" if t2t3 else "—"
        print(
            f"  {rank:>4}  {tau:>6.3f}  {window:>4}  {min_ep:>6}  "
            f"{t2t3_s:>8}  {t3ep_s:>7}  {s:>7.2f}"
        )

    if results[0][0] > -999:
        best = results[0]
        print(
            f"\n  Recommended: tau={best[1]}  window={best[2]}  "
            f"min_epochs_per_tier={best[3]}"
        )
        print(f"  Add to your run command:")
        print(
            f"    --lps_tau {best[1]} --lps_window {best[2]} --lps_min_epochs {best[3]}"
        )


if __name__ == "__main__":
    main()
