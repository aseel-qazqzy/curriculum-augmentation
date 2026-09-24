"""
experiments/madaug/cluster/summarize_smoke.py

Prints the Test 2 report items from a smoke run's config + metrics JSON.

    python -m experiments.madaug.cluster.summarize_smoke --out_dir results/madaug_smoke \
        --name madaug_cifar100_wrn28-10_ep200_s42_smoke
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--nvidia_smi_csv", default=None)
    a = p.parse_args()
    out = Path(a.out_dir)
    cfg = json.loads((out / "configs" / f"{a.name}.json").read_text())
    m = json.loads((out / "metrics" / f"{a.name}.json").read_text())
    h, v = m["history"], cfg["versions"]

    print(
        f"1. GPU                 : {v.get('gpu')} ({v.get('gpu_total_mem_gib') and round(v['gpu_total_mem_gib'], 1)} GiB)"
    )
    print(
        f"2. PyTorch / CUDA      : {v['torch']} / CUDA {v.get('cuda')} / cuDNN {v.get('cudnn')}"
    )
    print(f"3. higher              : {v.get('higher')}")
    print(
        f"4. batch size          : {cfg['args']['batch_size']} (policy-val batch {cfg['madaug']['policy_val_batch']}), AMP {cfg['amp']}"
    )
    print(f"   split               : {cfg['split']}")
    print(
        "   epoch | lr       | split_rate | h_upd | grad-norm min/mean/max        | nonfinite | peak alloc/reserved GiB | time s | val acc | test acc"
    )
    for i, ep in enumerate(h["epoch"]):
        g = lambda k: h[k][i]
        gn = (
            "—"
            if g("h_grad_norm_mean") is None
            else f"{g('h_grad_norm_min'):.3g}/{g('h_grad_norm_mean'):.3g}/{g('h_grad_norm_max'):.3g}"
        )
        mem = (
            "—"
            if g("gpu_mem_peak_alloc_gib") is None
            else f"{g('gpu_mem_peak_alloc_gib'):.1f}/{g('gpu_mem_peak_reserved_gib'):.1f}"
        )
        print(
            f"   {ep:>5} | {g('lr'):.5f} | {g('split_rate'):.4f}     | {g('h_updates'):>5} | {gn:<29} | {g('nonfinite'):>9} | {mem:<23} | {g('epoch_time_s'):>6.0f} | {g('val_acc'):>6.2f}% | {g('test_acc'):>6.2f}%"
        )
    if a.nvidia_smi_csv and Path(a.nvidia_smi_csv).exists():
        rows = Path(a.nvidia_smi_csv).read_text().splitlines()[1:]
        used = [int(r.split(",")[1].strip().split()[0]) for r in rows if "MiB" in r]
        print(f"5. GPU memory (nvidia-smi peak): {max(used) if used else '—'} MiB")
    bi = [n for n in h["h_updates"][1:]]
    print(
        f"7. bi-level step       : {'completed' if bi and all(n > 0 for n in bi) else 'NOT OBSERVED'} — h updates/epoch {h['h_updates']}"
    )
    print(
        f"8. NaN/Inf             : {sum(h['nonfinite'])} non-finite losses/grad-norms"
    )
    ok = h["epoch"] == list(range(len(h["epoch"])))
    print(
        f"10. checkpoint/resume  : epochs recorded {h['epoch']} ({'continuous' if ok else 'GAP/DUPLICATE'})"
    )
    print(
        f"11. last epoch         : val {h['val_acc'][-1]:.2f}% / test {h['test_acc'][-1]:.2f}% (smoke run, logging only; not a result)"
    )


if __name__ == "__main__":
    main()
