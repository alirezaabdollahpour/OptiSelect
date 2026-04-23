#!/usr/bin/env python3
"""
Collect and display results from OptiSelect experiments.

Reads log files (not summary.json) because the two training code paths
write incompatible summary schemas: standard-mode writes empty history
lists, selection-mode writes flat final_* scalars and no entropy. The
logs carry the ground-truth eval metrics and sel_entropy for both modes.

Usage:
  python collect_results.py --dataset shakespeare
  python collect_results.py --dataset wikitext
  python collect_results.py --dataset owt2 --scale full
  python collect_results.py --dataset slimpajama --scale full --seed 0
"""
import argparse
import json
import os
import re


DATASET_DISPLAY = {
    "shakespeare": "Shakespeare-Char",
    "wikitext": "WikiText-103",
    "owt2": "OpenWebText2",
    "slimpajama": "SlimPajama",
}
SCALED_DATASETS = {"owt2", "slimpajama"}

OPTIMIZERS = ["adamw", "ademamix", "d-muon", "mars", "sophiag", "soap",
              "lion", "signum", "adopt", "sgd"]
DISPLAY = {
    "adamw": "AdamW", "ademamix": "AdEMAMix", "d-muon": "D-Muon",
    "mars": "MARS", "sophiag": "Sophia", "soap": "SOAP",
    "lion": "Lion", "signum": "Signum", "adopt": "ADOPT", "sgd": "SGD",
}
MODES = ["standard", "selection"]


def log_filename(dataset, scale, mode, opt, seed):
    if dataset in SCALED_DATASETS:
        return f"{mode}_{dataset}_{scale}_{opt}_seed{seed}.log"
    return f"{mode}_{dataset}_{opt}_seed{seed}.log"


def parse_log(path):
    r = {"val_loss": None, "val_pp": None, "val_acc": None, "entropy": None}
    if not os.path.exists(path):
        return r
    entropies = []
    eval_re = re.compile(
        r">Eval[^\n]*?val_loss=([0-9.]+)\s+val_pp=([0-9.]+)\s+val_acc=([0-9.]+)"
    )
    ent_re = re.compile(r"sel_entropy=([0-9.]+)")
    with open(path, errors="replace") as f:
        for line in f:
            # Last >Eval wins — final metrics
            m = eval_re.search(line)
            if m:
                r["val_loss"] = float(m.group(1))
                r["val_pp"] = float(m.group(2))
                r["val_acc"] = float(m.group(3))
            for e in ent_re.finditer(line):
                entropies.append(float(e.group(1)))
    if entropies:
        r["entropy"] = sum(entropies) / len(entropies)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_DISPLAY))
    ap.add_argument("--scale", default="full",
                    help="only used for owt2/slimpajama (small|full)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--logs_dir",
        default="/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src/logs",
    )
    args = ap.parse_args()

    results = {}
    n_found = 0
    for opt in OPTIMIZERS:
        results[opt] = {}
        for mode in MODES:
            path = os.path.join(
                args.logs_dir,
                log_filename(args.dataset, args.scale, mode, opt, args.seed),
            )
            r = parse_log(path)
            results[opt][mode] = r
            if r["val_loss"] is not None:
                n_found += 1

    name = DATASET_DISPLAY[args.dataset]
    scale_note = f" [{args.scale}]" if args.dataset in SCALED_DATASETS else ""
    total = len(OPTIMIZERS) * len(MODES)
    print()
    print("=" * 130)
    print(f"  {name}{scale_note} Results: Standard vs OptiSelect (seed {args.seed})")
    print(f"  Found {n_found}/{total} runs")
    print("=" * 130)

    hdr = (
        f"{'Optimizer':<12} | {'Std Loss':>10} {'Std PP':>10} {'Std Acc%':>10} | "
        f"{'Sel Loss':>10} {'Sel PP':>10} {'Sel Acc%':>10} | "
        f"{'ΔLoss':>8} {'ΔAcc':>8} {'H_sel':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for opt in OPTIMIZERS:
        s = results[opt]["standard"]
        x = results[opt]["selection"]
        disp = DISPLAY.get(opt, opt)
        sl = f"{s['val_loss']:.3f}" if s["val_loss"] is not None else "    —"
        sp = f"{s['val_pp']:.2f}" if s["val_pp"] is not None else "    —"
        sa = f"{100*s['val_acc']:.2f}" if s["val_acc"] is not None else "    —"
        xl = f"{x['val_loss']:.3f}" if x["val_loss"] is not None else "    —"
        xp = f"{x['val_pp']:.2f}" if x["val_pp"] is not None else "    —"
        xa = f"{100*x['val_acc']:.2f}" if x["val_acc"] is not None else "    —"
        dl = (
            f"{x['val_loss']-s['val_loss']:+.3f}"
            if (s["val_loss"] is not None and x["val_loss"] is not None)
            else "    —"
        )
        da = (
            f"{100*(x['val_acc']-s['val_acc']):+.2f}"
            if (s["val_acc"] is not None and x["val_acc"] is not None)
            else "    —"
        )
        en = f"{x['entropy']:.2f}" if x["entropy"] is not None else "   —"
        print(
            f"{disp:<12} | {sl:>10} {sp:>10} {sa:>10} | "
            f"{xl:>10} {xp:>10} {xa:>10} | {dl:>8} {da:>8} {en:>6}"
        )
    print()
    print("ΔLoss < 0 → selection improved | ΔAcc > 0 → selection improved")
    print("H_sel     → mean sel_entropy across training (higher = more diverse)")

    suffix = f"_{args.scale}" if args.dataset in SCALED_DATASETS else ""
    out = os.path.join(
        args.logs_dir,
        f"{args.dataset}{suffix}_results_seed{args.seed}.json",
    )
    with open(out, "w") as f:
        json.dump(
            {
                "metadata": {
                    "dataset": args.dataset,
                    "scale": args.scale if args.dataset in SCALED_DATASETS else None,
                    "seed": args.seed,
                    "n_found": n_found,
                    "n_expected": total,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
