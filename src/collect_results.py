#!/usr/bin/env python3
"""
Collect results from all Shakespeare experiments and display a comparison table.
Usage: python collect_results.py [--results_dir PATH] [--logs_dir PATH]
"""

import os
import re
import json
import argparse
from pathlib import Path


def parse_log_file(log_path):
    """Extract final evaluation metrics from a log file."""
    results = {
        "final_val_loss": None,
        "final_val_pp": None,
        "final_val_acc": None,
        "best_val_loss": float("inf"),
        "selection_entropy": None,
    }

    if not os.path.exists(log_path):
        return results

    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Match eval lines: ">Eval: Iter=1000 ... val_loss=2.345 val_pp=10.43 val_acc=0.234"
        # or ">Eval [Selection]: ..."
        eval_match = re.search(
            r">Eval.*val_loss=([0-9.]+)\s+val_pp=([0-9.]+)\s+val_acc=([0-9.]+)", line
        )
        if eval_match:
            vl = float(eval_match.group(1))
            vp = float(eval_match.group(2))
            va = float(eval_match.group(3))
            results["final_val_loss"] = vl
            results["final_val_pp"] = vp
            results["final_val_acc"] = va
            if vl < results["best_val_loss"]:
                results["best_val_loss"] = vl

        # Match selection entropy
        entropy_match = re.search(r"sel_entropy=([0-9.]+)", line)
        if entropy_match:
            results["selection_entropy"] = float(entropy_match.group(1))

    if results["best_val_loss"] == float("inf"):
        results["best_val_loss"] = None

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs_dir",
        default="/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src/logs",
    )
    args = parser.parse_args()

    optimizers = [
        "adamw",
        "ademamix",
        "d-muon",
        "mars",
        "sophiag",
        "soap",
        "lion",
        "signum",
        "adopt",
        "sgd",
    ]
    modes = ["standard", "selection"]

    # Collect results
    all_results = {}
    for opt in optimizers:
        all_results[opt] = {}
        for mode in modes:
            # log_path = os.path.join(args.logs_dir, f"{mode}_shakespeare_{opt}.log")
            log_path = os.path.join(args.logs_dir, f"{mode}_wikitext_{opt}_seed0.log")
            all_results[opt][mode] = parse_log_file(log_path)

    # Print table
    print()
    print("=" * 100)
    print(
        "  OptiSelect WikiText Results: Standard vs Selection"
    )
    print("=" * 100)
    print()

    # Header
    header = f"{'Optimizer':<12} | {'Standard Loss':>14} {'Std PP':>10} {'Std Acc':>10} | {'Select Loss':>14} {'Sel PP':>10} {'Sel Acc':>10} | {'Δ Loss':>8} {'Entropy':>8}"
    print(header)
    print("-" * len(header))

    for opt in optimizers:
        std = all_results[opt]["standard"]
        sel = all_results[opt]["selection"]

        std_loss = f"{std['final_val_loss']:.4f}" if std["final_val_loss"] else "  N/A"
        std_pp = f"{std['final_val_pp']:.3f}" if std["final_val_pp"] else "  N/A"
        std_acc = f"{std['final_val_acc']:.4f}" if std["final_val_acc"] else "  N/A"

        sel_loss = f"{sel['final_val_loss']:.4f}" if sel["final_val_loss"] else "  N/A"
        sel_pp = f"{sel['final_val_pp']:.3f}" if sel["final_val_pp"] else "  N/A"
        sel_acc = f"{sel['final_val_acc']:.4f}" if sel["final_val_acc"] else "  N/A"

        if std["final_val_loss"] and sel["final_val_loss"]:
            delta = sel["final_val_loss"] - std["final_val_loss"]
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "  N/A"

        entropy_str = (
            f"{sel['selection_entropy']:.2f}" if sel["selection_entropy"] else "  N/A"
        )

        print(
            f"{opt:<12} | {std_loss:>14} {std_pp:>10} {std_acc:>10} | {sel_loss:>14} {sel_pp:>10} {sel_acc:>10} | {delta_str:>8} {entropy_str:>8}"
        )

    print()
    print("Δ Loss < 0 means selection IMPROVED over standard training")
    print()

    # Save as JSON
    json_path = os.path.join(args.logs_dir, "wikitext_results_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()