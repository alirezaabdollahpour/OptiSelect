#!/usr/bin/env python3
# Plot downstream zero-shot results produced by eval_downstream.py.
# Reads <results-dir>/downstream_eval.json and writes:
#   - downstream_bars.{pdf,png}    grouped bars (std vs sel) per task per opt
#   - downstream_delta.{pdf,png}   per-task ΔAcc per optimizer (heatmap-ish)
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt

OPT_ORDER = ["adamw", "ademamix", "d-muon", "mars", "sophiag",
             "soap", "lion", "signum", "adopt", "sgd"]
OPT_DISPLAY = {"adamw":"AdamW","ademamix":"AdEMAMix","d-muon":"D-Muon",
               "mars":"MARS","sophiag":"Sophia","soap":"SOAP",
               "lion":"Lion","signum":"Signum","adopt":"ADOPT","sgd":"SGD"}
TASK_DISPLAY = {"piqa":"PIQA","siqa":"SIQA","hellaswag":"HellaSwag",
                "anli":"ANLI","mmlu":"MMLU"}
RANDOM_BASELINE = {"piqa":50.0,"siqa":33.3,"hellaswag":25.0,"anli":33.3,"mmlu":25.0}


def setup():
    plt.rcParams.update({
        "font.family":"serif","font.size":10,"axes.titlesize":11,
        "axes.spines.top":False,"axes.spines.right":False,
        "figure.dpi":120,"savefig.bbox":"tight","savefig.pad_inches":0.05,
        "pdf.fonttype":42,"ps.fonttype":42,
    })


def load(path):
    blob = json.load(open(path))
    by_opt = {}
    for k, v in blob.items():
        if "/" not in k: continue
        mode, opt = k.split("/", 1)
        by_opt.setdefault(opt, {})[mode] = v
    return by_opt


def fig_bars(by_opt, tasks, out_dir, formats):
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(3.4 * n_tasks, 4.2), sharey=False)
    if n_tasks == 1: axes = [axes]
    x = np.arange(len(OPT_ORDER))
    w = 0.38
    for ax, t in zip(axes, tasks):
        std = [100*by_opt.get(o,{}).get("standard",  {}).get(t,{}).get("acc", np.nan) for o in OPT_ORDER]
        sel = [100*by_opt.get(o,{}).get("selection", {}).get(t,{}).get("acc", np.nan) for o in OPT_ORDER]
        ax.bar(x - w/2, std, w, label="Standard",     color="#bbbbbb", edgecolor="black", linewidth=0.4)
        ax.bar(x + w/2, sel, w, label="+ OptiSelect", color="#d62728", edgecolor="black", linewidth=0.4)
        if t in RANDOM_BASELINE:
            ax.axhline(RANDOM_BASELINE[t], color="black", linestyle=":", linewidth=0.8,
                       label=f"random ({RANDOM_BASELINE[t]:.0f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels([OPT_DISPLAY[o] for o in OPT_ORDER], rotation=40, ha="right")
        ax.set_title(TASK_DISPLAY.get(t, t))
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    for f in formats:
        fig.savefig(os.path.join(out_dir, f"downstream_bars.{f}"))
    plt.close(fig)
    print("  ✓ downstream_bars")


def fig_delta(by_opt, tasks, out_dir, formats):
    """Heatmap of ΔAcc (sel − std) per (optimizer × task)."""
    M = np.full((len(OPT_ORDER), len(tasks)), np.nan)
    for i, o in enumerate(OPT_ORDER):
        for j, t in enumerate(tasks):
            s = by_opt.get(o,{}).get("standard",  {}).get(t,{}).get("acc")
            x = by_opt.get(o,{}).get("selection", {}).get(t,{}).get("acc")
            if s is not None and x is not None:
                M[i, j] = 100 * (x - s)
    if np.isnan(M).all():
        print("  (skipping delta heatmap — no overlapping std+sel data)")
        return
    vmax = max(np.nanmax(np.abs(M)), 1.0)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(tasks), 0.5 * len(OPT_ORDER) + 1.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_DISPLAY.get(t,t) for t in tasks], rotation=20, ha="right")
    ax.set_yticks(range(len(OPT_ORDER)))
    ax.set_yticklabels([OPT_DISPLAY[o] for o in OPT_ORDER])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(v) < vmax*0.6 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label(r"$\Delta$ Accuracy (pp), selection $-$ standard")
    ax.set_title("OptiSelect downstream gain per task")
    fig.tight_layout()
    for f in formats:
        fig.savefig(os.path.join(out_dir, f"downstream_delta.{f}"))
    plt.close(fig)
    print("  ✓ downstream_delta")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="/mloscratch/homes/aabdolla/results/wikitext_exp_v3_proxy_downstream/downstream_eval.json")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tasks", default="piqa,siqa,hellaswag,anli,mmlu")
    p.add_argument("--format", default="pdf,png")
    args = p.parse_args()
    setup()
    tasks = [t.strip() for t in args.tasks.split(",")]
    formats = [f.strip() for f in args.format.split(",")]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "figures",
        "wikitext_25m_proxy-downstream",
    )
    os.makedirs(out_dir, exist_ok=True)
    by_opt = load(args.json)
    print(f"Loaded {sum(len(v) for v in by_opt.values())} mode-rows for "
          f"{len(by_opt)} optimizers")
    print("Generating downstream figures:")
    fig_bars(by_opt, tasks, out_dir, formats)
    fig_delta(by_opt, tasks, out_dir, formats)
    print(f"Done. {out_dir}")


if __name__ == "__main__":
    main()
