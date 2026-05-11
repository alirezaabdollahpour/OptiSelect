#!/usr/bin/env python3
# =============================================================================
# Publication-ready plots for the OptiSelect WikiText benchmark.
#
# Inputs (auto-discovered, all overridable via CLI):
#   - logs/wikitext_results_v3*.json           : aggregated final metrics
#   - logs/{mode}_wikitext_*_{opt}_seed*.log   : per-iteration trajectories
#
# Outputs (vector PDF + PNG) under figures/wikitext_<tag>/:
#   1. bar_final_metrics      — val loss & accuracy, standard vs selection
#   2. delta_acc              — ΔAcc per optimizer, sorted
#   3. delta_loss             — ΔLoss per optimizer, sorted
#   4. val_loss_curves        — small multiples (10 panels, std vs sel overlaid)
#   5. val_acc_curves         — small multiples
#   6. selection_entropy      — entropy trajectories (1 panel, all opts)
#   7. entropy_vs_delta_acc   — H1/H2 scatter (sign collapse → small ΔAcc)
#   8. flops_vs_acc           — compute-vs-quality Pareto (per-run FLOPs)
#
# Usage:
#   python plot_wikitext_results.py
#   python plot_wikitext_results.py --json logs/wikitext_results_v3.json
# =============================================================================
import argparse, glob, json, os, re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


OPT_ORDER = ["adamw", "ademamix", "d-muon", "mars", "sophiag",
             "soap", "lion", "signum", "adopt", "sgd"]
OPT_DISPLAY = {
    "adamw": "AdamW", "ademamix": "AdEMAMix", "d-muon": "D-Muon",
    "mars": "MARS", "sophiag": "Sophia", "soap": "SOAP",
    "lion": "Lion", "signum": "Signum", "adopt": "ADOPT", "sgd": "SGD",
}
# Family colors keyed to the paper's operator analysis (Section 4.3).
OPT_FAMILY = {
    "adamw":    ("#1f77b4", "Adam-family"),
    "ademamix": ("#2ca02c", "Adam-family + slow momentum"),
    "adopt":    ("#17becf", "Adam-family (lagged variance)"),
    "d-muon":   ("#9467bd", "Matrix preconditioner"),
    "mars":     ("#8c564b", "Variance-reduced Adam"),
    "sophiag":  ("#e377c2", "Curvature-aware"),
    "soap":     ("#bcbd22", "Rotated diagonal"),
    "lion":     ("#ff7f0e", "Sign-based"),
    "signum":   ("#d62728", "Sign-based"),
    "sgd":      ("#7f7f7f", "Identity baseline"),
}
MODE_STYLE = {
    "standard":  {"color": "#666666", "ls": "--", "label": "Standard"},
    "selection": {"color": "#d62728", "ls": "-",  "label": "+ OptiSelect"},
}


def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,    # editable text in vector PDFs
        "ps.fonttype": 42,
    })


# -----------------------------------------------------------------------------
# Aggregated JSON
# -----------------------------------------------------------------------------
def discover_json(logs_dir):
    candidates = sorted(glob.glob(os.path.join(logs_dir, "wikitext_results_v3*.json")))
    if not candidates:
        raise FileNotFoundError(f"No wikitext_results_v3*.json under {logs_dir}")
    # Prefer the proxy-downstream variant if present, else the train one.
    for c in candidates:
        if "proxy-downstream" in c:
            return c
    return candidates[0]


def parse_aggregated(json_path):
    blob = json.load(open(json_path))
    return blob["metadata"], blob["results"]


# -----------------------------------------------------------------------------
# Log parsing — per-iteration trajectories
# -----------------------------------------------------------------------------
EVAL_RE = re.compile(
    r">Eval:\s*Iter=(\d+)[^\n]*?val_loss=([0-9.]+)\s+val_pp=([0-9.]+)\s+val_acc=([0-9.]+)"
)
# Tight, non-greedy-free patterns to avoid backtracking on DDP rank-concatenated lines
# (each line can carry "Train: Iter=...sel_entropy=...Train: Iter=..." multiple times).
TRAIN_RE = re.compile(
    r"Train:\s*Iter=(\d+)\s+train_loss=([0-9.eE+\-]+)"
)
ENT_RE = re.compile(
    r"Iter=(\d+)[^\n]{0,200}?sel_entropy=([0-9.]+)"
)


def parse_log(path):
    """Parse one log file — dedup by iteration (DDP rank duplication)."""
    val = {}      # iter -> (loss, pp, acc)
    train = defaultdict(list)
    entropy = defaultdict(list)

    if not os.path.exists(path):
        return None

    with open(path) as f:
        for line in f:
            for m in EVAL_RE.finditer(line):
                it = int(m.group(1))
                val[it] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            for m in TRAIN_RE.finditer(line):
                train[int(m.group(1))].append(float(m.group(2)))
            for m in ENT_RE.finditer(line):
                entropy[int(m.group(1))].append(float(m.group(2)))

    if not val and not train:
        return None
    return {
        "val_iters": np.array(sorted(val.keys())),
        "val_loss":  np.array([val[i][0] for i in sorted(val.keys())]),
        "val_pp":    np.array([val[i][1] for i in sorted(val.keys())]),
        "val_acc":   np.array([val[i][2] for i in sorted(val.keys())]),
        "train_iters": np.array(sorted(train.keys())),
        "train_loss":  np.array([np.mean(train[i]) for i in sorted(train.keys())]),
        "ent_iters":   np.array(sorted(entropy.keys())),
        "ent":         np.array([np.mean(entropy[i]) for i in sorted(entropy.keys())]),
    }


def discover_logs(logs_dir, proxy_source, model_size):
    """Build {opt: {mode: trajectory_dict}} from log files."""
    size_tag = "" if model_size == "25m" else f"{model_size}_"
    proxy_tag = "" if proxy_source == "train" else f"proxy-{proxy_source}_"
    out = {}
    for opt in OPT_ORDER:
        out[opt] = {}
        for mode in ("standard", "selection"):
            name = f"{mode}_wikitext_{size_tag}{proxy_tag}{opt}_seed0.log"
            traj = parse_log(os.path.join(logs_dir, name))
            if traj is not None:
                out[opt][mode] = traj
    return out


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def save(fig, out_dir, name, formats):
    for fmt in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"))
    plt.close(fig)
    print(f"  ✓ {name}.{'/'.join(formats)}")


def fig_bar_final_metrics(results, out_dir, formats, title_suffix):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(OPT_ORDER))
    w = 0.38

    std_loss = [results[o]["standard"]["val_loss"] for o in OPT_ORDER]
    sel_loss = [results[o]["selection"]["val_loss"] for o in OPT_ORDER]
    std_acc  = [100 * (results[o]["standard"]["val_acc"]  or 0) for o in OPT_ORDER]
    sel_acc  = [100 * (results[o]["selection"]["val_acc"] or 0) for o in OPT_ORDER]

    axL.bar(x - w/2, std_loss, w, label="Standard",     color="#bbbbbb", edgecolor="black", linewidth=0.4)
    axL.bar(x + w/2, sel_loss, w, label="+ OptiSelect", color="#d62728", edgecolor="black", linewidth=0.4)
    axL.set_ylabel("Validation loss")
    axL.set_xticks(x)
    axL.set_xticklabels([OPT_DISPLAY[o] for o in OPT_ORDER], rotation=30, ha="right")
    axL.legend(frameon=False, loc="upper left")
    axL.set_title(f"Final validation loss {title_suffix}")
    axL.grid(axis="y", alpha=0.3)

    axR.bar(x - w/2, std_acc, w, label="Standard",     color="#bbbbbb", edgecolor="black", linewidth=0.4)
    axR.bar(x + w/2, sel_acc, w, label="+ OptiSelect", color="#d62728", edgecolor="black", linewidth=0.4)
    axR.set_ylabel("Validation accuracy (%)")
    axR.set_xticks(x)
    axR.set_xticklabels([OPT_DISPLAY[o] for o in OPT_ORDER], rotation=30, ha="right")
    axR.legend(frameon=False, loc="upper left")
    axR.set_title(f"Final validation accuracy {title_suffix}")
    axR.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, out_dir, "bar_final_metrics", formats)


def fig_delta(results, key, out_dir, formats, title_suffix):
    """key='val_loss' or 'val_acc'."""
    deltas = []
    for o in OPT_ORDER:
        s = results[o]["standard"].get(key)
        x = results[o]["selection"].get(key)
        if s is None or x is None:
            continue
        if key == "val_acc":
            deltas.append((o, 100 * (x - s)))
        else:
            deltas.append((o, x - s))
    deltas.sort(key=lambda kv: kv[1])
    if not deltas:
        return

    opts, vals = zip(*deltas)
    colors = [OPT_FAMILY[o][0] for o in opts]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh([OPT_DISPLAY[o] for o in opts], vals,
                   color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    if key == "val_acc":
        ax.set_xlabel(r"$\Delta$ Accuracy (pp)  ←  worse | better  →")
        title = f"OptiSelect accuracy gain {title_suffix}"
        fname = "delta_acc"
    else:
        ax.set_xlabel(r"$\Delta$ Validation loss  ←  better | worse  →")
        title = f"OptiSelect loss change {title_suffix}"
        fname = "delta_loss"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, vals):
        ha = "left" if val >= 0 else "right"
        offset = 0.02 * (max(abs(v) for v in vals) or 1) * (1 if val >= 0 else -1)
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.2f}", va="center", ha=ha, fontsize=8)

    fig.tight_layout()
    save(fig, out_dir, fname, formats)


def fig_curves_grid(traj, metric, ylabel, out_dir, formats, title_suffix, fname):
    """5x2 small multiples; metric in {'val_loss','val_acc'}. Plot vs iter."""
    n = len(OPT_ORDER)
    ncols, nrows = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 11), sharex=True)
    axes = axes.flatten()

    for ax, opt in zip(axes, OPT_ORDER):
        for mode in ("standard", "selection"):
            t = traj.get(opt, {}).get(mode)
            if t is None or len(t["val_iters"]) == 0:
                continue
            y = t[metric]
            if metric == "val_acc":
                y = 100 * y
            ax.plot(t["val_iters"], y, **MODE_STYLE[mode], linewidth=1.6)
        ax.set_title(OPT_DISPLAY[opt], fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylabel(ylabel)
    for ax in axes[-ncols:]:
        ax.set_xlabel("Iteration")

    # one shared legend
    handles = [plt.Line2D([0], [0], **{k: v for k, v in s.items() if k != "label"},
                           label=s["label"]) for s in MODE_STYLE.values()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f"{ylabel} curves {title_suffix}", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, fname, formats)


def fig_entropy(traj, out_dir, formats, title_suffix):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = 0
    for opt in OPT_ORDER:
        t = traj.get(opt, {}).get("selection")
        if t is None or len(t["ent_iters"]) == 0:
            continue
        color = OPT_FAMILY[opt][0]
        # light smoothing for readability
        y = t["ent"]
        if len(y) > 30:
            k = max(5, len(y) // 60)
            kernel = np.ones(k) / k
            y = np.convolve(y, kernel, mode="same")
        ax.plot(t["ent_iters"], y, color=color, linewidth=1.4, label=OPT_DISPLAY[opt])
        plotted += 1
    if plotted == 0:
        plt.close(fig); return
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Selection entropy $\mathcal{H}_{\mathrm{sel}}$ (nats)")
    ax.set_title(f"Boltzmann selection entropy over training {title_suffix}")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    save(fig, out_dir, "selection_entropy", formats)


def fig_entropy_vs_delta_acc(traj, results, out_dir, formats, title_suffix):
    """Scatter: mean entropy (selection run) vs ΔAcc — Theorem 1 / H1+H2 visual."""
    pts = []
    for opt in OPT_ORDER:
        t = traj.get(opt, {}).get("selection")
        s = results[opt]["standard"].get("val_acc")
        x = results[opt]["selection"].get("val_acc")
        if t is None or s is None or x is None or len(t["ent"]) == 0:
            continue
        # Use the post-warmup mean to avoid the warmup transient.
        keep = t["ent_iters"] > t["ent_iters"].max() * 0.1
        mean_ent = float(np.mean(t["ent"][keep])) if keep.any() else float(np.mean(t["ent"]))
        pts.append((opt, mean_ent, 100 * (x - s)))
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for opt, e, d in pts:
        color = OPT_FAMILY[opt][0]
        ax.scatter(e, d, s=85, color=color, edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(OPT_DISPLAY[opt], (e, d), xytext=(5, 4),
                    textcoords="offset points", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel(r"Mean selection entropy $\overline{\mathcal{H}}_{\mathrm{sel}}$ (nats)")
    ax.set_ylabel(r"$\Delta$ Validation accuracy (pp)")
    ax.set_title(f"Selection diversity vs OptiSelect gain {title_suffix}")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="best")

    # OLS trend if ≥3 points
    if len(pts) >= 3:
        es = np.array([p[1] for p in pts])
        ds = np.array([p[2] for p in pts])
        m, b = np.polyfit(es, ds, 1)
        xs = np.linspace(es.min(), es.max(), 50)
        ax.plot(xs, m * xs + b, "k--", linewidth=0.9, alpha=0.5,
                label=f"OLS: slope={m:.2f}")
        ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    save(fig, out_dir, "entropy_vs_delta_acc", formats)


def fig_flops_vs_acc(results, metadata, out_dir, formats, title_suffix):
    """Pareto-style scatter: total training FLOPs vs final val accuracy.

    FLOPs model (Kaplan/Chinchilla "C = 6ND"):
      - N = total params (incl. embedding) computed from architecture
      - D = total tokens processed (ITERS × batch × seq × acc)
      - standard:  C_std = 6 × N × D
      - selection: forward over 2B candidates + backward on B selected
                   = (2·2NB + 4NB) per step = 8NB  →  C_sel = (8/6) × C_std
    """
    n_embd  = int(metadata.get("n_embd", 384))
    n_layer = int(metadata.get("n_layer", 6))
    vocab   = 50257  # GPT-2 BPE
    # Llama transformer block ≈ 12·n_embd² (4 attn + 8 SwiGLU FFN)
    N_total = vocab * n_embd + n_layer * 12 * n_embd ** 2

    iters = int(metadata.get("iterations", 12000))
    tokens_per_step = (int(metadata.get("batch_size", 16))
                       * int(metadata.get("seq_len", 512))
                       * int(metadata.get("acc_steps", 1)))
    D_std = iters * tokens_per_step
    flops_std = 6.0 * N_total * D_std
    flops_sel = (8.0 / 6.0) * flops_std

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    opt_handles = []
    for opt in OPT_ORDER:
        color = OPT_FAMILY[opt][0]
        s_acc = results[opt]["standard"].get("val_acc")
        x_acc = results[opt]["selection"].get("val_acc")
        if s_acc is not None:
            ax.scatter(flops_std, 100 * s_acc, s=80, color=color,
                       edgecolor="black", linewidth=0.5, marker="o", zorder=3)
        if x_acc is not None:
            ax.scatter(flops_sel, 100 * x_acc, s=80, color=color,
                       edgecolor="black", linewidth=0.5, marker="s", zorder=3)
        if s_acc is not None and x_acc is not None:
            ax.plot([flops_std, flops_sel], [100 * s_acc, 100 * x_acc],
                    color=color, linewidth=1.2, alpha=0.6, zorder=2)
        opt_handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="-", color=color,
                       markersize=7, markeredgecolor="black", markeredgewidth=0.4,
                       label=OPT_DISPLAY[opt])
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training FLOPs  (log scale,  $C = 6\\,N\\,D$)")
    ax.set_ylabel("Final validation accuracy (%)")
    ax.set_title(f"FLOPs vs accuracy {title_suffix}")
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(flops_std * 0.9, flops_sel * 1.1)

    # Two-column legend on the right: marker shape (mode) + colors (optimizer).
    shape_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   markerfacecolor="#888", markeredgecolor="black",
                   label=f"Standard  ({flops_std:.2e})"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=8,
                   markerfacecolor="#888", markeredgecolor="black",
                   label=f"+ OptiSelect  ({flops_sel:.2e}, +33%)"),
    ]
    leg1 = ax.legend(handles=shape_handles, loc="upper left", frameon=False,
                     fontsize=9, title="Mode (FLOPs)")
    ax.add_artist(leg1)
    ax.legend(handles=opt_handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9, title="Optimizer")

    fig.tight_layout()
    save(fig, out_dir, "flops_vs_acc", formats)


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=None,
                   help="Path to aggregated wikitext_results_v3*.json (default: auto-detect).")
    p.add_argument("--logs-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: figures/wikitext_<tag>/ next to this script).")
    p.add_argument("--format", default="pdf,png", help="Comma-separated: pdf,png")
    args = p.parse_args()

    setup_matplotlib()
    formats = [f.strip() for f in args.format.split(",") if f.strip()]

    json_path = args.json or discover_json(args.logs_dir)
    print(f"Aggregated JSON: {json_path}")
    metadata, results = parse_aggregated(json_path)

    proxy = "train"
    base = os.path.basename(json_path)
    m = re.search(r"proxy-([a-zA-Z0-9_]+)", base)
    if m: proxy = m.group(1)
    size = metadata.get("model_size", "25m")

    tag_bits = [size]
    if proxy != "train": tag_bits.append(f"proxy-{proxy}")
    tag = "_".join(tag_bits)
    title_suffix = f"({size.upper()}, proxy={proxy})"

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "figures", f"wikitext_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir:      {out_dir}")
    print(f"Model:           {metadata.get('model')}")
    print(f"Iters:           {metadata.get('iterations')}  | seed={metadata.get('seed')}\n")

    print("Parsing logs...")
    traj = discover_logs(args.logs_dir, proxy, size)
    found = sum(1 for o in OPT_ORDER for m_ in ("standard", "selection") if traj.get(o, {}).get(m_))
    print(f"  parsed {found}/20 trajectories\n")

    print("Generating figures:")
    fig_bar_final_metrics(results, out_dir, formats, title_suffix)
    fig_delta(results, "val_acc",  out_dir, formats, title_suffix)
    fig_delta(results, "val_loss", out_dir, formats, title_suffix)
    fig_curves_grid(traj, "val_loss", "Validation loss",       out_dir, formats, title_suffix, "val_loss_curves")
    fig_curves_grid(traj, "val_acc",  "Validation accuracy (%)", out_dir, formats, title_suffix, "val_acc_curves")
    fig_entropy(traj, out_dir, formats, title_suffix)
    fig_entropy_vs_delta_acc(traj, results, out_dir, formats, title_suffix)
    fig_flops_vs_acc(results, metadata, out_dir, formats, title_suffix)

    print(f"\nDone. Figures in: {out_dir}")


if __name__ == "__main__":
    main()
