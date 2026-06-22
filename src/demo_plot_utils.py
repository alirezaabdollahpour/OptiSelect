"""Publication plotting utilities for the DCLM-Edu demo benchmark.

The DDP runner writes one ``summary.json`` per run.  This module turns those
summaries into tidy data frames and fair comparison plots for standard
optimizers versus their OptiSelect counterparts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_HOME = Path.home()
HOME_ROOT = REPO_ROOT.parent
RESULTS_ROOT = HOME_ROOT / "results"

OPTIMIZER_ORDER = [
    "adamw",
    "d-muon",
    "sgd",
    "signsgd",
    "ademamix",
    "sophia",
]
OPTIMIZER_DISPLAY = {
    "adamw": "AdamW",
    "d-muon": "d-Muon",
    "sgd": "SGD",
    "signsgd": "signSGD",
    "ademamix": "AdEMAMix",
    "sophia": "SophiaG",
}
OPTIMIZER_COLORS = {
    "adamw": "#4477AA",
    "d-muon": "#AA3377",
    "sgd": "#777777",
    "signsgd": "#CC6677",
    "ademamix": "#228833",
    "sophia": "#EE7733",
}
MODE_DISPLAY = {
    "standard": "Standard",
    "optiselect": "OptiSelect",
}
MODE_STYLE = {
    "standard": {
        "color": "#555555",
        "linestyle": "--",
        "marker": "o",
        "linewidth": 1.7,
        "markersize": 4.2,
    },
    "optiselect": {
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2.0,
        "markersize": 4.4,
    },
}
PRIMARY_LM_EVAL_METRICS = [
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "perplexity,none",
]
LOWER_IS_BETTER_METRICS = {"val_loss", "val_ppl", "perplexity,none"}


@dataclass
class DemoPlotData:
    run_df: pd.DataFrame
    curve_df: pd.DataFrame
    train_df: pd.DataFrame
    pair_df: pd.DataFrame
    fairness_df: pd.DataFrame
    summary_paths: list[Path] = field(default_factory=list)
    added_counterparts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def configure_matplotlib(usetex: bool = False) -> None:
    """Set compact, paper-friendly Matplotlib defaults."""

    use_tex = bool(usetex and shutil.which("latex"))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "TeX Gyre Termes",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "text.usetex": use_tex,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_path(path_like: str | Path) -> Path:
    """Resolve absolute paths and useful relative paths such as ``results/foo``."""

    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    candidates = [
        Path.cwd() / p,
        REPO_ROOT / p,
        REPO_ROOT.parent / p,
        HOME_ROOT / p,
        SYSTEM_HOME / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _json_hash(obj: object) -> str:
    text = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _infer_seed(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


def _infer_accelerator(path: Path) -> str | None:
    match = re.search(r"\b(h\d{2,3})\b", str(path).lower())
    return match.group(1) if match else None


def _mode_from_summary(summary: Mapping[str, object]) -> str:
    return "optiselect" if bool(summary.get("optiselect", False)) else "standard"


def _optimizer_display(opt: str) -> str:
    return OPTIMIZER_DISPLAY.get(opt, opt)


def _run_label(summary: Mapping[str, object]) -> str:
    opt = str(summary.get("optimizer", summary.get("run_key", "unknown")))
    mode = _mode_from_summary(summary)
    return f"{MODE_DISPLAY[mode]} {_optimizer_display(opt)}"


def discover_summary_paths(result_dirs: Sequence[str | Path]) -> list[Path]:
    """Find run-level ``summary.json`` files under user-provided paths."""

    paths: set[Path] = set()
    for raw in result_dirs:
        root = resolve_path(raw)
        if root.is_file() and root.name == "summary.json":
            paths.add(root.resolve())
            continue
        if not root.exists():
            continue
        if (root / "summary.json").exists():
            paths.add((root / "summary.json").resolve())
        for child in root.glob("*/summary.json"):
            paths.add(child.resolve())
        for child in root.glob("*/*/summary.json"):
            paths.add(child.resolve())
    return sorted(paths)


def _candidate_summary_paths(search_roots: Sequence[str | Path]) -> list[Path]:
    """Find plausible run summaries while avoiding a full recursive crawl."""

    paths: set[Path] = set()
    for raw in search_roots:
        root = resolve_path(raw)
        if not root.exists() or not root.is_dir():
            continue
        for pattern in ("*/summary.json", "*/*/summary.json"):
            for path in root.glob(pattern):
                paths.add(path.resolve())
    return sorted(paths)


def _load_summary(path: Path) -> dict:
    with path.open() as handle:
        data = json.load(handle)
    data["_summary_path"] = str(path)
    data["_run_dir"] = str(path.parent)
    data["_result_root"] = str(path.parent.parent)
    data["_result_name"] = path.parent.parent.name
    return data


def _summary_to_row(summary: Mapping[str, object]) -> dict:
    distributed = summary.get("distributed") or {}
    model_config = summary.get("model_config") or {}
    dataset_metadata = summary.get("dataset_metadata") or {}
    opt = str(summary.get("optimizer", summary.get("run_key", "unknown")))
    mode = _mode_from_summary(summary)
    steps = int(summary.get("steps", 0) or 0)
    final_loss = _as_float(summary.get("final_val_loss"))
    final_ppl = _as_float(summary.get("final_val_ppl"))
    if final_ppl is None and final_loss is not None:
        final_ppl = float(math.exp(final_loss))
    path = Path(str(summary["_summary_path"]))
    return {
        "run_key": summary.get("run_key"),
        "optimizer": opt,
        "optimizer_display": _optimizer_display(opt),
        "mode": mode,
        "mode_display": MODE_DISPLAY[mode],
        "label": _run_label(summary),
        "summary_path": str(path),
        "run_dir": str(summary.get("_run_dir")),
        "result_root": str(summary.get("_result_root")),
        "result_name": summary.get("_result_name"),
        "seed": _infer_seed(path),
        "accelerator": _infer_accelerator(path),
        "world_size": distributed.get("world_size", 1),
        "global_device_batch_size": distributed.get("global_device_batch_size"),
        "local_device_batch_size": distributed.get("local_device_batch_size"),
        "grad_accum_steps": distributed.get("grad_accum_steps"),
        "num_params": summary.get("num_params"),
        "steps": steps,
        "effective_batch_tokens": summary.get("effective_batch_tokens"),
        "effective_batch_examples": summary.get("effective_batch_examples"),
        "target_update_tokens": summary.get("target_update_tokens"),
        "actual_update_tokens": summary.get("actual_update_tokens"),
        "actual_candidate_tokens": summary.get("actual_candidate_tokens", 0),
        "actual_processed_tokens": summary.get("actual_processed_tokens"),
        "training_time_sec": summary.get("training_time_sec"),
        "wall_time_sec": summary.get("wall_time_sec"),
        "peak_vram_gb": summary.get("peak_vram_gb"),
        "eval_batches": summary.get("eval_batches"),
        "final_eval_batches": summary.get("final_eval_batches"),
        "final_val_loss": final_loss,
        "final_val_ppl": final_ppl,
        "model_hash": _json_hash(model_config),
        "dataset_fingerprint": dataset_metadata.get("fingerprint"),
        "dataset_hash": _json_hash(dataset_metadata),
    }


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _per_step_rates(summary: Mapping[str, object]) -> tuple[float, float, float]:
    steps = max(int(summary.get("steps", 0) or 0), 1)
    effective = float(summary.get("effective_batch_tokens", 0) or 0)
    update = float(summary.get("actual_update_tokens", 0) or 0) / steps
    candidate = float(summary.get("actual_candidate_tokens", 0) or 0) / steps
    processed = float(summary.get("actual_processed_tokens", 0) or 0) / steps
    if update <= 0:
        update = effective
    if processed <= 0:
        processed = update + candidate
    return update, candidate, processed


def _summary_to_curve_rows(summary: Mapping[str, object]) -> list[dict]:
    hist = summary.get("history") or {}
    if not isinstance(hist, Mapping):
        return []
    val_steps = list(hist.get("val_steps", []) or [])
    val_loss = list(hist.get("val_loss", []) or [])
    val_ppl = list(hist.get("val_ppl", []) or [])
    update_rate, candidate_rate, processed_rate = _per_step_rates(summary)
    base = _summary_to_row(summary)
    rows: list[dict] = []
    for i, step in enumerate(val_steps):
        loss = _as_float(val_loss[i]) if i < len(val_loss) else None
        ppl = _as_float(val_ppl[i]) if i < len(val_ppl) else None
        rows.append(
            {
                **base,
                "step": int(step),
                "val_loss": loss,
                "val_ppl": ppl,
                "update_tokens": int(round(int(step) * update_rate)),
                "candidate_tokens": int(round(int(step) * candidate_rate)),
                "processed_tokens": int(round(int(step) * processed_rate)),
                "eval_kind": "interval",
            }
        )
    final_step = int(summary.get("steps", 0) or 0)
    if final_step and (not val_steps or int(val_steps[-1]) != final_step):
        final_loss = _as_float(summary.get("final_val_loss"))
        final_ppl = _as_float(summary.get("final_val_ppl"))
        if final_ppl is None and final_loss is not None:
            final_ppl = float(math.exp(final_loss))
        rows.append(
            {
                **base,
                "step": final_step,
                "val_loss": final_loss,
                "val_ppl": final_ppl,
                "update_tokens": int(summary.get("actual_update_tokens", 0) or round(final_step * update_rate)),
                "candidate_tokens": int(summary.get("actual_candidate_tokens", 0) or round(final_step * candidate_rate)),
                "processed_tokens": int(summary.get("actual_processed_tokens", 0) or round(final_step * processed_rate)),
                "eval_kind": "final",
            }
        )
    return rows


def _summary_to_train_rows(summary: Mapping[str, object], stride: int = 10) -> list[dict]:
    hist = summary.get("history") or {}
    if not isinstance(hist, Mapping):
        return []
    steps = list(hist.get("steps", []) or [])
    losses = list(hist.get("train_loss", []) or [])
    update_rate, candidate_rate, processed_rate = _per_step_rates(summary)
    base = _summary_to_row(summary)
    rows: list[dict] = []
    keep_last = len(steps) - 1
    for i, raw_step in enumerate(steps):
        if i % stride != 0 and i != keep_last:
            continue
        step_count = int(raw_step) + 1
        rows.append(
            {
                **base,
                "step": int(raw_step),
                "step_count": step_count,
                "train_loss": _as_float(losses[i]) if i < len(losses) else None,
                "update_tokens": int(round(step_count * update_rate)),
                "candidate_tokens": int(round(step_count * candidate_rate)),
                "processed_tokens": int(round(step_count * processed_rate)),
            }
        )
    return rows


def _find_counterparts(
    initial: Sequence[dict],
    search_roots: Sequence[str | Path],
    existing_paths: set[Path],
) -> list[Path]:
    if not initial:
        return []
    initial_rows = pd.DataFrame([_summary_to_row(s) for s in initial])
    all_modes = {"standard", "optiselect"}
    needed: list[tuple[pd.Series, str]] = []
    for opt, group in initial_rows.groupby("optimizer", sort=False):
        have = set(group["mode"])
        for mode in sorted(all_modes - have):
            needed.append((group.iloc[0], mode))
    if not needed:
        return []

    candidates = []
    for path in _candidate_summary_paths(search_roots):
        if path in existing_paths:
            continue
        try:
            summary = _load_summary(path)
            row = _summary_to_row(summary)
        except Exception:
            continue
        candidates.append((path, row))

    added: list[Path] = []
    for ref, wanted_mode in needed:
        matches = []
        for path, row in candidates:
            if row["optimizer"] != ref["optimizer"] or row["mode"] != wanted_mode:
                continue
            if ref.get("model_hash") and row.get("model_hash") != ref.get("model_hash"):
                continue
            if ref.get("dataset_fingerprint") and row.get("dataset_fingerprint") != ref.get("dataset_fingerprint"):
                continue
            score = 0
            for key, weight in (
                ("seed", 8),
                ("world_size", 4),
                ("effective_batch_tokens", 3),
                ("accelerator", 2),
            ):
                if pd.notna(ref.get(key)) and ref.get(key) == row.get(key):
                    score += weight
            matches.append((score, path))
        if matches:
            matches.sort(key=lambda item: (-item[0], str(item[1])))
            added.append(matches[0][1])
    return sorted(set(added))


def load_demo_results(
    result_dirs: Sequence[str | Path],
    *,
    auto_discover_counterparts: bool = True,
    search_roots: Sequence[str | Path] | None = None,
    train_stride: int = 10,
) -> DemoPlotData:
    """Load DDP/notebook run summaries into tidy data frames.

    Parameters
    ----------
    result_dirs:
        Directories such as ``results/demo_dclm_optiselect_sophia_ddp2_h100_seed42``.
        A directory can contain one run or many run subdirectories.
    auto_discover_counterparts:
        If true, search nearby result directories for the missing standard or
        OptiSelect counterpart with the same optimizer, model, dataset, seed,
        and world size when possible.
    search_roots:
        Optional roots for counterpart discovery.  Defaults to the parents of
        ``result_dirs`` plus ``~/results``.
    """

    paths = discover_summary_paths(result_dirs)
    warnings: list[str] = []
    if not paths:
        warnings.append("No summary.json files were found under the requested result directories.")
        empty = pd.DataFrame()
        return DemoPlotData(empty, empty, empty, empty, empty, [], [], warnings)

    summaries = [_load_summary(path) for path in paths]
    added: list[Path] = []
    if auto_discover_counterparts:
        if search_roots is None:
            roots = {Path(p).parent if Path(p).is_file() else resolve_path(p).parent for p in result_dirs}
            roots.add(RESULTS_ROOT)
            search_roots = sorted(roots)
        added = _find_counterparts(summaries, search_roots, set(paths))
        for path in added:
            paths.append(path)
            summaries.append(_load_summary(path))
    paths = sorted(set(paths))

    run_df = pd.DataFrame([_summary_to_row(s) for s in summaries])
    curve_df = pd.DataFrame([row for s in summaries for row in _summary_to_curve_rows(s)])
    train_df = pd.DataFrame([row for s in summaries for row in _summary_to_train_rows(s, stride=train_stride)])
    pair_df = make_pair_table(run_df)
    fairness_df = make_fairness_report(run_df)
    for _, row in fairness_df[fairness_df["status"] == "warn"].iterrows():
        warnings.append(str(row["message"]))
    return DemoPlotData(run_df, curve_df, train_df, pair_df, fairness_df, paths, added, warnings)


def make_fairness_report(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame(columns=["check", "status", "message"])
    rows: list[dict] = []

    def add(check: str, ok: bool, message: str) -> None:
        rows.append({"check": check, "status": "ok" if ok else "warn", "message": message})

    for col, label in (
        ("model_hash", "model architecture"),
        ("dataset_fingerprint", "dataset fingerprint"),
        ("effective_batch_tokens", "global update batch tokens"),
        ("world_size", "DDP world size"),
    ):
        vals = sorted(str(v) for v in run_df[col].dropna().unique())
        add(label, len(vals) <= 1, f"{label}: {', '.join(vals) if vals else 'missing'}")

    for opt in sorted(run_df["optimizer"].unique(), key=_optimizer_sort_key):
        modes = set(run_df.loc[run_df["optimizer"] == opt, "mode"])
        missing = sorted({"standard", "optiselect"} - modes)
        add(
            f"{opt} pair",
            not missing,
            f"{_optimizer_display(opt)} has {', '.join(sorted(modes))}; missing {', '.join(missing) or 'none'}",
        )
    return pd.DataFrame(rows)


def make_pair_table(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    numeric_cols = [
        "steps",
        "actual_update_tokens",
        "actual_candidate_tokens",
        "actual_processed_tokens",
        "wall_time_sec",
        "training_time_sec",
        "peak_vram_gb",
        "final_val_loss",
        "final_val_ppl",
    ]
    agg = (
        run_df.groupby(["optimizer", "mode"], as_index=False)[numeric_cols]
        .mean(numeric_only=True)
    )
    rows = []
    for opt in sorted(agg["optimizer"].unique(), key=_optimizer_sort_key):
        std = agg[(agg["optimizer"] == opt) & (agg["mode"] == "standard")]
        sel = agg[(agg["optimizer"] == opt) & (agg["mode"] == "optiselect")]
        if std.empty or sel.empty:
            continue
        s = std.iloc[0]
        o = sel.iloc[0]
        rows.append(
            {
                "optimizer": opt,
                "optimizer_display": _optimizer_display(opt),
                "standard_final_val_loss": s["final_val_loss"],
                "optiselect_final_val_loss": o["final_val_loss"],
                "delta_val_loss": o["final_val_loss"] - s["final_val_loss"],
                "standard_final_val_ppl": s["final_val_ppl"],
                "optiselect_final_val_ppl": o["final_val_ppl"],
                "ppl_ratio": _safe_ratio(o["final_val_ppl"], s["final_val_ppl"]),
                "update_token_ratio": _safe_ratio(o["actual_update_tokens"], s["actual_update_tokens"]),
                "processed_token_ratio": _safe_ratio(o["actual_processed_tokens"], s["actual_processed_tokens"]),
                "wall_time_ratio": _safe_ratio(o["wall_time_sec"], s["wall_time_sec"]),
                "peak_vram_delta_gb": o["peak_vram_gb"] - s["peak_vram_gb"],
            }
        )
    return pd.DataFrame(rows)


def _safe_ratio(num: float, den: float) -> float:
    if den is None or den == 0 or pd.isna(den):
        return np.nan
    return float(num) / float(den)


def _optimizer_sort_key(opt: str) -> tuple[int, str]:
    try:
        return (OPTIMIZER_ORDER.index(opt), opt)
    except ValueError:
        return (len(OPTIMIZER_ORDER), opt)


def save_tables(data: DemoPlotData, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "run_summary.csv": data.run_df,
        "validation_curves.csv": data.curve_df,
        "train_curves.csv": data.train_df,
        "optimizer_pair_table.csv": data.pair_df,
        "fairness_report.csv": data.fairness_df,
    }
    written = []
    for name, frame in tables.items():
        path = out / name
        frame.to_csv(path, index=False)
        written.append(path)
    return written


def _scaled_x(frame: pd.DataFrame, x: str) -> tuple[pd.Series, str]:
    if x in {"update_tokens", "processed_tokens", "candidate_tokens"}:
        return frame[x] / 1e9, {
            "update_tokens": "Selected update tokens (B)",
            "processed_tokens": "Total processed tokens (B)",
            "candidate_tokens": "Candidate-scored tokens (B)",
        }[x]
    if x in {"step", "step_count"}:
        return frame[x], "Iterations"
    if x == "wall_time_sec":
        return frame[x] / 3600.0, "Wall-clock time (hours)"
    return frame[x], x.replace("_", " ").title()


def _x_filename(x: str) -> str:
    if x in {"step", "step_count"}:
        return "iterations"
    return x


def _metric_label(metric: str) -> str:
    return {
        "val_loss": "Validation loss",
        "val_ppl": "Validation perplexity",
        "train_loss": "Training loss",
        "final_val_loss": "Final validation loss",
        "final_val_ppl": "Final validation perplexity",
    }.get(metric, metric.replace("_", " ").title())


def _curve_label(row: pd.Series, *, include_seed: bool, include_result: bool) -> str:
    opt = str(row["optimizer"])
    mode = str(row["mode"])
    opt_name = _optimizer_display(opt)
    if mode == "optiselect":
        label = f"OptiSelect {opt_name}"
    else:
        label = opt_name
    suffix = []
    if include_seed and pd.notna(row.get("seed")):
        suffix.append(f"seed {int(row['seed'])}")
    if include_result and row.get("result_name"):
        suffix.append(str(row["result_name"]))
    if suffix:
        label += f" ({', '.join(suffix)})"
    return label


def _save(fig, out_dir: Path, name: str, formats: Sequence[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_validation_curves(
    curve_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    metric: str = "val_loss",
    x: str = "update_tokens",
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    if curve_df.empty:
        return []
    out_dir = Path(out_dir)
    optimizers = sorted(curve_df["optimizer"].unique(), key=_optimizer_sort_key)
    n = len(optimizers)
    ncols = min(3, max(1, n))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.55 * ncols, 2.65 * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, opt in zip(axes.ravel(), optimizers):
        sub_opt = curve_df[curve_df["optimizer"] == opt]
        for mode in ("standard", "optiselect"):
            sub = sub_opt[sub_opt["mode"] == mode].sort_values(x)
            if sub.empty:
                continue
            xx, xlabel = _scaled_x(sub, x)
            style = dict(MODE_STYLE[mode])
            if mode == "optiselect":
                style["color"] = OPTIMIZER_COLORS.get(opt, "#CC6677")
            ax.plot(xx, sub[metric], label=MODE_DISPLAY[mode], **style)
        ax.set_title(_optimizer_display(opt))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(_metric_label(metric))
        if metric == "val_ppl":
            ax.set_yscale("log")
        ax.margins(x=0.04)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    return _save(fig, out_dir, f"{metric}_vs_{_x_filename(x)}", formats)


def plot_validation_overlay(
    curve_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    metric: str = "val_loss",
    x: str = "update_tokens",
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    """Plot all optimizer curves on the same axes for easy ranking."""

    if curve_df.empty:
        return []
    out_dir = Path(out_dir)
    run_keys = (
        curve_df[["optimizer", "mode", "summary_path", "seed", "result_name"]]
        .drop_duplicates()
        .sort_values(
            ["optimizer", "mode", "seed", "result_name"],
            key=lambda col: col.map(_optimizer_sort_key) if col.name == "optimizer" else col,
        )
    )
    pair_counts = run_keys.groupby(["optimizer", "mode"]).size()
    include_seed = bool((pair_counts > 1).any())
    include_result = bool((pair_counts > 1).any() and run_keys["result_name"].nunique() > 1)

    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    ordered = sorted(
        run_keys.to_dict("records"),
        key=lambda row: (
            _optimizer_sort_key(str(row["optimizer"])),
            0 if row["mode"] == "standard" else 1,
            -1 if pd.isna(row.get("seed")) else int(row["seed"]),
            str(row.get("summary_path", "")),
        ),
    )
    for row in ordered:
        sub = curve_df[curve_df["summary_path"] == row["summary_path"]].sort_values(x)
        sub = sub[pd.notna(sub[metric]) & pd.notna(sub[x])]
        if sub.empty:
            continue
        xx, xlabel = _scaled_x(sub, x)
        opt = str(row["optimizer"])
        mode = str(row["mode"])
        style = dict(MODE_STYLE[mode])
        style["color"] = OPTIMIZER_COLORS.get(opt, "#777777")
        if mode == "standard":
            style["alpha"] = 0.82
            style["linewidth"] = 1.65
        else:
            style["linewidth"] = 2.15
        label = _curve_label(pd.Series(row), include_seed=include_seed, include_result=include_result)
        ax.plot(xx, sub[metric], label=label, markevery=max(len(sub) // 8, 1), **style)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(_metric_label(metric))
    if metric == "val_ppl":
        ax.set_yscale("log")
    ax.set_title(f"{_metric_label(metric)} ranking")
    ax.margins(x=0.03)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        handlelength=2.8,
    )
    fig.tight_layout()
    return _save(fig, out_dir, f"ranking_{metric}_vs_{_x_filename(x)}", formats)


def plot_train_curves(
    train_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    x: str = "update_tokens",
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    if train_df.empty:
        return []
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for opt in sorted(train_df["optimizer"].unique(), key=_optimizer_sort_key):
        sub_opt = train_df[train_df["optimizer"] == opt]
        for mode in ("standard", "optiselect"):
            sub = sub_opt[sub_opt["mode"] == mode].sort_values(x)
            if sub.empty:
                continue
            xx, xlabel = _scaled_x(sub, x)
            style = dict(MODE_STYLE[mode])
            style["linewidth"] = 1.2
            style["markersize"] = 0
            if mode == "optiselect":
                style["color"] = OPTIMIZER_COLORS.get(opt, "#CC6677")
            label = f"{_optimizer_display(opt)} {MODE_DISPLAY[mode]}"
            ax.plot(xx, sub["train_loss"], label=label, **style)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Training loss")
    ax.legend(ncol=2, loc="best")
    fig.tight_layout()
    return _save(fig, out_dir, f"train_loss_vs_{_x_filename(x)}", formats)


def plot_final_metric_bars(
    run_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    metric: str = "final_val_loss",
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    if run_df.empty:
        return []
    out_dir = Path(out_dir)
    optimizers = sorted(run_df["optimizer"].unique(), key=_optimizer_sort_key)
    x = np.arange(len(optimizers))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(5.4, 0.82 * len(optimizers) + 2.2), 3.4))
    for offset, mode in ((-width / 2, "standard"), (width / 2, "optiselect")):
        values = []
        errors = []
        colors = []
        for opt in optimizers:
            vals = pd.to_numeric(
                run_df[(run_df["optimizer"] == opt) & (run_df["mode"] == mode)][metric],
                errors="coerce",
            ).dropna()
            values.append(vals.mean() if len(vals) else np.nan)
            errors.append(vals.std(ddof=1) if len(vals) > 1 else 0)
            colors.append("#B8B8B8" if mode == "standard" else OPTIMIZER_COLORS.get(opt, "#CC6677"))
        ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            label=MODE_DISPLAY[mode],
            color=colors,
            edgecolor="black",
            linewidth=0.45,
            capsize=2.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_optimizer_display(o) for o in optimizers], rotation=25, ha="right")
    ax.set_ylabel(_metric_label(metric))
    if metric.endswith("_ppl"):
        ax.set_yscale("log")
    ax.legend(loc="best")
    fig.tight_layout()
    return _save(fig, out_dir, f"{metric}_bars", formats)


def plot_pair_ratios(
    pair_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    if pair_df.empty:
        return []
    out_dir = Path(out_dir)
    pair_df = pair_df.sort_values("optimizer", key=lambda s: s.map(_optimizer_sort_key))
    labels = list(pair_df["optimizer_display"])
    colors = [OPTIMIZER_COLORS.get(opt, "#777777") for opt in pair_df["optimizer"]]
    x = np.arange(len(pair_df))
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.35))

    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].bar(x, pair_df["delta_val_loss"], color=colors, edgecolor="black", linewidth=0.45)
    axes[0].set_ylabel("OptiSelect - Standard")
    axes[0].set_title("Final validation loss")

    ratio_cols = [
        ("update_token_ratio", "Update tokens"),
        ("processed_token_ratio", "Processed tokens"),
        ("wall_time_ratio", "Wall time"),
    ]
    width = 0.25
    for i, (col, label) in enumerate(ratio_cols):
        axes[1].bar(x + (i - 1) * width, pair_df[col], width, label=label, edgecolor="black", linewidth=0.35)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=0.9)
    axes[1].set_title("Resource ratios")
    axes[1].set_ylabel("OptiSelect / Standard")
    axes[1].legend(loc="best")

    axes[2].bar(x, pair_df["ppl_ratio"], color=colors, edgecolor="black", linewidth=0.45)
    axes[2].axhline(1.0, color="black", linestyle=":", linewidth=0.9)
    axes[2].set_title("Perplexity ratio")
    axes[2].set_ylabel("OptiSelect / Standard")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()
    return _save(fig, out_dir, "optiselect_vs_standard_ratios", formats)


def plot_resource_bars(
    run_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    if run_df.empty:
        return []
    out_dir = Path(out_dir)
    metrics = [
        ("actual_update_tokens", "Update tokens (B)", 1e9),
        ("actual_processed_tokens", "Processed tokens (B)", 1e9),
        ("wall_time_sec", "Wall time (h)", 3600.0),
        ("peak_vram_gb", "Peak VRAM (GB)", 1.0),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.2, 3.35))
    optimizers = sorted(run_df["optimizer"].unique(), key=_optimizer_sort_key)
    x = np.arange(len(optimizers))
    width = 0.36
    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        for offset, mode in ((-width / 2, "standard"), (width / 2, "optiselect")):
            values = []
            colors = []
            for opt in optimizers:
                vals = pd.to_numeric(
                    run_df[(run_df["optimizer"] == opt) & (run_df["mode"] == mode)][metric],
                    errors="coerce",
                ).dropna()
                values.append(vals.mean() / scale if len(vals) else np.nan)
                colors.append("#B8B8B8" if mode == "standard" else OPTIMIZER_COLORS.get(opt, "#CC6677"))
            ax.bar(x + offset, values, width, label=MODE_DISPLAY[mode], color=colors, edgecolor="black", linewidth=0.4)
        ax.set_title(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([_optimizer_display(o) for o in optimizers], rotation=25, ha="right")
    axes[0].legend(loc="best")
    fig.tight_layout()
    return _save(fig, out_dir, "resource_bars", formats)


def make_publication_plots(
    data: DemoPlotData,
    out_dir: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    include_train_curves: bool = True,
) -> list[Path]:
    """Write the standard publication figure set and return output paths."""

    configure_matplotlib()
    written = []
    written += save_tables(data, out_dir)
    written += plot_validation_curves(data.curve_df, out_dir, metric="val_loss", x="update_tokens", formats=formats)
    written += plot_validation_curves(data.curve_df, out_dir, metric="val_loss", x="processed_tokens", formats=formats)
    written += plot_validation_curves(data.curve_df, out_dir, metric="val_loss", x="step", formats=formats)
    written += plot_validation_curves(data.curve_df, out_dir, metric="val_ppl", x="update_tokens", formats=formats)
    written += plot_validation_overlay(data.curve_df, out_dir, metric="val_loss", x="update_tokens", formats=formats)
    written += plot_validation_overlay(data.curve_df, out_dir, metric="val_loss", x="processed_tokens", formats=formats)
    written += plot_validation_overlay(data.curve_df, out_dir, metric="val_loss", x="step", formats=formats)
    written += plot_validation_overlay(data.curve_df, out_dir, metric="val_ppl", x="update_tokens", formats=formats)
    written += plot_final_metric_bars(data.run_df, out_dir, metric="final_val_loss", formats=formats)
    written += plot_final_metric_bars(data.run_df, out_dir, metric="final_val_ppl", formats=formats)
    written += plot_pair_ratios(data.pair_df, out_dir, formats=formats)
    written += plot_resource_bars(data.run_df, out_dir, formats=formats)
    if include_train_curves:
        written += plot_train_curves(data.train_df, out_dir, x="update_tokens", formats=formats)
    return written


def load_lm_eval_metrics(result_dirs: Sequence[str | Path]) -> pd.DataFrame:
    """Load optional lm-evaluation-harness JSON files found near result dirs."""

    files: set[Path] = set()
    for raw in result_dirs:
        root = resolve_path(raw)
        if root.is_file() and root.name.startswith("lm_eval"):
            files.add(root.resolve())
            continue
        if not root.exists() or not root.is_dir():
            continue
        for path in root.glob("lm_eval*.json"):
            files.add(path.resolve())
        for path in root.glob("*/lm_eval*.json"):
            files.add(path.resolve())

    rows = []
    for path in sorted(files):
        try:
            blob = json.loads(path.read_text())
        except Exception:
            continue
        run_label = path.parent.name
        if path.parent.name.startswith("demo_"):
            run_label = path.stem.replace("lm_eval_harness_", "").replace("_results", "")
        for task, metrics in (blob.get("results") or {}).items():
            metric_name = None
            value = None
            stderr = None
            for name in PRIMARY_LM_EVAL_METRICS:
                if name in metrics and isinstance(metrics[name], (int, float)):
                    metric_name = name
                    value = float(metrics[name])
                    stderr_key = name.replace(",none", "_stderr,none")
                    stderr = metrics.get(stderr_key)
                    break
            if metric_name is None:
                continue
            rows.append(
                {
                    "file": str(path),
                    "run_label": run_label,
                    "task": task,
                    "metric": metric_name,
                    "value": value,
                    "stderr": stderr,
                    "higher_is_better": metric_name not in LOWER_IS_BETTER_METRICS,
                }
            )
    return pd.DataFrame(rows)


def plot_lm_eval_metrics(
    lm_eval_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    if lm_eval_df.empty:
        return []
    out_dir = Path(out_dir)
    df = lm_eval_df.copy()
    df["task_metric"] = df["task"] + "\n" + df["metric"].str.replace(",none", "", regex=False)
    tasks = list(dict.fromkeys(df["task_metric"]))
    labels = list(dict.fromkeys(df["run_label"]))
    x = np.arange(len(tasks))
    width = min(0.8 / max(len(labels), 1), 0.28)
    fig, ax = plt.subplots(figsize=(max(6.6, 0.55 * len(tasks) + 2.4), 3.8))
    for i, label in enumerate(labels):
        sub = df[df["run_label"] == label].set_index("task_metric")
        vals = [sub.loc[t, "value"] if t in sub.index else np.nan for t in tasks]
        vals = [float(v.iloc[0]) if hasattr(v, "iloc") else v for v in vals]
        ax.bar(x + (i - (len(labels) - 1) / 2) * width, vals, width, label=label, edgecolor="black", linewidth=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=35, ha="right")
    ax.set_ylabel("Metric value")
    ax.legend(loc="best")
    fig.tight_layout()
    return _save(fig, out_dir, "lm_eval_metrics", formats)
