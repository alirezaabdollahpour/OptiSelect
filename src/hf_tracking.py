#!/usr/bin/env python3
"""Hugging Face Hub tracking helpers for optimizer benchmark runs.

This uploads lightweight experiment artifacts to a Hub repository:
summary.json, logs, curve/downstream JSON files, and a small metadata card.
Checkpoint upload is optional because 720M x 20 runs can be very large.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from huggingface_hub.utils import HfHubHTTPError


def _bool(s: str | bool | None) -> bool:
    if isinstance(s, bool):
        return s
    if s is None:
        return False
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _safe_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def ensure_repo(api: HfApi, repo_id: str, repo_type: str, private: bool) -> None:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except HfHubHTTPError as exc:
        raise SystemExit(
            f"Could not access Hugging Face repo {repo_id!r} "
            f"(repo_type={repo_type}). Check HF_TOKEN/HF_REPO_ID. Error: {exc}"
        ) from exc


def build_run_card(
    out_dir: Path,
    *,
    dataset: str,
    model_size: str,
    owner: str,
    contact_email: str,
    split: str,
    seed: str,
    results_dir: Path,
    curves_file: Path | None,
    downstream_file: Path | None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    card = out_dir / "README.md"
    timestamp = datetime.now(timezone.utc).isoformat()
    lines = [
        "---",
        "license: other",
        "tags:",
        "- optimizer-benchmark",
        "- optiselect",
        "- fineweb",
        "- language-modeling",
        "---",
        "",
        f"# {dataset} {model_size} Optimizer Benchmark",
        "",
        f"Uploaded at: `{timestamp}`",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Dataset | `{dataset}` |",
        f"| Model size | `{model_size}` |",
        f"| Owner | `{owner}` |",
        f"| Contact | `{contact_email}` |",
        f"| Split | `{split}` |",
        f"| Seed | `{seed}` |",
        f"| Local results dir | `{results_dir}` |",
        "",
        "Artifacts in this folder are intended for experiment tracking and paper plots.",
        "Large checkpoints are uploaded only when `HF_UPLOAD_CHECKPOINTS=1`.",
    ]
    if curves_file:
        lines.append(f"\nValidation/FLOP curves: `{curves_file.name}`")
    if downstream_file:
        lines.append(f"\nDownstream evaluation: `{downstream_file.name}`")
    card.write_text("\n".join(lines) + "\n")
    return card


def iter_lightweight_files(run_dir: Path) -> Iterable[Path]:
    for rel in ("summary.json",):
        p = run_dir / rel
        if p.exists():
            yield p


def upload_one_file(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str,
    local_path: Path,
    path_in_repo: str,
    commit_message: str,
) -> None:
    if not local_path.exists():
        return
    upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    p.add_argument("--private", default="1")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--src-dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--dataset", default="fineweb100b")
    p.add_argument("--model-size", default="720m")
    p.add_argument("--owner", default="AlirezaAbdollahpoorrostam")
    p.add_argument("--contact-email", default="alireza.abdollahpoorrostam@epfl.ch")
    p.add_argument("--proxy-source", default="train")
    p.add_argument("--optimizers", default="")
    p.add_argument("--upload-checkpoints", default="0")
    p.add_argument("--curves-file", default="")
    p.add_argument("--downstream-file", default="")
    args = p.parse_args()

    repo_id = args.repo_id
    repo_type = args.repo_type
    private = _bool(args.private)
    upload_checkpoints = _bool(args.upload_checkpoints)
    results_dir = Path(args.results_dir).resolve()
    src_dir = Path(args.src_dir).resolve()
    logs_dir = src_dir / "logs"
    curves_file = Path(args.curves_file).resolve() if args.curves_file else None
    downstream_file = Path(args.downstream_file).resolve() if args.downstream_file else None

    api = HfApi()
    ensure_repo(api, repo_id, repo_type, private)

    run_prefix = f"{args.dataset}/{args.model_size}/seed{args.seed}/split{args.split}"
    if args.proxy_source != "train":
        run_prefix += f"/proxy-{_safe_name(args.proxy_source)}"

    tmp_dir = results_dir / ".hf_tracking"
    card = build_run_card(
        tmp_dir,
        dataset=args.dataset,
        model_size=args.model_size,
        owner=args.owner,
        contact_email=args.contact_email,
        split=args.split,
        seed=args.seed,
        results_dir=results_dir,
        curves_file=curves_file,
        downstream_file=downstream_file,
    )
    upload_one_file(
        api,
        repo_id=repo_id,
        repo_type=repo_type,
        local_path=card,
        path_in_repo=f"{run_prefix}/README.md",
        commit_message=f"Update {args.dataset} {args.model_size} split {args.split} tracking card",
    )

    if curves_file and curves_file.exists():
        upload_one_file(
            api,
            repo_id=repo_id,
            repo_type=repo_type,
            local_path=curves_file,
            path_in_repo=f"{run_prefix}/{curves_file.name}",
            commit_message=f"Upload curves for {args.dataset} {args.model_size} split {args.split}",
        )

    if downstream_file and downstream_file.exists():
        upload_one_file(
            api,
            repo_id=repo_id,
            repo_type=repo_type,
            local_path=downstream_file,
            path_in_repo=f"{run_prefix}/{downstream_file.name}",
            commit_message=f"Upload downstream eval for {args.dataset} {args.model_size} split {args.split}",
        )

    optimizers = [o.strip() for o in args.optimizers.split(",") if o.strip()]
    modes = ["standard", "selection"]
    proxy_tag = "" if args.proxy_source == "train" else f"proxy-{args.proxy_source}_"
    uploaded = 0
    for opt in optimizers:
        for mode in modes:
            run_name = f"{mode}_{args.dataset}_{args.model_size}_{proxy_tag}{opt}_seed{args.seed}"
            run_dir = results_dir / run_name
            if not run_dir.is_dir():
                continue
            summary = _read_json(run_dir / "summary.json")
            opt_dir = f"{run_prefix}/runs/{mode}/{_safe_name(opt)}"

            for local_file in iter_lightweight_files(run_dir):
                upload_one_file(
                    api,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    local_path=local_file,
                    path_in_repo=f"{opt_dir}/{local_file.name}",
                    commit_message=f"Upload {run_name} summary",
                )
                uploaded += 1

            log_file = logs_dir / f"{run_name}.log"
            if log_file.exists():
                upload_one_file(
                    api,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    local_path=log_file,
                    path_in_repo=f"{opt_dir}/{log_file.name}",
                    commit_message=f"Upload {run_name} log",
                )
                uploaded += 1

            if upload_checkpoints:
                for ckpt_name in ("final.pt", "ckpts/latest", "best.pt"):
                    ckpt_dir = run_dir / ckpt_name
                    if not ckpt_dir.is_dir():
                        continue
                    upload_folder(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        folder_path=str(ckpt_dir),
                        path_in_repo=f"{opt_dir}/{ckpt_name}",
                        commit_message=f"Upload {run_name} checkpoint {ckpt_name}",
                    )
                    uploaded += 1

            metrics_file = tmp_dir / f"{run_name}_metrics.json"
            metrics = {
                "run_name": run_name,
                "mode": mode,
                "optimizer": opt,
                "seed": int(args.seed),
                "dataset": args.dataset,
                "model_size": args.model_size,
                "owner": args.owner,
                "contact_email": args.contact_email,
                "proxy_source": args.proxy_source,
                "summary": {
                    k: summary.get(k)
                    for k in ("final_val_loss", "final_val_pp", "final_val_acc", "best_val_loss")
                    if k in summary
                },
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            metrics_file.write_text(json.dumps(metrics, indent=2) + "\n")
            upload_one_file(
                api,
                repo_id=repo_id,
                repo_type=repo_type,
                local_path=metrics_file,
                path_in_repo=f"{opt_dir}/hf_metrics.json",
                commit_message=f"Upload {run_name} metrics",
            )
            uploaded += 1

    print(
        f"[HF] Uploaded tracking artifacts to https://huggingface.co/"
        f"{'datasets/' if repo_type == 'dataset' else ''}{repo_id}/tree/main/{run_prefix}"
    )
    print(f"[HF] Uploaded/updated {uploaded} run artifact(s)")


if __name__ == "__main__":
    main()
