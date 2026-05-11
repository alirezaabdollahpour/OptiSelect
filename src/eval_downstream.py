#!/usr/bin/env python3
# =============================================================================
# Downstream zero-shot evaluation for OptiSelect checkpoints.
#
# Tasks (all multiple-choice, scored by per-token loglikelihood):
#   - HellaSwag (Rowan/hellaswag)         : 4 endings, length-normalized LL
#   - ARC-Easy  (allenai/ai2_arc)         : multiple choice, normalized LL
#   - ARC-Chal. (allenai/ai2_arc)         : multiple choice, normalized LL
#   - PIQA      (lighteval/piqa)          : 2 solutions, length-normalized LL
#   - SciQ      (allenai/sciq)            : 4 choices, normalized LL
#   - SIQA      (lighteval/siqa)          : 3 answers,  length-normalized LL
#   - ANLI      (facebook/anli, test_r1)  : 3 verbalized labels (entail/neutral/contradict)
#   - MMLU      (cais/mmlu, all/validation): 4 letter labels (A/B/C/D)
#
# Loads checkpoints from benchmark run dirs. For each (mode, optimizer)
# row matching --dataset, it:
#   1. Reconstructs the Llama config from summary.json["args"]
#   2. Loads model weights from one of:
#        - <run>/final.pt/main.pt          (selection trainer)
#        - <run>/best.pt/main.pt           (selection trainer, --use-best)
#        - <run>/ckpts/latest/main.pt      (standard trainer w/ latest_ckpt_interval)
#   3. Runs all 5 tasks
#
# Usage:
#   python eval_downstream.py --results-dir /mloscratch/homes/aabdolla/results/wikitext_exp_v3_proxy_downstream
#   python eval_downstream.py --optimizers adamw,d-muon,sophiag --tasks piqa,hellaswag
# =============================================================================
import argparse
import glob
import json
import os
import re
import sys
import time
import types
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def setup_paths(src_dir: str):
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    ghost = "/mloscratch/homes/aabdolla/GhostSuite"
    if os.path.isdir(ghost) and ghost not in sys.path:
        sys.path.insert(0, ghost)
    os.environ.setdefault("HF_HOME", "/mloscratch/homes/aabdolla/.hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/mloscratch/homes/aabdolla/.hf_cache/datasets")


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def find_checkpoint(run_dir: str, prefer: str = "final") -> str:
    """Return path to main.pt for a given run directory.

    prefer ∈ {'final', 'best', 'latest'} controls which checkpoint to use
    when multiple are present.
    """
    candidates = []
    if prefer == "best":
        candidates.append(os.path.join(run_dir, "best.pt", "main.pt"))
    candidates.extend([
        os.path.join(run_dir, "final.pt", "main.pt"),
        os.path.join(run_dir, "ckpts", "latest", "main.pt"),
        os.path.join(run_dir, "best.pt", "main.pt"),
    ])
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_model(run_dir: str, device: torch.device, prefer: str = "final"):
    from models.utils import get_model
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    args = types.SimpleNamespace(**summary["args"])
    # The model constructor reads many fields directly from cfg; fill
    # any missing defaults that newer-than-checkpoint configs added.
    defaults = {"n_kv_head": None, "moe": False, "parallel_block": False,
                "use_pretrained": "none", "from_dense": False, "init_std": 0.02,
                "scale_emb": 1.0, "scale_base_model": 1.0, "scale_depth": 1.0,
                "moe_routing": None, "moe_num_experts": 0, "capacity_factor": 1.0,
                "moe_num_shared_experts": 0, "moe_router_loss": None,
                "moe_num_experts_per_tok": 1, "moe_entropy_loss_factor": 0.0,
                "moe_aux_loss_factor": 0.0, "moe_z_loss_factor": 0.0,
                "moe_softmax_order": "softmax-topk", "plot_router_logits": False,
                "untied_embeds": False}
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    model = get_model(args)
    ckpt_path = find_checkpoint(run_dir, prefer=prefer)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint under {run_dir}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"]
    # Strip DDP / torch.compile prefixes
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"    [warn] missing keys: {len(missing)}  (e.g. {missing[:2]})")
    if unexpected:
        print(f"    [warn] unexpected keys: {len(unexpected)}  (e.g. {unexpected[:2]})")
    model.eval().to(device)
    # Inference in fp32: the Llama RoPE buffer is fp32 and casting weights
    # to bf16 produces a query/key/value dtype mismatch in attention.
    # 25M is tiny enough that fp32 inference is cheap.
    return model, args, ckpt_path


# -----------------------------------------------------------------------------
# Tokenization (GPT-2 BPE, tiktoken)
# -----------------------------------------------------------------------------
def get_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def encode(tok, s: str) -> List[int]:
    return tok.encode_ordinary(s)


# -----------------------------------------------------------------------------
# Loglikelihood scoring
# -----------------------------------------------------------------------------
@torch.no_grad()
def loglikelihood_batch(model, ctx_lens: List[int], full_ids: List[List[int]],
                        max_len: int, device: torch.device) -> List[Tuple[float, int]]:
    """Score multiple (ctx, cont) pairs in a single forward.

    Returns list of (sum_logp, cont_len). Inputs are independently left-padded
    to max(len(full_ids[i])-1) and right-truncated to max_len-1 if too long.
    """
    if not full_ids:
        return []

    # Truncate so total length ≤ max_len; keep the continuation, drop early ctx.
    truncated = []
    new_ctx_lens = []
    for c_len, ids in zip(ctx_lens, full_ids):
        if len(ids) > max_len:
            drop = len(ids) - max_len
            ids = ids[drop:]
            c_len = max(1, c_len - drop)
        truncated.append(ids)
        new_ctx_lens.append(c_len)

    # Each example's input = ids[:-1], target = ids[1:]; pad to common length.
    in_lens = [len(x) - 1 for x in truncated]
    L = max(in_lens)
    B = len(truncated)
    inp = torch.zeros((B, L), dtype=torch.long, device=device)
    tgt = torch.full((B, L), -1, dtype=torch.long, device=device)
    for i, ids in enumerate(truncated):
        inp[i, : len(ids) - 1] = torch.tensor(ids[:-1], dtype=torch.long, device=device)
        tgt[i, : len(ids) - 1] = torch.tensor(ids[1:], dtype=torch.long, device=device)

    # Pass `tgt` as targets so the model returns full-sequence logits (it
    # otherwise short-circuits to last-token-only at inference time).
    out = model(inp, targets=tgt.clamp(min=0), get_logits=True)
    logits = out["logits"] if isinstance(out, dict) else out[0]
    logp = F.log_softmax(logits.float(), dim=-1)

    results = []
    for i, ids in enumerate(truncated):
        ctx_len = new_ctx_lens[i]
        cont_start = max(0, ctx_len - 1)
        cont_end = len(ids) - 1
        if cont_end <= cont_start:
            results.append((0.0, 0))
            continue
        sel_logp = logp[i, cont_start:cont_end, :]
        sel_tgt = tgt[i, cont_start:cont_end]
        valid = sel_tgt >= 0
        if not valid.any():
            results.append((0.0, 0))
            continue
        gathered = sel_logp.gather(1, sel_tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        ll = gathered[valid].sum().item()
        results.append((ll, int(valid.sum().item())))
    return results


def score_choices(model, tok, ctx: str, choices: List[str],
                  max_len: int, device: torch.device,
                  normalize: bool = True) -> int:
    """Return argmax choice index by per-token-normalized loglikelihood."""
    ctx_ids = encode(tok, ctx)
    full_ids, ctx_lens = [], []
    for ch in choices:
        cont_ids = encode(tok, ch)
        full = ctx_ids + cont_ids
        full_ids.append(full)
        ctx_lens.append(len(ctx_ids))
    scores = loglikelihood_batch(model, ctx_lens, full_ids, max_len, device)
    if normalize:
        out = [ll / max(1, n) for ll, n in scores]
    else:
        out = [ll for ll, _ in scores]
    return int(np.argmax(out))


# -----------------------------------------------------------------------------
# Task-specific runners
# -----------------------------------------------------------------------------
def task_hellaswag(model, tok, max_len, device, max_examples=None):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    correct, total = 0, 0
    for ex in ds:
        ctx = ex["ctx"].strip()
        choices = [" " + e.strip() for e in ex["endings"]]
        gold = int(ex["label"])
        pred = score_choices(model, tok, ctx, choices, max_len, device, normalize=True)
        correct += int(pred == gold)
        total += 1
    return correct / total, total


def task_piqa(model, tok, max_len, device, max_examples=None):
    from datasets import load_dataset
    ds = load_dataset("lighteval/piqa", split="validation")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    correct, total = 0, 0
    for ex in ds:
        ctx = "Question: " + ex["goal"].strip() + "\nAnswer:"
        choices = [" " + ex["sol1"].strip(), " " + ex["sol2"].strip()]
        gold = int(ex["label"])
        pred = score_choices(model, tok, ctx, choices, max_len, device, normalize=True)
        correct += int(pred == gold)
        total += 1
    return correct / total, total


def _arc_task(model, tok, max_len, device, config_name, max_examples=None):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", config_name, split="test")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    correct, total = 0, 0
    for ex in ds:
        labels = list(ex["choices"]["label"])
        texts = list(ex["choices"]["text"])
        answer = str(ex["answerKey"])
        if answer not in labels:
            continue
        ctx = "Question: " + ex["question"].strip() + "\nAnswer:"
        choices = [" " + t.strip() for t in texts]
        gold = labels.index(answer)
        pred = score_choices(model, tok, ctx, choices, max_len, device, normalize=True)
        correct += int(pred == gold)
        total += 1
    return correct / max(1, total), total


def task_arc_easy(model, tok, max_len, device, max_examples=None):
    return _arc_task(model, tok, max_len, device, "ARC-Easy", max_examples)


def task_arc_challenge(model, tok, max_len, device, max_examples=None):
    return _arc_task(model, tok, max_len, device, "ARC-Challenge", max_examples)


def task_sciq(model, tok, max_len, device, max_examples=None):
    from datasets import load_dataset
    ds = load_dataset("sciq", split="test")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    correct, total = 0, 0
    rng = np.random.default_rng(12345)
    for ex in ds:
        choices = [
            ex["correct_answer"].strip(),
            ex["distractor1"].strip(),
            ex["distractor2"].strip(),
            ex["distractor3"].strip(),
        ]
        order = list(rng.permutation(4))
        shuffled = [choices[i] for i in order]
        gold = order.index(0)
        ctx = "Question: " + ex["question"].strip() + "\nAnswer:"
        pred = score_choices(
            model, tok, ctx, [" " + c for c in shuffled], max_len, device,
            normalize=True,
        )
        correct += int(pred == gold)
        total += 1
    return correct / total, total


def task_siqa(model, tok, max_len, device, max_examples=None):
    from datasets import load_dataset
    ds = load_dataset("lighteval/siqa", split="validation")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    correct, total = 0, 0
    for ex in ds:
        ctx = (ex["context"].strip() + "\nQuestion: " + ex["question"].strip() + "\nAnswer:")
        choices = [" " + ex[k].strip() for k in ("answerA", "answerB", "answerC")]
        # SIQA labels are 1/2/3 strings.
        gold = int(ex["label"]) - 1
        pred = score_choices(model, tok, ctx, choices, max_len, device, normalize=True)
        correct += int(pred == gold)
        total += 1
    return correct / total, total


def task_anli(model, tok, max_len, device, max_examples=None, split="test_r1"):
    from datasets import load_dataset
    ds = load_dataset("facebook/anli", split=split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    correct, total = 0, 0
    # ANLI label: 0=entailment, 1=neutral, 2=contradiction
    verbalizations = [" True", " Neither", " False"]
    for ex in ds:
        ctx = (f"{ex['premise'].strip()}\nQuestion: {ex['hypothesis'].strip()} "
               "True, False, or Neither?\nAnswer:")
        gold = int(ex["label"])
        # Single short labels — raw LL is fine, no length norm.
        pred = score_choices(model, tok, ctx, verbalizations, max_len, device, normalize=False)
        correct += int(pred == gold)
        total += 1
    return correct / total, total


def task_mmlu(model, tok, max_len, device, max_examples=None):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="validation")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    correct, total = 0, 0
    letters = [" A", " B", " C", " D"]
    for ex in ds:
        q = ex["question"].strip()
        c = ex["choices"]
        ctx = (f"Question: {q}\n"
               f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:")
        gold = int(ex["answer"])
        pred = score_choices(model, tok, ctx, letters, max_len, device, normalize=False)
        correct += int(pred == gold)
        total += 1
    return correct / total, total


TASKS = {
    "hellaswag": task_hellaswag,
    "arc_easy":  task_arc_easy,
    "arc_challenge": task_arc_challenge,
    "piqa":      task_piqa,
    "sciq":      task_sciq,
    "siqa":      task_siqa,
    "anli":      task_anli,
    "mmlu":      task_mmlu,
}


# -----------------------------------------------------------------------------
def discover_runs(results_dir: str, dataset: str, modes: List[str], optimizers: List[str]):
    """Yield (mode, optimizer, run_dir) tuples that contain a usable checkpoint."""
    pat = re.compile(r"^(?P<mode>standard|selection)_" + re.escape(dataset) +
                     r"(?:_proxy-[^_]+)?_(?P<opt>.+?)_seed\d+$")
    for entry in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        m = pat.match(entry)
        if not m:
            continue
        mode, opt = m.group("mode"), m.group("opt")
        if modes and mode not in modes:
            continue
        if optimizers and opt not in optimizers:
            continue
        if find_checkpoint(run_dir) is None:
            continue
        yield mode, opt, run_dir


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="/mloscratch/homes/aabdolla/results/wikitext_exp_v3_proxy_downstream")
    p.add_argument("--out", default=None,
                   help="JSON output path (default: <results-dir>/downstream_eval.json)")
    p.add_argument("--dataset", default="wikitext",
                   help="Run-name dataset prefix, e.g. wikitext or fineweb100b_720m.")
    p.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge,piqa,sciq")
    p.add_argument("--modes", default="standard,selection")
    p.add_argument("--optimizers", default=None,
                   help="Comma list to restrict (default: all 10)")
    p.add_argument("--prefer", default="final", choices=["final", "best", "latest"])
    p.add_argument("--max-examples", type=int, default=None,
                   help="For debugging; cap items per task")
    p.add_argument("--src-dir", default="/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    setup_paths(args.src_dir)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Device: {device}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in TASKS:
            raise SystemExit(f"Unknown task '{t}'; valid: {list(TASKS)}")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    optimizers = ([o.strip() for o in args.optimizers.split(",")]
                  if args.optimizers else None)

    runs = list(discover_runs(args.results_dir, args.dataset, modes, optimizers))
    if not runs:
        raise SystemExit(f"No runs with checkpoints found under {args.results_dir} "
                         f"(modes={modes}, optimizers={optimizers}).")

    # Group printout
    by_mode = {}
    for mode, opt, _ in runs:
        by_mode.setdefault(mode, []).append(opt)
    for m, opts in sorted(by_mode.items()):
        print(f"  {m}: {len(opts)} runs ({', '.join(sorted(opts))})")

    out_path = args.out or os.path.join(args.results_dir, "downstream_eval.json")
    # Resume-friendly: load existing results if present.
    if os.path.exists(out_path):
        all_results = json.load(open(out_path))
    else:
        all_results = {}

    tok = get_tokenizer()

    for mode, opt, run_dir in runs:
        key = f"{mode}/{opt}"
        existing = all_results.get(key, {})
        pending = [t for t in tasks if t not in existing]
        if not pending:
            print(f"[skip] {key} — all tasks already evaluated.")
            continue
        print(f"\n[{key}]  loading {os.path.basename(run_dir)}")
        try:
            model, cfg, ckpt_path = load_model(run_dir, device, prefer=args.prefer)
        except Exception as e:
            print(f"  [error] failed to load: {e}")
            existing["_error"] = str(e)
            all_results[key] = existing
            continue
        max_len = int(cfg.sequence_length)
        param_count = sum(p.numel() for p in model.parameters())
        world_size = int(getattr(cfg, "world_size", 1))
        update_tokens = (
            int(cfg.iterations)
            * int(cfg.batch_size)
            * int(cfg.acc_steps)
            * world_size
            * int(cfg.sequence_length)
        )
        processed_factor = (
            1 + int(getattr(cfg, "candidate_multiplier", 2))
            if mode == "selection" else 1
        )
        processed_tokens = update_tokens * processed_factor
        existing.setdefault("_meta", {})
        existing["_meta"].update({
            "checkpoint": os.path.relpath(ckpt_path, args.results_dir),
            "n_embd": cfg.n_embd, "n_head": cfg.n_head, "n_layer": cfg.n_layer,
            "sequence_length": max_len,
            "param_count": param_count,
            "update_tokens": update_tokens,
            "update_tokens_B": update_tokens / 1e9,
            "processed_tokens": processed_tokens,
            "processed_tokens_B": processed_tokens / 1e9,
            "estimated_training_flops": 6 * param_count * processed_tokens,
            "estimated_training_flops_EF": 6 * param_count * processed_tokens / 1e18,
        })

        for tname in pending:
            t0 = time.time()
            try:
                acc, n = TASKS[tname](model, tok, max_len, device,
                                      max_examples=args.max_examples)
                dt = time.time() - t0
                existing[tname] = {"acc": acc, "n": n, "seconds": round(dt, 1)}
                print(f"  {tname:>10s}: acc={100*acc:.2f}%  (n={n}, {dt:.1f}s)")
            except Exception as e:
                existing[tname] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  {tname:>10s}: ERROR — {e}")

        all_results[key] = existing

        # Persist after every model so partial progress survives crashes.
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Clean up between models to keep VRAM low.
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nSaved: {out_path}")

    # ---- Comparison table (standard vs selection per optimizer) -------------
    print("\n" + "=" * 90)
    print(f"  Downstream zero-shot accuracy (%) — {os.path.basename(args.results_dir)}")
    print("=" * 90)
    by_opt = {}
    for key, row in all_results.items():
        if "/" not in key:
            continue
        mode, opt = key.split("/", 1)
        by_opt.setdefault(opt, {})[mode] = row
    hdr = f"{'Optimizer':<12} | " + " | ".join(
        f"{t:>9s} std/sel/Δ" for t in tasks
    )
    print(hdr)
    print("-" * len(hdr))
    for opt in sorted(by_opt):
        row = by_opt[opt]
        cells = []
        for t in tasks:
            s = (row.get("standard")  or {}).get(t, {})
            x = (row.get("selection") or {}).get(t, {})
            sa = 100 * s["acc"] if isinstance(s, dict) and "acc" in s else None
            xa = 100 * x["acc"] if isinstance(x, dict) and "acc" in x else None
            sa_s = f"{sa:5.1f}" if sa is not None else "  -- "
            xa_s = f"{xa:5.1f}" if xa is not None else "  -- "
            d_s = f"{xa - sa:+5.1f}" if (sa is not None and xa is not None) else "  -- "
            cells.append(f"{sa_s}/{xa_s}/{d_s}")
        print(f"{opt:<12} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
