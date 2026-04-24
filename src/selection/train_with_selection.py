"""
Training loop with online in-batch data selection (OPUS-style).

Implements the OPUS pipeline from Paper Section 3:
  1. Draw candidate batch B̃ of size 2B by concatenating
     `candidate_multiplier` fresh mini-batches of size B
  2. Forward-backward on candidates -> capture Ghost factors
  3. Score candidates using optimizer-specific frozen-state operator
  4. Greedy Boltzmann selection with redundancy penalty (Paper Eq. 4)
  5. Standard optimizer step on selected B samples

Key robustness feature: this loop NEVER mutates DataReader.batch_size.
Instead, it draws `candidate_multiplier` separate batches of the reader's
native batch_size and concatenates them. This guarantees a consistent
B_cand across iterations regardless of DataReader internals.

Paper references:
  Section 3:    Online selection paradigm
  Section 4.2:  Optimizer-aware scoring
  Appendix C:   4,096-doc proxy refreshed every 5,000 steps; τ=0.1; λ_r=1.0
"""

import os
import time
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from data.utils import get_dataset, DataReader
from optim.base import get_batch
from optim.utils import eval as eval_fn, save_checkpoint

from selection.optiselect_engine import OptiSelectEngine
from selection.influence_scoring import InfluenceConfig


def _draw_candidate_batch(
    train_reader,
    candidate_multiplier: int,
    device: str,
):
    """
    Draw a candidate batch of size B̃ = candidate_multiplier × B by
    concatenating `candidate_multiplier` separate mini-batches of size B.

    This avoids mutating train_reader.batch_size, which is unreliable
    across DataReader implementations.
    """
    xs, ys = [], []
    for _ in range(candidate_multiplier):
        x, y = get_batch(train_reader, device=device)
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train_with_selection(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    cfg,
    distributed_backend=None,
    **kwargs,
):
    """
    Training loop with online in-batch data selection.

    Args:
        model:                transformer model
        opt:                  optimizer (already constructed)
        datareaders:          {"train": DataReader, "val": DataReader}
        scheduler:            LR scheduler
        exp_dir:              output directory
        cfg:                  config namespace with selection attributes
        distributed_backend:  unused in single-GPU path (accepted for main.py API)
    """
    # ------------------------------------------------------------------
    # Configuration (Paper Appendix C defaults)
    # ------------------------------------------------------------------
    candidate_multiplier = getattr(cfg, "candidate_multiplier", 2)
    temperature = getattr(cfg, "selection_temperature", 0.1)
    sketch_dim = getattr(cfg, "selection_sketch_dim", 1024)
    redundancy_weight = getattr(cfg, "selection_redundancy_weight", 1.0)
    val_proxy_size = getattr(cfg, "val_proxy_size", 4096)
    val_proxy_refresh = getattr(cfg, "val_proxy_refresh", 5000)
    val_proxy_source = getattr(cfg, "val_proxy_source", "train")
    val_proxy_tasks = [
        t.strip()
        for t in getattr(
            cfg,
            "val_proxy_tasks",
            "hellaswag,arc_easy,arc_challenge,piqa,sciq",
        ).split(",")
        if t.strip()
    ]

    B = cfg.batch_size
    B_cand = B * candidate_multiplier
    B_select = B

    device = cfg.device
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    type_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device.startswith("cuda")
        else nullcontext()
    )

    # ------------------------------------------------------------------
    # DataReaders: use as-is, no batch_size mutation
    # ------------------------------------------------------------------
    train_reader = datareaders["train"]
    val_reader = datareaders["val"]

    # ------------------------------------------------------------------
    # Build validation proxy (Paper Appendix C: 4,096 documents by default,
    # drawn from the training dataset's val split). When
    # cfg.val_proxy_source == "downstream", the proxy is instead drawn from
    # a mixture of held-out multiple-choice benchmark tasks (HellaSwag,
    # ARC-Easy, ARC-Challenge, PIQA, SciQ) — a task-aware alignment signal.
    # ------------------------------------------------------------------
    proxy_refresh_count = 0
    base_seed = int(getattr(cfg, "seed", 0))

    if val_proxy_source == "downstream":
        from selection.downstream_proxy import build_downstream_proxy_batches

    def build_val_proxy():
        """Draw a fresh val_proxy_size-document proxy set."""
        nonlocal proxy_refresh_count
        if val_proxy_source == "train":
            val_reader.set_step(0)
            reader_bs = val_reader.batch_size
            n_batches = max(1, (val_proxy_size + reader_bs - 1) // reader_bs)
            batches = []
            for _ in range(n_batches):
                vx, vy = get_batch(val_reader, device=device)
                batches.append((vx, vy))
            proxy_refresh_count += 1
            return batches

        # Downstream task-mixture proxy
        rng = np.random.default_rng(
            base_seed + 10_000 * (proxy_refresh_count + 1)
        )
        batches, counts = build_downstream_proxy_batches(
            tasks=val_proxy_tasks,
            n_docs=val_proxy_size,
            batch_size=val_reader.batch_size,
            sequence_length=cfg.sequence_length,
            device=device,
            rng=rng,
            vocab_size=int(getattr(cfg, "vocab_size", 50304)),
        )
        per_task_str = ", ".join(f"{t}:{n}" for t, n in counts.items())
        print(f"[OptiSelect] Downstream proxy composition — {per_task_str}")
        proxy_refresh_count += 1
        return batches

    val_proxy_batches = build_val_proxy()
    actual_proxy_docs = sum(b[0].size(0) for b in val_proxy_batches)
    print(
        f"[OptiSelect] Built validation proxy: "
        f"{len(val_proxy_batches)} batches × "
        f"{val_reader.batch_size} docs = "
        f"{actual_proxy_docs} documents"
    )

    # ------------------------------------------------------------------
    # Initialize OptiSelect engine
    # ------------------------------------------------------------------
    sel_config = InfluenceConfig(
        sketch_dim=sketch_dim,
        temperature=temperature,
        redundancy_weight=redundancy_weight,
        use_countsketch=False,
    )

    engine = OptiSelectEngine(
        model=model,
        optimizer=opt,
        opt_name=cfg.opt,
        config=sel_config,
        device=device,
    )

    print(f"[OptiSelect] Computing initial validation gradient factors...")
    engine.compute_validation_gradient_factors(model, val_proxy_batches, type_ctx)

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    log_file = os.path.join(exp_dir, "summary.json")
    history: List[Dict] = []
    best_val_loss = float("inf")

    print(f"\n[OptiSelect] *** Selection mode enabled ***")
    print(f"  Optimizer:            {cfg.opt}")
    print(f"  Candidate multiplier: {candidate_multiplier} (B̃={B_cand}, B={B_select})")
    print(f"  Temperature τ:        {temperature}")
    print(f"  Redundancy weight:    {redundancy_weight}")
    print(f"  Val proxy size:       {val_proxy_size} (actual: {actual_proxy_docs})")
    print(f"  Val proxy refresh:    every {val_proxy_refresh} steps")
    print(f"  Val proxy source:     {val_proxy_source}"
          + (f" [{', '.join(val_proxy_tasks)}]"
             if val_proxy_source == "downstream" else ""))
    print(f"  Strategy:             Concatenate {candidate_multiplier} "
          f"batches of {B} (no batch_size mutation)\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    step_times: List[float] = []

    for curr_iter in range(cfg.iterations):
        t_start = time.time()

        # ---- Refresh validation proxy periodically ----
        if curr_iter > 0 and curr_iter % val_proxy_refresh == 0:
            print(
                f"[Iter {curr_iter}] Refreshing validation proxy "
                f"({val_proxy_size} docs)"
            )
            val_proxy_batches = build_val_proxy()
            engine.compute_validation_gradient_factors(
                model, val_proxy_batches, type_ctx
            )

        # ==============================================================
        #  1. Draw candidate batch B̃ = candidate_multiplier × B
        # ==============================================================
        cand_x, cand_y = _draw_candidate_batch(
            train_reader, candidate_multiplier, device
        )

        # Sanity check: verify we got the expected batch size
        actual_B_cand = cand_x.size(0)
        if actual_B_cand == 0:
            print(f"[ERROR] Iter {curr_iter}: candidate batch is empty, skipping")
            continue
        if curr_iter == 0:
            print(
                f"[OptiSelect] Candidate batch shape verified: "
                f"{tuple(cand_x.shape)} (expected B̃≈{B_cand})"
            )

        # ==============================================================
        #  2. Forward-backward on candidates -> Ghost factors
        # ==============================================================
        engine.start_capture()
        opt.zero_grad()
        with type_ctx:
            outputs = model(cand_x, targets=cand_y)
        outputs["loss"].backward()
        engine.stop_capture()

        # Hand over Ghost factors from the engine. The hooks will re-populate
        # fresh dicts on the next capture pass, so a clone here just doubles
        # peak bf16 memory for no reason.
        candidate_activations = engine._layer_activations
        candidate_backprops = engine._layer_backprops
        engine._layer_activations = {}
        engine._layer_backprops = {}

        # Verify candidates captured (should equal actual_B_cand in dim 0)
        if len(candidate_activations) == 0:
            print(f"[ERROR] Iter {curr_iter}: no Ghost factors captured, skipping")
            continue
        any_key = next(iter(candidate_activations))
        if candidate_activations[any_key].size(0) != actual_B_cand:
            print(
                f"[WARN] Iter {curr_iter}: Ghost factor size mismatch "
                f"({candidate_activations[any_key].size(0)} vs {actual_B_cand})"
            )

        # Paper Remark 4: MARS raw-gradient variance update
        if cfg.opt == "mars":
            engine.update_mars_scoring_v(beta2=getattr(cfg, "beta2", 0.999))

        # ==============================================================
        #  3. Score candidates
        # ==============================================================
        with torch.no_grad():
            alignment_scores = engine.score_candidates(
                candidate_activations, candidate_backprops
            )

        # ==============================================================
        #  4. Greedy Boltzmann selection with redundancy (Paper Eq. 4)
        #     Cap n_select to actual_B_cand - 1 to always leave at least
        #     one candidate in the pool (no-op for normal B_cand=2B case)
        # ==============================================================
        n_select = min(B_select, actual_B_cand)
        if n_select <= 0:
            print(f"[ERROR] Iter {curr_iter}: cannot select from "
                  f"B_cand={actual_B_cand}, skipping")
            continue

        with torch.no_grad():
            selected_idx = engine.select_batch_with_redundancy(
                candidate_activations,
                candidate_backprops,
                alignment_scores=alignment_scores,
                n_select=n_select,
                eta=cfg.lr,
                lambda_r=redundancy_weight,
            )

        # ==============================================================
        #  5. Recompute loss on selected samples, optimizer step
        # ==============================================================
        sel_x = cand_x[selected_idx]
        sel_y = cand_y[selected_idx]

        opt.zero_grad()
        with type_ctx:
            sel_outputs = model(sel_x, targets=sel_y)
        loss = sel_outputs["loss"]
        loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        opt.step()

        # Paper Section 4.3.1: ADOPT lagged variance cache
        if cfg.opt == "adopt":
            engine.cache_adopt_prev_v()

        if scheduler is not None:
            scheduler.step()

        step_times.append(time.time() - t_start)

        # ==============================================================
        #  Logging
        # ==============================================================
        if curr_iter % cfg.log_interval == 0:
            sel_summary = engine.get_selection_summary()
            current_lr = scheduler.get_last_lr()[0] if scheduler else cfg.lr
            entropy = sel_summary.get("mean_entropy", float("nan"))
            print(
                f"Train: Iter={curr_iter} "
                f"train_loss={loss.item():.4f} "
                f"iter_dt={step_times[-1]:.2f}s "
                f"lr={current_lr:.2e} "
                f"sel_entropy={entropy:.2f}"
            )

        # ==============================================================
        #  Evaluation
        # ==============================================================
        if curr_iter > 0 and curr_iter % cfg.eval_interval == 0:
            model.eval()
            val_acc, val_loss, val_pp, _, _ = eval_fn(
                model=model,
                reader=val_reader,
                device=device,
                max_num_batches=cfg.eval_batches,
                ctx=type_ctx,
                cfg=cfg,
            )
            model.train()

            print(
                f">Eval: Iter={curr_iter} "
                f"val_loss={val_loss:.4f} "
                f"val_pp={val_pp:.3f} "
                f"val_acc={val_acc:.6f}"
            )

            history.append({
                "iter": curr_iter,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
                "val_pp": float(val_pp),
                "val_acc": float(val_acc),
                "entropy": engine.selection_stats["entropy"][-1]
                if engine.selection_stats["entropy"]
                else None,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if distributed_backend is None or distributed_backend.is_master_process():
                    save_checkpoint(
                        model, opt, scheduler, curr_iter,
                        os.path.join(exp_dir, "best.pt"),
                    )

            with open(log_file, "w") as f:
                json.dump({
                    "history": history,
                    "best_val_loss": float(best_val_loss),
                    "selection_summary": engine.get_selection_summary(),
                    "config": {
                        "opt": cfg.opt,
                        "lr": cfg.lr,
                        "iterations": cfg.iterations,
                        "batch_size": B,
                        "candidate_multiplier": candidate_multiplier,
                        "temperature": temperature,
                        "val_proxy_size": val_proxy_size,
                        "val_proxy_refresh": val_proxy_refresh,
                        "val_proxy_source": val_proxy_source,
                        "val_proxy_tasks": (
                            val_proxy_tasks
                            if val_proxy_source == "downstream" else None
                        ),
                        "redundancy_weight": redundancy_weight,
                    },
                }, f, indent=2)

    # ------------------------------------------------------------------
    # Final evaluation and save
    # ------------------------------------------------------------------
    model.eval()
    val_acc, val_loss, val_pp, _, _ = eval_fn(
        model=model,
        reader=val_reader,
        device=device,
        max_num_batches=cfg.eval_batches,
        ctx=type_ctx,
        cfg=cfg,
    )
    model.train()

    print(
        f">Eval: Iter={cfg.iterations} "
        f"val_loss={val_loss:.4f} val_pp={val_pp:.3f} val_acc={val_acc:.6f}"
    )

    history.append({
        "iter": cfg.iterations,
        "val_loss": float(val_loss),
        "val_pp": float(val_pp),
        "val_acc": float(val_acc),
        "entropy": engine.selection_stats["entropy"][-1]
        if engine.selection_stats["entropy"]
        else None,
    })

    if distributed_backend is None or distributed_backend.is_master_process():
        save_checkpoint(
            model, opt, scheduler, cfg.iterations,
            os.path.join(exp_dir, "final.pt"),
        )

    with open(log_file, "w") as f:
        json.dump({
            "history": history,
            "best_val_loss": float(best_val_loss),
            "selection_summary": engine.get_selection_summary(),
            "final_val_loss": float(val_loss),
            "final_val_pp": float(val_pp),
            "final_val_acc": float(val_acc),
        }, f, indent=2)

    engine.detach()

    return {
        "final_val_loss": val_loss,
        "final_val_pp": val_pp,
        "final_val_acc": val_acc,
        "best_val_loss": best_val_loss,
    }