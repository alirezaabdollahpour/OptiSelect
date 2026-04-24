"""
Downstream-task validation proxy for OptiSelect.

Instead of drawing the gradient-direction proxy g_V from the same distribution
as training (the default in Paper Appendix C), this module builds it from a
mixture of held-out multiple-choice benchmark tasks: HellaSwag, ARC-Easy,
ARC-Challenge, PIQA, SciQ. Each item is the concatenation of the prompt and
its CORRECT completion, GPT-2 BPE tokenized by data/benchmarks.py.

This changes the meaning of the score U(z,t) from "how much does candidate z
reduce validation loss on the training distribution" to "how much does z
improve downstream-task next-token prediction" — a targeted, task-aware
alignment signal.

Switch on via cfg.val_proxy_source == "downstream".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

from data.benchmarks import (
    get_arc_challenge,
    get_arc_easy,
    get_hellaswag,
    get_piqa,
    get_sciq,
)


SUPPORTED_PROXY_TASKS = {
    "hellaswag": get_hellaswag,
    "arc_easy": get_arc_easy,
    "arc_challenge": get_arc_challenge,
    "piqa": get_piqa,
    "sciq": get_sciq,
}


def _ensure_task_built(task: str) -> Dict:
    """
    Load a task's tokenized val split, materializing the .bin cache on first
    call. In distributed runs, rank 0 triggers the build and broadcasts any
    failure to all ranks, so a broken loader (e.g., HF datasets>=3 dropping
    a script-based task) fails cleanly everywhere rather than deadlocking.
    """
    loader = SUPPORTED_PROXY_TASKS[task]
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        error_box: List[Optional[str]] = [None]
        if rank == 0:
            try:
                loader()
            except Exception as e:  # noqa: BLE001 — we rethrow below
                error_box[0] = f"{type(e).__name__}: {e}"
        dist.broadcast_object_list(error_box, src=0)
        dist.barrier()
        if error_box[0] is not None:
            raise RuntimeError(
                f"[downstream_proxy] task '{task}' failed on rank 0: "
                f"{error_box[0]}"
            )
        return loader()
    return loader()


def _sample_document_sequences(
    val_data: np.memmap,
    val_lens: np.ndarray,
    n_samples: int,
    sequence_length: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Sample `n_samples` length-(sequence_length+1) contiguous chunks, each
    starting at the first token of a randomly chosen document. The +1 is for
    the next-token label used in the causal LM loss.

    Documents shorter than sequence_length are padded with the task's EOT
    padding (already embedded in the bin by data/benchmarks.py); documents
    longer than sequence_length are truncated.
    """
    doc_starts = np.concatenate([[0], np.cumsum(val_lens)[:-1]]).astype(np.int64)
    n_docs = len(val_lens)
    total_tokens = len(val_data)
    need = sequence_length + 1

    replace = n_samples > n_docs
    idxs = rng.choice(n_docs, size=n_samples, replace=replace)

    seqs: List[np.ndarray] = []
    for d_idx in idxs:
        start = int(doc_starts[d_idx])
        end = start + need
        if end > total_tokens:
            # Document at/near the end of the bin — fall back to a random
            # valid offset so we still contribute something.
            start = int(rng.integers(0, max(1, total_tokens - need)))
            end = start + need
            if end > total_tokens:
                continue
        seqs.append(np.asarray(val_data[start:end], dtype=np.int64))
    return seqs


def build_downstream_proxy_batches(
    tasks: Sequence[str],
    n_docs: int,
    batch_size: int,
    sequence_length: int,
    device: str,
    rng: np.random.Generator,
    vocab_size: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, int]]:
    """
    Build a val proxy from a mixture of downstream tasks.

    Args:
        tasks:             ordered list of task names from SUPPORTED_PROXY_TASKS
        n_docs:            total proxy size in documents (split evenly across tasks)
        batch_size:        mini-batch size for the returned proxy batches
        sequence_length:   per-sample sequence length (must match model)
        device:            device to place returned tensors on
        rng:               numpy RNG (caller controls seeding for reproducibility)
        vocab_size:        model vocab size; must be ≥ 50257 (GPT-2 BPE)

    Returns:
        (batches, per_task_doc_counts)
          batches:  List[(x, y)] where x,y are int64 tensors of shape (B, seq)
          per_task_doc_counts:  {task_name: number of documents contributed}
    """
    if vocab_size < 50257:
        raise ValueError(
            f"val_proxy_source='downstream' requires a GPT-2 BPE model "
            f"(vocab_size ≥ 50257); got vocab_size={vocab_size}. The "
            f"downstream task bins contain BPE token IDs up to 50257 that "
            f"would be out-of-range for this model. Either switch the run "
            f"to val_proxy_source=train or use a BPE-tokenized dataset."
        )
    if not tasks:
        raise ValueError("val_proxy_tasks is empty")
    unknown = [t for t in tasks if t not in SUPPORTED_PROXY_TASKS]
    if unknown:
        raise ValueError(
            f"Unknown proxy tasks: {unknown}. "
            f"Supported: {sorted(SUPPORTED_PROXY_TASKS.keys())}"
        )

    # Pass 1: load each task's tokenized val split. Failures (e.g., HF
    # datasets>=3 rejecting a script-based task like piqa/sciq) are captured
    # so the proxy can still be built from the surviving tasks.
    loaded: Dict[str, Dict] = {}
    failed: Dict[str, str] = {}
    for task in tasks:
        try:
            loaded[task] = _ensure_task_built(task)
        except Exception as e:  # noqa: BLE001
            failed[task] = f"{type(e).__name__}: {e}"
            print(
                f"[OptiSelect][WARN] Downstream task '{task}' could not be "
                f"loaded: {failed[task]}"
            )

    if not loaded:
        raise RuntimeError(
            f"Downstream proxy: all requested tasks failed to load. "
            f"Requested={list(tasks)}. Errors={failed}. "
            f"Tip: set PROXY_TASKS explicitly to exclude broken loaders, "
            f"or fix data/benchmarks.py (e.g., HF datasets>=3 no longer "
            f"supports script-based tasks like piqa/sciq)."
        )
    if failed:
        print(
            f"[OptiSelect][WARN] Proxy will use "
            f"{len(loaded)}/{len(tasks)} tasks "
            f"(dropped: {sorted(failed.keys())}; "
            f"using: {sorted(loaded.keys())})"
        )

    # Rebalance the per-task target across tasks that actually loaded so the
    # total proxy size stays close to n_docs.
    per_task_target = max(1, n_docs // len(loaded))

    all_sequences: List[np.ndarray] = []
    per_task_counts: Dict[str, int] = {}

    for task, data in loaded.items():
        val_data = data["val"]
        val_lens = data["val_len"]
        if isinstance(val_data, torch.Tensor):
            val_data = val_data.numpy()
        if isinstance(val_lens, torch.Tensor):
            val_lens = val_lens.numpy()

        seqs = _sample_document_sequences(
            val_data=val_data,
            val_lens=val_lens,
            n_samples=per_task_target,
            sequence_length=sequence_length,
            rng=rng,
        )
        per_task_counts[task] = len(seqs)
        all_sequences.extend(seqs)

    if not all_sequences:
        raise RuntimeError(
            "Downstream proxy produced zero sequences — check that "
            "pre-tokenized task .bin files built successfully."
        )

    # Interleave tasks so each mini-batch sees a mixture rather than one task
    # at a time. This matters for the averaged gradient proxy in
    # compute_validation_gradient_factors.
    order = rng.permutation(len(all_sequences))
    all_sequences = [all_sequences[i] for i in order]

    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(all_sequences), batch_size):
        chunk = all_sequences[i : i + batch_size]
        if not chunk:
            continue
        stacked = np.stack(chunk, axis=0)  # (B, seq+1)
        x = torch.from_numpy(stacked[:, :-1]).contiguous().to(device)
        y = torch.from_numpy(stacked[:, 1:]).contiguous().to(device)
        batches.append((x, y))

    return batches, per_task_counts
