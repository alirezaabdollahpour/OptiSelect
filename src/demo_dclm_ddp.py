#!/usr/bin/env python
"""Distributed DCLM-Edu demo runner.

This script is the non-interactive, torchrun-friendly counterpart to
demo.ipynb.  It intentionally reads demo_config.yaml and writes to a separate
results directory, but it does not modify either the notebook or the YAML.

The DDP semantics preserve the notebook's global hyperparameters:
  - training.device_batch_size is the global microbatch size.
  - training.total_batch_tokens is the global update batch size.
  - each rank receives an equal shard of every global sampled batch.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from optim.ademamix import AdEMAMix
from optim.lion import Lion
from optim.muon import DistributedMuon
from optim.sign import Signum
from optim.sophia import SophiaG
from selection.influence_scoring import InfluenceConfig
from selection.optiselect_engine import OptiSelectEngine


SUPPORTED_DOWNSTREAM_PROXY_TASKS = {
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
}
_DOWNSTREAM_PROXY_DOC_CACHE: Dict[Tuple[str, str, int], List[np.ndarray]] = {}


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_master(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("demo_dclm_ddp.py requires CUDA.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    return DistInfo(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def barrier(info: DistInfo):
    if info.world_size > 1:
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device, info: DistInfo) -> float:
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    if info.world_size > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= info.world_size
    return float(t.item())


def all_reduce_sum_tensor(tensor: torch.Tensor, info: DistInfo) -> torch.Tensor:
    if info.world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_gather_equal(tensor: torch.Tensor, info: DistInfo) -> torch.Tensor:
    if info.world_size == 1:
        return tensor
    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(info.world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def broadcast_long_tensor(tensor: Optional[torch.Tensor], shape: Tuple[int, ...], info: DistInfo) -> torch.Tensor:
    if info.world_size == 1:
        assert tensor is not None
        return tensor
    if info.is_master:
        assert tensor is not None
        out = tensor.to(info.device, dtype=torch.long).contiguous()
    else:
        out = torch.empty(shape, device=info.device, dtype=torch.long)
    dist.broadcast(out, src=0)
    return out


def rank_print(info: DistInfo, *args, **kwargs):
    if info.is_master:
        print(*args, **kwargs, flush=True)


def find_repo_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    candidates.append(Path("/mloscratch/homes/aabdolla/llm-optimizer-benchmark"))
    for candidate in candidates:
        if (candidate / "src" / "optim").is_dir() and (candidate / "src" / "selection").is_dir():
            return candidate.resolve()
    raise RuntimeError("Could not find llm-optimizer-benchmark repo root.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run demo.ipynb DCLM-Edu training under torchrun/DDP.")
    p.add_argument("--config", default=None, help="Path to demo_config.yaml.")
    p.add_argument("--run-keys", default=None, help="Comma-separated YAML run keys, e.g. adamw,optiselect_adamw.")
    p.add_argument("--results-dir", default=None, help="Output root. Defaults to YAML results_dir with _ddpN suffix.")
    p.add_argument("--summary-name", default=None, help="Sweep summary filename written inside results-dir.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device-batch-size", type=int, default=None, help="Global microbatch examples, matching demo_config.yaml.")
    p.add_argument("--total-batch-tokens", type=int, default=None, help="Global optimizer-update tokens.")
    p.add_argument("--standard-update-tokens", type=int, default=None)
    p.add_argument("--optiselect-update-tokens", type=int, default=None)
    p.add_argument("--iterations", default=None, help="Integer override or auto_from_token_budget.")
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--eval-tokens", type=int, default=None)
    p.add_argument("--final-eval-tokens", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--candidate-multiplier", type=int, default=None)
    p.add_argument("--proxy-source", choices=["dclm_edu_heldout", "downstream"], default=None)
    p.add_argument("--proxy-tasks", default=None, help="Comma-separated downstream proxy tasks.")
    p.add_argument("--candidate-chunk-size", type=int, default=None, help="Local candidate chunk size per rank.")
    p.add_argument("--proxy-batch-size", type=int, default=None)
    p.add_argument("--val-proxy-size", type=int, default=None)
    p.add_argument("--val-proxy-refresh", type=int, default=None)
    p.add_argument("--sketch-dim", type=int, default=None)
    p.add_argument("--countsketch-row-block", type=int, default=None)
    p.add_argument("--countsketch-token-block", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--redundancy-weight", type=float, default=None)
    p.add_argument("--use-countsketch", choices=["0", "1"], default=None)
    p.add_argument("--smoke-steps", type=int, default=None, help="Shortcut: run this many iterations for every run.")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile even if enabled in YAML.")
    return p.parse_args()


def load_cfg(args: argparse.Namespace, info: DistInfo) -> Tuple[dict, Path, Path]:
    repo_root = find_repo_root()
    src_dir = repo_root / "src"
    config_path = Path(args.config).expanduser().resolve() if args.config else src_dir / "demo_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"]["repo_root"] = str(repo_root)
    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)
    if args.device_batch_size is not None:
        cfg["training"]["device_batch_size"] = int(args.device_batch_size)
    if args.total_batch_tokens is not None:
        cfg["training"]["total_batch_tokens"] = int(args.total_batch_tokens)
    if args.standard_update_tokens is not None:
        cfg["training"]["standard_update_tokens"] = int(args.standard_update_tokens)
    if args.optiselect_update_tokens is not None:
        cfg["training"]["optiselect_update_tokens"] = int(args.optiselect_update_tokens)
    if args.iterations is not None:
        cfg["training"]["iterations"] = int(args.iterations) if str(args.iterations).isdigit() else args.iterations
    if args.smoke_steps is not None:
        cfg["training"]["iterations"] = int(args.smoke_steps)
    if args.eval_interval is not None:
        cfg["training"]["eval_interval"] = int(args.eval_interval)
    if args.eval_tokens is not None:
        cfg["training"]["eval_tokens"] = int(args.eval_tokens)
        cfg["training"]["eval_batches"] = "auto_from_eval_tokens"
    if args.final_eval_tokens is not None:
        cfg["training"]["final_eval_tokens"] = int(args.final_eval_tokens)
        cfg["training"]["final_eval_batches"] = "auto_from_final_eval_tokens"
    if args.log_interval is not None:
        cfg["training"]["log_interval"] = int(args.log_interval)

    sel = cfg["selection"]
    if args.candidate_multiplier is not None:
        sel["candidate_multiplier"] = int(args.candidate_multiplier)
    if args.proxy_source is not None:
        sel["proxy_source"] = args.proxy_source
    if args.proxy_tasks is not None:
        sel["proxy_tasks"] = [x.strip() for x in args.proxy_tasks.split(",") if x.strip()]
    if args.candidate_chunk_size is not None:
        sel["candidate_chunk_size"] = int(args.candidate_chunk_size)
    if args.proxy_batch_size is not None:
        sel["proxy_batch_size"] = int(args.proxy_batch_size)
    if args.val_proxy_size is not None:
        sel["val_proxy_size"] = int(args.val_proxy_size)
    if args.val_proxy_refresh is not None:
        sel["val_proxy_refresh"] = int(args.val_proxy_refresh)
    if args.sketch_dim is not None:
        sel["sketch_dim"] = int(args.sketch_dim)
    if args.countsketch_row_block is not None:
        sel["countsketch_row_block"] = int(args.countsketch_row_block)
    if args.countsketch_token_block is not None:
        sel["countsketch_token_block"] = int(args.countsketch_token_block)
    if args.temperature is not None:
        sel["temperature"] = float(args.temperature)
    if args.redundancy_weight is not None:
        sel["redundancy_weight"] = float(args.redundancy_weight)
    if args.use_countsketch is not None:
        sel["use_countsketch"] = bool(int(args.use_countsketch))

    if args.run_keys:
        cfg["experiment"]["run_keys"] = [x.strip() for x in args.run_keys.split(",") if x.strip()]
    if args.no_compile or info.world_size > 1:
        cfg["model"]["use_compile"] = False

    default_results = Path(cfg["paths"]["results_dir"]).expanduser()
    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser()
    else:
        results_dir = default_results.parent / f"{default_results.name}_ddp{info.world_size}_seed{cfg['training']['seed']}"
    cfg["paths"]["results_dir"] = str(results_dir)

    return cfg, config_path, results_dir


def model_dimensions(cfg: dict) -> Tuple[int, int]:
    depth = int(cfg["model"]["depth"])
    aspect = int(cfg["model"]["aspect_ratio"])
    head_dim = int(cfg["model"]["head_dim"])
    base_dim = depth * aspect
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return model_dim, num_heads


def effective_batch_tokens(cfg: dict) -> int:
    return int(cfg["training"]["device_batch_size"]) * int(cfg["model"]["max_seq_len"])


def planned_grad_accum_steps(cfg: dict) -> int:
    per_micro = effective_batch_tokens(cfg)
    requested = int(cfg["training"]["total_batch_tokens"])
    if requested % per_micro != 0:
        raise ValueError(
            f"total_batch_tokens={requested:,} must be divisible by "
            f"device_batch_size*max_seq_len={per_micro:,}."
        )
    return max(1, requested // per_micro)


def update_batch_tokens(cfg: dict) -> int:
    return planned_grad_accum_steps(cfg) * effective_batch_tokens(cfg)


def update_batch_examples(cfg: dict) -> int:
    return planned_grad_accum_steps(cfg) * int(cfg["training"]["device_batch_size"])


def resolve_run_iterations(optiselect: bool, cfg: dict) -> Optional[int]:
    raw = cfg["training"].get("iterations")
    if isinstance(raw, int):
        return int(raw)
    if raw == "auto_from_token_budget":
        key = "optiselect_update_tokens" if optiselect else "standard_update_tokens"
        return int(math.ceil(int(cfg["training"][key]) / update_batch_tokens(cfg)))
    if raw is None:
        return None
    raise ValueError(f"Unknown training.iterations value: {raw!r}")


def target_update_tokens_for_run(optiselect: bool, cfg: dict) -> int:
    key = "optiselect_update_tokens" if optiselect else "standard_update_tokens"
    return int(cfg["training"].get(key, 0) or 0)


def resolve_eval_batches(cfg: dict, final: bool = False) -> int:
    raw_key = "final_eval_batches" if final else "eval_batches"
    token_key = "final_eval_tokens" if final else "eval_tokens"
    raw = cfg["training"][raw_key]
    if isinstance(raw, int):
        return int(raw)
    expected = "auto_from_final_eval_tokens" if final else "auto_from_eval_tokens"
    if raw == expected:
        return int(math.ceil(int(cfg["training"][token_key]) / effective_batch_tokens(cfg)))
    raise ValueError(f"Unknown training.{raw_key} value: {raw!r}")


def scaled_lr(opt_key: str, cfg: dict) -> float:
    opt_cfg = cfg["optimizers"][opt_key]
    lr = float(opt_cfg["lr"])
    scaling = cfg.get("scaling", {})
    if not scaling.get("enabled", False):
        return lr
    model_dim, _ = model_dimensions(cfg)
    width_power = float(opt_cfg.get("width_power", scaling.get("default_width_power", 0.0)))
    batch_power = float(opt_cfg.get("batch_power", scaling.get("default_batch_power", 0.0)))
    width_scale = (model_dim / float(scaling["reference_n_embd"])) ** width_power
    batch_scale = (int(cfg["training"]["total_batch_tokens"]) / float(scaling["reference_batch_tokens"])) ** batch_power
    return lr * width_scale * batch_scale


def validate_learning_rates(cfg: dict):
    adamw_lr = scaled_lr("adamw", cfg)
    bad = []
    for opt_key in cfg.get("safety", {}).get("forbid_same_lr_as_adamw_for", []):
        if math.isclose(scaled_lr(opt_key, cfg), adamw_lr, rel_tol=0.0, abs_tol=1e-15):
            bad.append(opt_key)
    if bad:
        raise ValueError(
            "Unsafe LR config: these optimizers share AdamW LR even though they "
            "need their own scale: " + ", ".join(bad)
        )


def validate_cfg(cfg: dict, info: DistInfo):
    run_keys = cfg["experiment"]["run_keys"]
    missing = [k for k in run_keys if k not in cfg["runs"]]
    if missing:
        raise ValueError(f"Unknown run key(s): {missing}")
    supported_opts = {"adamw", "d-muon", "sgd", "signsgd", "lion", "signum", "ademamix", "sophia"}
    unsupported = [k for k in run_keys if cfg["runs"][k]["optimizer"] not in supported_opts]
    if unsupported:
        raise ValueError(
            "demo_dclm_ddp.py supports the optimizer keys used by demo_config.yaml "
            f"({sorted(supported_opts)}). "
            f"Unsupported run key(s): {unsupported}"
        )
    validate_learning_rates(cfg)
    global_batch = int(cfg["training"]["device_batch_size"])
    if global_batch % info.world_size != 0:
        raise ValueError(
            f"training.device_batch_size={global_batch} must be divisible by "
            f"world_size={info.world_size} so every rank has the same local batch."
        )
    planned_grad_accum_steps(cfg)
    sel = cfg.get("selection", {})
    proxy_source = sel.get("proxy_source", "dclm_edu_heldout")
    if proxy_source not in {"dclm_edu_heldout", "downstream"}:
        raise ValueError("selection.proxy_source must be one of: dclm_edu_heldout, downstream")
    if proxy_source == "downstream":
        tasks = [str(t).strip() for t in sel.get("proxy_tasks", []) if str(t).strip()]
        if not tasks:
            raise ValueError("selection.proxy_source=downstream requires selection.proxy_tasks.")
        unknown = sorted(set(tasks) - SUPPORTED_DOWNSTREAM_PROXY_TASKS)
        if unknown:
            raise ValueError(
                f"Unknown downstream proxy task(s): {unknown}. "
                f"Supported: {sorted(SUPPORTED_DOWNSTREAM_PROXY_TASKS)}"
            )


def seed_everything(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def autocast_context(cfg: dict, device: torch.device):
    dtype_name = cfg["training"]["dtype"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    if device.type == "cuda" and dtype != torch.float32:
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def dataset_fingerprint(cfg: dict) -> str:
    relevant = {"dataset": cfg["dataset"], "tokenizer": cfg["tokenizer"]}
    blob = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def dclm_cache_paths(cfg: dict) -> Dict[str, Path]:
    local_dir = Path(cfg["paths"]["datasets_dir"]).expanduser() / cfg["dataset"]["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": local_dir,
        "train": local_dir / "train.bin",
        "val": local_dir / "val.bin",
        "train_len": local_dir / "train.len.npy",
        "val_len": local_dir / "val.len.npy",
        "train_score": local_dir / "train.edu_score.npy",
        "val_score": local_dir / "val.edu_score.npy",
        "tokenizer": local_dir / "tokenizer.json",
        "metadata": local_dir / "metadata.json",
    }


def cache_is_valid(paths: Dict[str, Path], cfg: dict) -> bool:
    needed = [
        paths["train"], paths["val"], paths["train_len"], paths["val_len"],
        paths["train_score"], paths["val_score"], paths["tokenizer"], paths["metadata"],
    ]
    if not all(p.exists() for p in needed):
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text())
    except Exception:
        return False
    return metadata.get("fingerprint") == dataset_fingerprint(cfg)


def build_dclm_cache(paths: Dict[str, Path], cfg: dict):
    from datasets import load_dataset
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import decoders, models, pre_tokenizers, trainers
    from tqdm.auto import tqdm

    ds_cfg = cfg["dataset"]
    tok_cfg = cfg["tokenizer"]
    total_needed = int(ds_cfg["num_train_docs"]) + int(ds_cfg["num_val_docs"])
    data_files = [f"hf://datasets/{ds_cfg['hf_dataset']}/{rel}" for rel in ds_cfg["data_files"]]
    print("Building local DCLM-Edu token cache from configured shards...", flush=True)
    ds = load_dataset("parquet", data_files={"train": data_files}, split="train", streaming=bool(ds_cfg.get("streaming", True)))

    texts = []
    scores = []
    with tqdm(total=total_needed, desc="DCLM-Edu docs") as pbar:
        for ex in ds:
            score = ex.get(ds_cfg["score_column"], None)
            if ds_cfg.get("min_edu_score") is not None and (score is None or float(score) < float(ds_cfg["min_edu_score"])):
                continue
            text = ex.get(ds_cfg["text_column"], "")
            if not isinstance(text, str) or not text.strip():
                continue
            texts.append(text)
            scores.append(float(score) if score is not None else float("nan"))
            pbar.update(1)
            if len(texts) >= total_needed:
                break
    if len(texts) < total_needed:
        raise RuntimeError(f"Only collected {len(texts)} docs, expected {total_needed}.")

    train_texts = texts[: int(ds_cfg["num_train_docs"])]
    val_texts = texts[int(ds_cfg["num_train_docs"]) :]
    train_scores = np.asarray(scores[: int(ds_cfg["num_train_docs"])], dtype=np.float32)
    val_scores = np.asarray(scores[int(ds_cfg["num_train_docs"]) :], dtype=np.float32)
    tokenizer = HFTokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=int(tok_cfg["vocab_size"]),
        special_tokens=[tok_cfg["bos_token"]],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if tok_cfg.get("initial_alphabet", "byte_level") == "byte_level" else [],
        min_frequency=int(tok_cfg["min_frequency"]),
        show_progress=True,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    bos_id = tokenizer.token_to_id(tok_cfg["bos_token"])

    def tokenize_texts(texts_in: List[str], desc: str):
        all_ids, lengths = [], []
        for start in tqdm(range(0, len(texts_in), 1000), desc=desc):
            encoded = tokenizer.encode_batch(texts_in[start : start + 1000])
            for enc in encoded:
                doc_ids = [bos_id] + enc.ids
                all_ids.extend(doc_ids)
                lengths.append(len(doc_ids))
        return np.asarray(all_ids, dtype=np.uint16), np.asarray(lengths, dtype=np.int32)

    train_arr, train_lens = tokenize_texts(train_texts, "Tokenizing train")
    val_arr, val_lens = tokenize_texts(val_texts, "Tokenizing val")
    for split, arr in (("train", train_arr), ("val", val_arr)):
        mm = np.memmap(paths[split], dtype=np.uint16, mode="w+", shape=arr.shape)
        mm[:] = arr[:]
        mm.flush()
        del mm
    np.save(paths["train_len"], train_lens)
    np.save(paths["val_len"], val_lens)
    np.save(paths["train_score"], train_scores)
    np.save(paths["val_score"], val_scores)
    tokenizer.save(str(paths["tokenizer"]))
    metadata = {
        "fingerprint": dataset_fingerprint(cfg),
        "hf_dataset": ds_cfg["hf_dataset"],
        "data_files": ds_cfg["data_files"],
        "min_edu_score": ds_cfg["min_edu_score"],
        "num_train_docs": int(ds_cfg["num_train_docs"]),
        "num_val_docs": int(ds_cfg["num_val_docs"]),
        "train_tokens": int(train_arr.size),
        "val_tokens": int(val_arr.size),
        "train_docs": int(train_lens.size),
        "val_docs": int(val_lens.size),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "score_column": ds_cfg["score_column"],
        "train_edu_score_mean": float(np.nanmean(train_scores)),
        "val_edu_score_mean": float(np.nanmean(val_scores)),
        "cache_dir": str(paths["dir"]),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2))
    gc.collect()


def load_dclm_data(cfg: dict, info: DistInfo):
    from tokenizers import Tokenizer as HFTokenizer

    paths = dclm_cache_paths(cfg)
    if info.is_master and not cache_is_valid(paths, cfg):
        build_dclm_cache(paths, cfg)
    barrier(info)
    if not cache_is_valid(paths, cfg):
        raise RuntimeError(f"DCLM cache is not valid at {paths['dir']}. Run rank 0 preparation first.")

    in_ram = bool(cfg["dataset"].get("load_tokens_in_ram", True))
    def load_tokens(path: Path):
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        if in_ram:
            return torch.from_numpy(np.array(mm, dtype=np.int64))
        return mm

    train_data = load_tokens(paths["train"])
    val_data = load_tokens(paths["val"])
    train_lens = np.load(paths["train_len"])
    val_lens = np.load(paths["val_len"])
    train_scores = np.load(paths["train_score"])
    val_scores = np.load(paths["val_score"])
    tokenizer = HFTokenizer.from_file(str(paths["tokenizer"]))
    metadata = json.loads(paths["metadata"].read_text())

    if len(train_data) < int(cfg["dataset"].get("min_train_tokens", 0) or 0):
        raise RuntimeError("DCLM train cache is smaller than dataset.min_train_tokens.")
    if len(val_data) < int(cfg["dataset"].get("min_val_tokens", 0) or 0):
        raise RuntimeError("DCLM val cache is smaller than dataset.min_val_tokens.")
    if cfg.get("selection", {}).get("proxy_source", "dclm_edu_heldout") == "dclm_edu_heldout" and len(val_lens) < int(cfg["selection"]["val_proxy_size"]):
        raise RuntimeError("DCLM val cache has fewer documents than selection.val_proxy_size.")

    rank_print(info, f"Using cached DCLM-Edu tokens at {paths['dir']}")
    rank_print(info, f"Train tokens={len(train_data):,} | Val tokens={len(val_data):,} | Vocab={tokenizer.get_vocab_size():,}")
    return train_data, val_data, train_lens, val_lens, train_scores, val_scores, tokenizer.token_to_id(cfg["tokenizer"]["bos_token"]), tokenizer.get_vocab_size(), metadata


def _slice_token_source(data_source, start_indices: Iterable[int], seq_len: int):
    if torch.is_tensor(data_source):
        xs = torch.stack([data_source[int(i) : int(i) + seq_len] for i in start_indices])
        ys = torch.stack([data_source[int(i) + 1 : int(i) + seq_len + 1] for i in start_indices])
    else:
        xs = torch.stack([torch.from_numpy(np.asarray(data_source[int(i) : int(i) + seq_len], dtype=np.int64)) for i in start_indices])
        ys = torch.stack([torch.from_numpy(np.asarray(data_source[int(i) + 1 : int(i) + seq_len + 1], dtype=np.int64)) for i in start_indices])
    return xs.contiguous(), ys.contiguous()


def make_sharded_dataloader(
    data_source,
    global_batch_size: int,
    seq_len: int,
    device: torch.device,
    seed: int,
    info: DistInfo,
    return_indices: bool = False,
):
    if global_batch_size % info.world_size != 0:
        raise ValueError("global_batch_size must be divisible by world_size")
    local_batch = global_batch_size // info.world_size
    n = len(data_source) - seq_len - 1
    if n <= 0:
        raise RuntimeError(f"Data too short ({len(data_source)} tokens) for seq_len={seq_len}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    while True:
        global_ix = torch.randint(0, n, (global_batch_size,), generator=gen)
        start = info.rank * local_batch
        local_ix = global_ix[start : start + local_batch]
        x, y = _slice_token_source(data_source, local_ix.tolist(), seq_len)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if return_indices:
            yield x, y, local_ix.to(device, non_blocking=True)
        else:
            yield x, y


def _token_slice_1d(data_source, start: int, end: int):
    if torch.is_tensor(data_source):
        return data_source[int(start) : int(end)].to(dtype=torch.long)
    return torch.from_numpy(np.asarray(data_source[int(start) : int(end)], dtype=np.int64))


def _make_proxy_batches_from_sequences(
    sequences: Sequence[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    for start in range(0, len(sequences), batch_size):
        stacked = torch.stack(list(sequences[start : start + batch_size]), dim=0).contiguous()
        batches.append((stacked[:, :-1].to(device, non_blocking=True), stacked[:, 1:].to(device, non_blocking=True)))
    return batches


def _make_proxy_batches_from_xy(
    sequences: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    for start in range(0, len(sequences), batch_size):
        chunk = list(sequences[start : start + batch_size])
        x = torch.stack([item[0] for item in chunk], dim=0).contiguous()
        y = torch.stack([item[1] for item in chunk], dim=0).contiguous()
        batches.append((x.to(device, non_blocking=True), y.to(device, non_blocking=True)))
    return batches


def build_dclm_heldout_proxy(
    val_data,
    val_doc_lens,
    cfg: dict,
    device: torch.device,
    bos_token_id: int,
    refresh_index: int = 0,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], dict]:
    sel_cfg = cfg["selection"]
    batch_size = int(sel_cfg.get("proxy_batch_size", cfg["training"]["device_batch_size"]))
    seq_len = int(cfg["model"]["max_seq_len"])
    need = seq_len + 1
    proxy_size = int(sel_cfg["val_proxy_size"])
    starts = np.concatenate([[0], np.cumsum(val_doc_lens)[:-1]]).astype(np.int64)
    rng = np.random.default_rng(int(cfg["training"]["seed"]) + 10_000 + int(refresh_index) * 104_729)
    replace = proxy_size > len(val_doc_lens)
    doc_idxs = rng.choice(len(val_doc_lens), size=proxy_size, replace=replace)

    sequences = []
    for doc_idx in doc_idxs:
        doc_start = int(starts[int(doc_idx)])
        doc_len = int(val_doc_lens[int(doc_idx)])
        offset = int(rng.integers(0, doc_len - need + 1)) if doc_len > need else 0
        seq = _token_slice_1d(val_data, doc_start + offset, min(doc_start + offset + need, doc_start + doc_len))
        if seq.numel() < need:
            pad = torch.full((need - seq.numel(),), int(bos_token_id), dtype=torch.long)
            seq = torch.cat([seq, pad], dim=0)
        sequences.append(seq[:need])

    batches = _make_proxy_batches_from_sequences(sequences, batch_size, device)
    return batches, {"source": "dclm_edu_heldout", "counts": {"dclm_edu_heldout": len(sequences)}}


def _broadcast_rank0_error(info: DistInfo, action: str, fn):
    error_box: List[Optional[str]] = [None]
    if info.is_master:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - rethrow on all ranks below
            error_box[0] = f"{type(exc).__name__}: {exc}"
    if info.world_size > 1:
        dist.broadcast_object_list(error_box, src=0)
    if error_box[0] is not None:
        raise RuntimeError(f"{action} failed on rank 0: {error_box[0]}")
    barrier(info)


def _load_raw_downstream_dataset(task: str):
    from datasets import load_dataset

    if task == "hellaswag":
        return load_dataset("Rowan/hellaswag", split="validation")
    if task == "arc_easy":
        return load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    if task == "arc_challenge":
        return load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    if task == "openbookqa":
        return load_dataset("allenai/openbookqa", "main", split="validation")
    raise ValueError(f"Unsupported downstream proxy task: {task}")


def _choice_by_answer_key(choices: dict, answer_key) -> str:
    labels = [str(x) for x in choices.get("label", [])]
    texts = [str(x) for x in choices.get("text", [])]
    key = str(answer_key)
    if key in labels:
        return texts[labels.index(key)]
    if key.isdigit():
        idx = int(key)
        if 0 <= idx < len(texts):
            return texts[idx]
        one_based = idx - 1
        if 0 <= one_based < len(texts):
            return texts[one_based]
    raise ValueError(f"Could not map answer key {answer_key!r} to choices {labels!r}")


def _format_downstream_proxy_text(task: str, ex: dict) -> str:
    if task == "hellaswag":
        endings = list(ex["endings"])
        label = int(ex["label"])
        return f"{str(ex['ctx']).strip()} {str(endings[label]).strip()}".strip()
    if task in {"arc_easy", "arc_challenge"}:
        answer = _choice_by_answer_key(ex["choices"], ex["answerKey"])
        return f"Question: {str(ex['question']).strip()}\nAnswer: {answer.strip()}"
    if task == "openbookqa":
        answer = _choice_by_answer_key(ex["choices"], ex["answerKey"])
        return f"Question: {str(ex['question_stem']).strip()}\nAnswer: {answer.strip()}"
    raise ValueError(f"Unsupported downstream proxy task: {task}")


def _load_downstream_proxy_docs(
    task: str,
    cfg: dict,
    info: DistInfo,
    bos_token_id: int,
) -> List[np.ndarray]:
    from tokenizers import Tokenizer as HFTokenizer

    paths = dclm_cache_paths(cfg)
    tokenizer_key = str(paths["tokenizer"].resolve())
    cache_key = (task, tokenizer_key, int(bos_token_id))
    if cache_key in _DOWNSTREAM_PROXY_DOC_CACHE:
        return _DOWNSTREAM_PROXY_DOC_CACHE[cache_key]

    _broadcast_rank0_error(
        info,
        f"Downloading downstream proxy task '{task}'",
        lambda: _load_raw_downstream_dataset(task),
    )
    ds = _load_raw_downstream_dataset(task)
    tokenizer = HFTokenizer.from_file(tokenizer_key)
    docs: List[np.ndarray] = []
    skipped = 0
    for ex in ds:
        try:
            text = _format_downstream_proxy_text(task, ex)
            ids = [int(bos_token_id)] + [int(i) for i in tokenizer.encode(text).ids]
        except Exception:
            skipped += 1
            continue
        if len(ids) >= 2:
            docs.append(np.asarray(ids, dtype=np.int64))

    if not docs:
        raise RuntimeError(f"Downstream proxy task '{task}' produced zero tokenized examples.")
    if skipped and info.is_master:
        rank_print(info, f"[WARN] Downstream proxy task '{task}' skipped {skipped} malformed examples.")
    _DOWNSTREAM_PROXY_DOC_CACHE[cache_key] = docs
    return docs


def build_downstream_validation_proxy(
    cfg: dict,
    info: DistInfo,
    device: torch.device,
    bos_token_id: int,
    refresh_index: int = 0,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], dict]:
    sel_cfg = cfg["selection"]
    tasks = [str(t).strip() for t in sel_cfg.get("proxy_tasks", []) if str(t).strip()]
    unknown = sorted(set(tasks) - SUPPORTED_DOWNSTREAM_PROXY_TASKS)
    if unknown:
        raise ValueError(f"Unknown downstream proxy task(s): {unknown}")
    batch_size = int(sel_cfg.get("proxy_batch_size", cfg["training"]["device_batch_size"]))
    seq_len = int(cfg["model"]["max_seq_len"])
    proxy_size = int(sel_cfg["val_proxy_size"])
    rng = np.random.default_rng(int(cfg["training"]["seed"]) + 20_000 + int(refresh_index) * 104_729)

    docs_by_task = {
        task: _load_downstream_proxy_docs(task, cfg, info, bos_token_id)
        for task in tasks
    }
    base = proxy_size // len(tasks)
    remainder = proxy_size % len(tasks)
    sequences: List[Tuple[torch.Tensor, torch.Tensor]] = []
    counts: Dict[str, int] = {}
    real_target_tokens = 0
    for task_idx, task in enumerate(tasks):
        docs = docs_by_task[task]
        n_task = base + (1 if task_idx < remainder else 0)
        replace = n_task > len(docs)
        idxs = rng.choice(len(docs), size=n_task, replace=replace)
        task_count = 0
        for doc_idx in idxs:
            x, y, target_tokens = _sample_downstream_xy(docs[int(doc_idx)], seq_len, bos_token_id, rng)
            sequences.append((x, y))
            real_target_tokens += int(target_tokens)
            task_count += 1
        counts[task] = task_count

    order = rng.permutation(len(sequences))
    ordered_sequences = [sequences[int(i)] for i in order]
    batches = _make_proxy_batches_from_xy(ordered_sequences, batch_size, device)
    meta = {
        "source": "downstream",
        "tasks": tasks,
        "counts": counts,
        "real_target_tokens": int(real_target_tokens),
    }
    return batches, meta


def _sample_downstream_xy(
    ids: np.ndarray,
    seq_len: int,
    bos_token_id: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    need = seq_len + 1
    if len(ids) > need:
        offset = int(rng.integers(0, len(ids) - need + 1))
        ids = ids[offset : offset + need]
    x = torch.full((seq_len,), int(bos_token_id), dtype=torch.long)
    y = torch.full((seq_len,), -1, dtype=torch.long)
    ids_t = torch.from_numpy(np.asarray(ids, dtype=np.int64)).to(dtype=torch.long)
    valid_targets = max(0, min(seq_len, ids_t.numel() - 1))
    if valid_targets > 0:
        x[:valid_targets] = ids_t[:valid_targets]
        y[:valid_targets] = ids_t[1 : valid_targets + 1]
    return x, y, valid_targets


def build_validation_proxy(
    val_data,
    val_doc_lens,
    cfg: dict,
    info: DistInfo,
    device: torch.device,
    bos_token_id: int,
    refresh_index: int = 0,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], dict]:
    source = cfg["selection"].get("proxy_source", "dclm_edu_heldout")
    if source == "dclm_edu_heldout":
        return build_dclm_heldout_proxy(val_data, val_doc_lens, cfg, device, bos_token_id, refresh_index)
    if source == "downstream":
        return build_downstream_validation_proxy(cfg, info, device, bos_token_id, refresh_index)
    raise ValueError(f"Unknown selection.proxy_source={source!r}")


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 512
    logit_chunk_tokens: int = 16384


def rms_norm(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], -1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = min(32, self.n_embd)
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if ve is not None and self.ve_gate is not None:
            ve_reshaped = ve.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.transpose(1, 2).unsqueeze(-1) * ve_reshaped
        cos, sin = cos_sin
        q = rms_norm(apply_rotary_emb(q, cos, sin))
        k = rms_norm(apply_rotary_emb(k, cos, sin))
        if self.n_kv_head < self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin):
        x = x + self.attn(rms_norm(x), ve, cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        })
        cos, sin = self._precompute_rotary(config.sequence_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
        return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for block in self.transformer.h:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                nn.init.zeros_(block.attn.ve_gate.weight)

    def _chunked_lm_loss(self, x, targets, softcap, reduction):
        flat_x = x.reshape(-1, x.size(-1))
        flat_targets = targets.reshape(-1)
        chunk_tokens = max(1, int(getattr(self.config, "logit_chunk_tokens", 0) or flat_x.size(0)))

        if reduction == "none":
            losses = []
            for start in range(0, flat_x.size(0), chunk_tokens):
                end = min(start + chunk_tokens, flat_x.size(0))
                logits = self.lm_head(flat_x[start:end]).float()
                logits = softcap * torch.tanh(logits / softcap)
                losses.append(F.cross_entropy(logits, flat_targets[start:end], ignore_index=-1, reduction="none"))
            return torch.cat(losses, dim=0).view_as(targets)

        def chunk_loss(chunk_x, chunk_targets):
            logits = self.lm_head(chunk_x).float()
            logits = softcap * torch.tanh(logits / softcap)
            return F.cross_entropy(logits, chunk_targets, ignore_index=-1, reduction="sum")

        total_loss = flat_x.new_zeros((), dtype=torch.float32)
        total_count = flat_x.new_zeros((), dtype=torch.float32)
        use_checkpoint = torch.is_grad_enabled() and flat_x.requires_grad
        for start in range(0, flat_x.size(0), chunk_tokens):
            end = min(start + chunk_tokens, flat_x.size(0))
            target_chunk = flat_targets[start:end]
            if use_checkpoint:
                total_loss = total_loss + checkpoint(
                    lambda chunk_x, targets=target_chunk: chunk_loss(chunk_x, targets),
                    flat_x[start:end],
                    use_reentrant=False,
                )
            else:
                total_loss = total_loss + chunk_loss(flat_x[start:end], target_chunk)
            total_count = total_count + (target_chunk != -1).sum().to(torch.float32)
        if reduction == "sum":
            return total_loss
        if reduction == "mean":
            return total_loss / total_count.clamp_min(1.0)
        raise ValueError(f"Unsupported reduction: {reduction}")

    def forward(self, idx, targets=None, reduction="mean", get_logits=False, **kwargs):
        B, T = idx.size()
        cos_sin = self.cos[:, :, :T, :], self.sin[:, :, :T, :]
        x = rms_norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin)
        x = rms_norm(x)
        softcap = 15
        logits = None
        if get_logits or targets is None:
            logits = self.lm_head(x).float()
            logits = softcap * torch.tanh(logits / softcap)
        loss = None
        if targets is not None:
            if logits is None:
                loss = self._chunked_lm_loss(x, targets, softcap, reduction)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)
        return {"loss": loss, "logits": logits if (get_logits or targets is None) else None, "aux_losses": {}}


def build_model(cfg: dict, vocab_size: int, device: torch.device):
    model_dim, num_heads = model_dimensions(cfg)
    config = GPTConfig(
        sequence_len=int(cfg["model"]["max_seq_len"]),
        vocab_size=int(vocab_size),
        n_layer=int(cfg["model"]["depth"]),
        n_head=int(num_heads),
        n_kv_head=int(num_heads),
        n_embd=int(model_dim),
        logit_chunk_tokens=int(cfg["model"].get("logit_chunk_tokens", 16384)),
    )
    model = GPT(config).to(device)
    model.init_weights()
    return model, config


def should_no_decay(name: str, param: nn.Parameter) -> bool:
    return param.ndim < 2 or "transformer.wte" in name or "value_embeds" in name


def adam_style_param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if should_no_decay(name, p) else decay).append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def dmuon_param_groups(model: nn.Module, opt_cfg: dict, primary_lr: float, fallback_lr: float):
    muon_decay, fallback_decay, fallback_no_decay = [], [], []
    fallback_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                fallback_ids.add(id(p))
    for p in model.lm_head.parameters(recurse=False):
        fallback_ids.add(id(p))

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        can_muon = p.ndim >= 2 and id(p) not in fallback_ids and p.size(0) < 10000
        if can_muon:
            muon_decay.append(p)
        elif should_no_decay(name, p):
            fallback_no_decay.append(p)
        else:
            fallback_decay.append(p)

    wd = float(opt_cfg.get("weight_decay", 0.0))
    groups = []
    if muon_decay:
        groups.append({"params": muon_decay, "lr": float(primary_lr), "weight_decay": wd})
    if fallback_decay:
        groups.append({"params": fallback_decay, "lr": float(fallback_lr), "weight_decay": wd})
    if fallback_no_decay:
        groups.append({"params": fallback_no_decay, "lr": float(fallback_lr), "weight_decay": 0.0})
    return groups, fallback_ids


def force_dmuon_adamw_fallback(optimizer, fallback_ids) -> int:
    forced = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) in fallback_ids or p.ndim < 2:
                optimizer.state[p]["use_muon"] = False
                forced += p.numel()
    return forced


def build_optimizer(model: nn.Module, opt_key: str, cfg: dict, resolved_iterations: Optional[int] = None):
    opt_cfg = cfg["optimizers"][opt_key]
    lr = scaled_lr(opt_key, cfg)
    wd = float(opt_cfg.get("weight_decay", 0.0))

    if opt_key == "adamw":
        groups = adam_style_param_groups(model, wd)
        opt = torch.optim.AdamW(groups, lr=lr, betas=(float(opt_cfg["beta1"]), float(opt_cfg["beta2"])))
        repo_opt_name = "adamw"
    elif opt_key == "ademamix":
        groups = adam_style_param_groups(model, wd)
        iterations = int(resolved_iterations or max(1, int(cfg["training"].get("time_budget_sec") or 1)))
        beta3_warmup = opt_cfg.get("beta3_warmup")
        alpha_warmup = opt_cfg.get("alpha_warmup")
        beta3_warmup = iterations if beta3_warmup == "auto_iterations" else beta3_warmup
        alpha_warmup = iterations if alpha_warmup == "auto_iterations" else alpha_warmup
        opt = AdEMAMix(
            groups,
            lr=lr,
            betas=(float(opt_cfg["beta1"]), float(opt_cfg["beta2"]), float(opt_cfg["beta3"])),
            alpha=float(opt_cfg["alpha"]),
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            weight_decay=wd,
        )
        repo_opt_name = "ademamix"
    elif opt_key == "sgd":
        groups = adam_style_param_groups(model, wd)
        opt = torch.optim.SGD(
            groups,
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.0)),
            nesterov=bool(opt_cfg.get("nesterov", False)),
            weight_decay=wd,
        )
        repo_opt_name = "sgd"
    elif opt_key == "signsgd":
        groups = adam_style_param_groups(model, wd)
        opt = Signum(
            groups,
            lr=lr,
            momentum=0.0,
            dampening=0.0,
            nesterov=False,
            weight_decay=wd,
            sign_update=True,
        )
        repo_opt_name = "signsgd"
    elif opt_key == "lion":
        groups = adam_style_param_groups(model, wd)
        opt = Lion(
            groups,
            lr=lr,
            betas=(float(opt_cfg["beta1"]), float(opt_cfg["beta2"])),
            weight_decay=wd,
        )
        repo_opt_name = "lion"
    elif opt_key == "signum":
        groups = adam_style_param_groups(model, wd)
        opt = Signum(
            groups,
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.9)),
            dampening=float(opt_cfg.get("dampening", 0.0)),
            nesterov=bool(opt_cfg.get("nesterov", False)),
            weight_decay=wd,
            sign_update=True,
        )
        repo_opt_name = "signum"
    elif opt_key == "sophia":
        groups = adam_style_param_groups(model, wd)
        opt = SophiaG(
            groups,
            lr=lr,
            betas=(float(opt_cfg["beta1"]), float(opt_cfg["beta2"])),
            rho=float(opt_cfg["rho"]),
            weight_decay=wd,
        )
        repo_opt_name = "sophiag"
    elif opt_key == "d-muon":
        fallback_lr = float(opt_cfg.get("adamw_lr", cfg["optimizers"]["adamw"]["lr"]))
        groups, fallback_ids = dmuon_param_groups(model, opt_cfg, primary_lr=lr, fallback_lr=fallback_lr)
        opt = DistributedMuon(
            groups,
            lr=lr,
            weight_decay=wd,
            matched_adamw_rms=float(opt_cfg.get("matched_adamw_rms", 0.2)),
            momentum=float(opt_cfg.get("momentum", 0.95)),
            nesterov=bool(opt_cfg.get("nesterov", True)),
            ns_steps=int(opt_cfg.get("muon_ns_steps", 5)),
            adamw_betas=(float(opt_cfg.get("beta1", 0.9)), float(opt_cfg.get("beta2", 0.99))),
        )
        forced = force_dmuon_adamw_fallback(opt, fallback_ids)
        print(f"d-Muon AdamW fallback params forced: {forced:,}", flush=True)
        repo_opt_name = "d-muon"
    else:
        raise ValueError(f"Unknown optimizer: {opt_key}")

    for group in opt.param_groups:
        group["initial_lr"] = float(group["lr"])
    return opt, repo_opt_name


def resolve_sophia_bs_examples(cfg: dict, opt_cfg: dict) -> int:
    raw = opt_cfg.get("sophia_bs", "auto_effective_batch_examples")
    if raw == "auto_effective_batch_examples":
        return update_batch_examples(cfg)
    return int(raw)


def step_sophia_hessian(ddp_model, opt, opt_cfg: dict, batch, seq_len: int, cfg: dict, info: DistInfo):
    x, y = batch
    opt.zero_grad(set_to_none=True)
    with autocast_context(cfg, info.device):
        sampled = ddp_model(x, targets=y, get_logits=True)
    logits = sampled["logits"]
    y_sample = torch.distributions.Categorical(logits=logits).sample()
    loss_sampled = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y_sample.view(-1),
        ignore_index=-1,
    )
    loss_sampled.backward()
    opt.update_hessian()
    opt.zero_grad(set_to_none=True)


def optimizer_step(ddp_model, raw_model, opt, opt_key: str, opt_cfg: dict, step_idx: int, grad_clip: float, hessian_batch, seq_len: int, cfg: dict, info: DistInfo):
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(grad_clip))
    if opt_key == "sophia":
        opt.step(bs=resolve_sophia_bs_examples(cfg, opt_cfg) * int(seq_len))
    else:
        opt.step()
    opt.zero_grad(set_to_none=True)

    if opt_key == "sophia" and hessian_batch is not None:
        freq = int(opt_cfg.get("precondition_frequency", 10))
        if freq > 0 and step_idx % freq == freq - 1:
            step_sophia_hessian(ddp_model, opt, opt_cfg, hessian_batch, seq_len, cfg, info)


def get_lr_multiplier(progress: float, cfg: dict) -> float:
    warmup = float(cfg["training"]["warmup_ratio"])
    cooldown = float(cfg["training"]["cooldown_ratio"])
    final_frac = float(cfg["training"]["final_lr_fraction"])
    progress = min(max(float(progress), 0.0), 1.0)
    if warmup > 0 and progress < warmup:
        return progress / warmup
    if progress < 1.0 - cooldown:
        return 1.0
    if cooldown <= 0:
        return final_frac
    cool = (1.0 - progress) / cooldown
    return cool + (1.0 - cool) * final_frac


def apply_lr_schedule(opt, multiplier: float):
    for group in opt.param_groups:
        group["lr"] = float(group["initial_lr"]) * float(multiplier)


def primary_lr(opt) -> float:
    return float(opt.param_groups[0]["lr"])


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, num_batches: int, cfg: dict, info: DistInfo) -> Tuple[float, float]:
    model.eval()
    local_loss_sum = torch.zeros((), device=info.device, dtype=torch.float64)
    local_batches = torch.zeros((), device=info.device, dtype=torch.float64)
    for _ in range(int(num_batches)):
        x, y = next(val_loader)
        with autocast_context(cfg, info.device):
            out = model(x, targets=y)
        local_loss_sum += float(out["loss"].item())
        local_batches += 1.0
    packed = torch.stack([local_loss_sum, local_batches])
    all_reduce_sum_tensor(packed, info)
    avg_loss = float((packed[0] / packed[1].clamp_min(1.0)).item())
    model.train()
    return avg_loss, math.exp(min(avg_loss, 100.0))


def draw_candidate_batch(loader, candidate_multiplier: int):
    xs, ys, starts = [], [], []
    has_indices = False
    for _ in range(int(candidate_multiplier)):
        batch = next(loader)
        if len(batch) == 3:
            x, y, ix = batch
            starts.append(ix)
            has_indices = True
        else:
            x, y = batch
        xs.append(x)
        ys.append(y)
    cand_x = torch.cat(xs, dim=0)
    cand_y = torch.cat(ys, dim=0)
    cand_starts = torch.cat(starts, dim=0) if has_indices else None
    return cand_x, cand_y, cand_starts


def capture_candidate_factors(model, opt, engine, cfg: dict, cand_x, cand_y, chunk_size: int, device: torch.device):
    actual_b = cand_x.size(0)
    if chunk_size <= 0:
        chunk_size = actual_b
    if actual_b % chunk_size != 0:
        for cs in range(min(chunk_size, actual_b), 0, -1):
            if actual_b % cs == 0:
                chunk_size = cs
                break

    opt.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    engine.start_capture()
    chunked_a, chunked_b = {}, {}
    chunked_sketches = {}
    loss_scale = float(chunk_size) / float(actual_b)

    for start in range(0, actual_b, chunk_size):
        end = start + chunk_size
        engine._layer_activations.clear()
        engine._layer_backprops.clear()
        with autocast_context(cfg, device):
            out = model(cand_x[start:end], targets=cand_y[start:end])
        (out["loss"] * loss_scale).backward()
        if engine.config.use_countsketch:
            sketches = engine.build_candidate_sketches(
                engine._layer_activations,
                engine._layer_backprops,
            )
            for name, sketch in sketches.items():
                chunked_sketches.setdefault(name, []).append(sketch.detach())
        else:
            for name, tensor in engine._layer_activations.items():
                chunked_a.setdefault(name, []).append(tensor.detach())
            for name, tensor in engine._layer_backprops.items():
                chunked_b.setdefault(name, []).append(tensor.detach())
        del out

    engine.stop_capture()
    engine._layer_activations = {}
    engine._layer_backprops = {}
    if engine.config.use_countsketch:
        engine._last_candidate_sketches = {
            name: torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            for name, parts in chunked_sketches.items()
        }
        if not engine._last_candidate_sketches:
            raise RuntimeError("No CountSketch factors were captured for OptiSelect scoring.")
        return {}, {}
    candidate_activations = {name: torch.cat(parts, dim=0) if len(parts) > 1 else parts[0] for name, parts in chunked_a.items()}
    candidate_backprops = {name: torch.cat(parts, dim=0) if len(parts) > 1 else parts[0] for name, parts in chunked_b.items()}
    if not candidate_activations:
        raise RuntimeError("No Ghost factors were captured for OptiSelect scoring.")
    return candidate_activations, candidate_backprops


def ddp_backward(model_for_backward, loss, micro_idx: int, grad_accum: int, info: DistInfo):
    if info.world_size > 1 and micro_idx < grad_accum - 1:
        with model_for_backward.no_sync():
            loss.backward()
    else:
        loss.backward()


def select_global_candidates(
    engine: OptiSelectEngine,
    local_scores: torch.Tensor,
    cand_x: torch.Tensor,
    cand_y: torch.Tensor,
    cand_starts: Optional[torch.Tensor],
    n_select_global: int,
    eta: float,
    lambda_r: float,
    info: DistInfo,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not engine.config.use_countsketch:
        raise RuntimeError("DDP global OptiSelect currently requires CountSketch features.")
    local_sketches = engine._last_candidate_sketches
    if local_sketches is None:
        raise RuntimeError("Candidate sketches missing; call score_candidates first.")

    gathered_scores = all_gather_equal(local_scores, info)
    gathered_x = all_gather_equal(cand_x, info)
    gathered_y = all_gather_equal(cand_y, info)
    gathered_starts = all_gather_equal(cand_starts, info) if cand_starts is not None else None
    gathered_sketches = {name: all_gather_equal(tensor, info) for name, tensor in local_sketches.items()}

    selected_global = None
    if info.is_master:
        selected_global = engine._select_batch_from_sketches(
            gathered_sketches,
            alignment_scores=gathered_scores,
            n_select=n_select_global,
            eta=eta,
            lambda_r=lambda_r,
        )
    selected_global = broadcast_long_tensor(selected_global, (n_select_global,), info)

    local_batch = n_select_global // info.world_size
    start = info.rank * local_batch
    selected_local = selected_global[start : start + local_batch]
    return (
        gathered_x[selected_local].detach(),
        gathered_y[selected_local].detach(),
        selected_global.detach() if selected_global is not None else None,
        gathered_starts.detach() if gathered_starts is not None else None,
    )


def doc_scores_for_token_starts(
    token_starts: np.ndarray,
    doc_starts: np.ndarray,
    doc_scores: np.ndarray,
) -> np.ndarray:
    doc_idxs = np.searchsorted(doc_starts, token_starts.astype(np.int64), side="right") - 1
    doc_idxs = np.clip(doc_idxs, 0, len(doc_scores) - 1)
    return doc_scores[doc_idxs]


def selected_doc_score_stats(
    candidate_starts: torch.Tensor,
    selected_idx: torch.Tensor,
    doc_starts: np.ndarray,
    doc_scores: np.ndarray,
) -> Optional[dict]:
    if candidate_starts is None or selected_idx is None or len(doc_scores) == 0:
        return None
    cand_starts_np = candidate_starts.detach().cpu().numpy()
    selected_idx_np = selected_idx.detach().cpu().numpy()
    cand_scores = doc_scores_for_token_starts(cand_starts_np, doc_starts, doc_scores)
    selected_scores = cand_scores[selected_idx_np]
    valid_cand = np.isfinite(cand_scores)
    valid_selected = np.isfinite(selected_scores)
    if not valid_cand.any() or not valid_selected.any():
        return None
    valid_candidate_scores = cand_scores[valid_cand]
    selected_percentiles = np.asarray([
        np.mean(valid_candidate_scores <= score)
        for score in selected_scores[valid_selected]
    ], dtype=np.float32)
    return {
        "candidate_edu_score_mean": float(np.nanmean(cand_scores)),
        "selected_edu_score_mean": float(np.nanmean(selected_scores)),
        "selected_edu_score_percentile": float(np.nanmean(selected_percentiles)),
    }


def summarize_doc_score_stats(stats: List[dict]) -> Optional[dict]:
    if not stats:
        return None
    keys = sorted({key for item in stats for key in item})
    summary = {"num_selection_batches": int(len(stats))}
    for key in keys:
        values = [float(item[key]) for item in stats if key in item and math.isfinite(float(item[key]))]
        if values:
            summary[f"mean_{key}"] = float(sum(values) / len(values))
    return summary


def run_experiment(
    run_key: str,
    cfg: dict,
    info: DistInfo,
    train_data,
    val_data,
    train_doc_lens,
    train_doc_scores,
    val_doc_lens,
    bos_token_id: int,
    vocab_size: int,
    data_metadata: dict,
    results_dir: Path,
):
    run_cfg = cfg["runs"][run_key]
    opt_key = run_cfg["optimizer"]
    opt_cfg = cfg["optimizers"][opt_key]
    optiselect = bool(run_cfg["optiselect"])
    seq_len = int(cfg["model"]["max_seq_len"])
    global_batch = int(cfg["training"]["device_batch_size"])
    local_batch = global_batch // info.world_size
    grad_accum = planned_grad_accum_steps(cfg)
    effective_batch = grad_accum * global_batch * seq_len
    iterations = resolve_run_iterations(optiselect, cfg)
    if iterations is None:
        raise ValueError("DDP runner requires a finite iteration count.")
    eval_interval = int(cfg["training"]["eval_interval"])
    eval_batches = resolve_eval_batches(cfg, final=False)
    final_eval_batches = resolve_eval_batches(cfg, final=True)
    log_interval = int(cfg["training"]["log_interval"])
    grad_clip = float(cfg["training"]["grad_clip"])

    seed_everything(int(cfg["training"]["seed"]))
    raw_model, model_cfg = build_model(cfg, vocab_size, info.device)
    opt, repo_opt_name = build_optimizer(raw_model, opt_key, cfg, resolved_iterations=iterations)

    use_compile = bool(cfg["model"].get("use_compile", False))
    compiled_model = raw_model
    if use_compile and hasattr(torch, "compile") and not optiselect and info.world_size == 1:
        rank_print(info, "Compiling model for standard run...")
        compiled_model = torch.compile(raw_model)
    elif use_compile and optiselect:
        rank_print(info, "torch.compile disabled for OptiSelect so Ghost hooks can see Linear modules.")

    ddp_model = (
        DDP(raw_model, device_ids=[info.local_rank], broadcast_buffers=False)
        if info.world_size > 1
        else compiled_model
    )
    eval_model = ddp_model if info.world_size == 1 else raw_model

    num_params = sum(p.numel() for p in raw_model.parameters())
    train_loader = make_sharded_dataloader(
        train_data,
        global_batch,
        seq_len,
        info.device,
        int(cfg["training"]["seed"]),
        info,
        return_indices=optiselect,
    )
    val_loader = make_sharded_dataloader(val_data, global_batch, seq_len, info.device, int(cfg["training"]["seed"]) + 1, info)

    engine = None
    proxy_meta = None
    if optiselect:
        sel_cfg = cfg["selection"]
        engine = OptiSelectEngine(
            model=raw_model,
            optimizer=opt,
            opt_name=repo_opt_name,
            config=InfluenceConfig(
                sketch_dim=int(sel_cfg["sketch_dim"]),
                temperature=float(sel_cfg["temperature"]),
                redundancy_weight=float(sel_cfg["redundancy_weight"]),
                use_countsketch=bool(sel_cfg.get("use_countsketch", True)),
                countsketch_row_block=int(sel_cfg.get("countsketch_row_block", 32)),
                countsketch_token_block=int(sel_cfg.get("countsketch_token_block", 128)),
            ),
            device=str(info.device),
        )
        proxy_batches, proxy_meta = build_validation_proxy(val_data, val_doc_lens, cfg, info, info.device, bos_token_id, refresh_index=0)
        proxy_desc = proxy_meta.get("source", cfg["selection"].get("proxy_source", "dclm_edu_heldout"))
        if proxy_meta.get("tasks"):
            proxy_desc += f" [{', '.join(proxy_meta['tasks'])}]"
        rank_print(info, f"Computing OptiSelect proxy factors from {sum(b[0].size(0) for b in proxy_batches)} docs in {len(proxy_batches)} batches: {proxy_desc}")
        engine.compute_validation_gradient_factors(raw_model, proxy_batches, autocast_context(cfg, info.device))

    history = {"steps": [], "train_loss": [], "val_steps": [], "val_loss": [], "val_ppl": []}
    selection_doc_score_stats: List[dict] = []
    train_doc_starts = np.concatenate([[0], np.cumsum(train_doc_lens)[:-1]]).astype(np.int64)
    smooth_loss = 0.0
    ema_beta = 0.9
    total_training_time = 0.0
    wall_start = time.time()
    warmup_timing_steps = 5

    rank_print(info, "\n" + "=" * 78)
    rank_print(info, f"Run: {run_key} | optimizer={opt_key} | mode={'OptiSelect' if optiselect else 'standard'} | DDP world={info.world_size}")
    rank_print(info, f"Global batch={effective_batch:,} tokens/update, local microbatch={local_batch} examples/rank")
    rank_print(info, f"Params={num_params:,} ({num_params/1e6:.1f}M), updates={iterations:,}")
    if optiselect:
        cm = int(cfg["selection"]["candidate_multiplier"])
        rank_print(info, f"OptiSelect CountSketch={bool(cfg['selection'].get('use_countsketch', True))}, m={cfg['selection']['sketch_dim']}, candidate_multiplier={cm}, token_block={cfg['selection'].get('countsketch_token_block', 128)}")
    rank_print(info, f"Primary LR={scaled_lr(opt_key, cfg):.3e}, repo_opt_name={repo_opt_name}")
    rank_print(info, "=" * 78)

    for step_idx in range(int(iterations)):
        torch.cuda.synchronize(info.device)
        t0 = time.time()
        progress = step_idx / max(1, int(iterations))
        apply_lr_schedule(opt, get_lr_multiplier(progress, cfg))
        hessian_batch = None

        if optiselect:
            selected_batches = []
            sel_cfg = cfg["selection"]
            if step_idx > 0 and int(sel_cfg["val_proxy_refresh"]) > 0 and step_idx % int(sel_cfg["val_proxy_refresh"]) == 0:
                refresh_index = step_idx // int(sel_cfg["val_proxy_refresh"])
                proxy_batches, proxy_meta = build_validation_proxy(val_data, val_doc_lens, cfg, info, info.device, bos_token_id, refresh_index=refresh_index)
                engine.compute_validation_gradient_factors(raw_model, proxy_batches, autocast_context(cfg, info.device))

            for _ in range(grad_accum):
                cand_x, cand_y, cand_starts = draw_candidate_batch(train_loader, int(sel_cfg["candidate_multiplier"]))
                local_chunk = int(sel_cfg.get("candidate_chunk_size", 0)) or local_batch
                candidate_activations, candidate_backprops = capture_candidate_factors(raw_model, opt, engine, cfg, cand_x, cand_y, local_chunk, info.device)
                with torch.no_grad():
                    scores = engine.score_candidates(candidate_activations, candidate_backprops)
                    sel_x, sel_y, selected_global_idx, gathered_starts = select_global_candidates(
                        engine,
                        scores,
                        cand_x,
                        cand_y,
                        cand_starts,
                        n_select_global=global_batch,
                        eta=primary_lr(opt),
                        lambda_r=float(sel_cfg["redundancy_weight"]),
                        info=info,
                    )
                    if info.is_master and gathered_starts is not None:
                        doc_stats = selected_doc_score_stats(
                            gathered_starts,
                            selected_global_idx,
                            train_doc_starts,
                            train_doc_scores,
                        )
                        if doc_stats is not None:
                            selection_doc_score_stats.append(doc_stats)
                selected_batches.append((sel_x, sel_y))
                del candidate_activations, candidate_backprops, scores

            opt.zero_grad(set_to_none=True)
            train_loss_value = 0.0
            for micro_idx, (sel_x, sel_y) in enumerate(selected_batches):
                with autocast_context(cfg, info.device):
                    out = ddp_model(sel_x, targets=sel_y)
                loss = out["loss"] / grad_accum
                ddp_backward(ddp_model, loss, micro_idx, grad_accum, info)
                train_loss_value += float(out["loss"].detach().item()) / grad_accum
                hessian_batch = (sel_x, sel_y)
        else:
            opt.zero_grad(set_to_none=True)
            train_loss_value = 0.0
            for micro_idx in range(grad_accum):
                x, y = next(train_loader)
                with autocast_context(cfg, info.device):
                    out = ddp_model(x, targets=y)
                loss = out["loss"] / grad_accum
                ddp_backward(ddp_model, loss, micro_idx, grad_accum, info)
                train_loss_value += float(out["loss"].detach().item()) / grad_accum
                hessian_batch = (x, y)

        train_loss_value = all_reduce_mean(train_loss_value, info.device, info)
        optimizer_step(ddp_model, raw_model, opt, opt_key, opt_cfg, step_idx, grad_clip, hessian_batch, seq_len, cfg, info)

        if math.isnan(train_loss_value) or train_loss_value > 100:
            raise RuntimeError(f"{run_key} diverged at step {step_idx}: loss={train_loss_value:.4f}")

        torch.cuda.synchronize(info.device)
        dt = time.time() - t0
        if step_idx > warmup_timing_steps:
            total_training_time += dt
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss_value
        debiased = smooth_loss / (1 - ema_beta ** (step_idx + 1))
        if info.is_master:
            history["steps"].append(step_idx)
            history["train_loss"].append(debiased)

        if (step_idx + 1) % eval_interval == 0:
            val_loss, val_ppl = evaluate(eval_model, val_loader, eval_batches, cfg, info)
            if info.is_master:
                history["val_steps"].append(step_idx + 1)
                history["val_loss"].append(val_loss)
                history["val_ppl"].append(val_ppl)
                tok_per_sec = int(effective_batch / max(dt, 1e-9))
                print(
                    f"{run_key:20s} step={step_idx+1:05d} train={debiased:.4f} "
                    f"val={val_loss:.4f} ppl={val_ppl:.1f} lr={primary_lr(opt):.2e} global_tok/s={tok_per_sec:,}",
                    flush=True,
                )
        elif log_interval and (step_idx + 1) % log_interval == 0:
            rank_print(info, f"{run_key:20s} step={step_idx+1:05d} train={debiased:.4f} lr={primary_lr(opt):.2e}")

    final_loss, final_ppl = evaluate(eval_model, val_loader, final_eval_batches, cfg, info)
    peak_vram = torch.cuda.max_memory_allocated(info.device) / 1024**3
    peak_vram_t = torch.tensor(peak_vram, device=info.device, dtype=torch.float32)
    if info.world_size > 1:
        dist.all_reduce(peak_vram_t, op=dist.ReduceOp.MAX)
    peak_vram = float(peak_vram_t.item())

    actual_update_tokens = int(iterations * effective_batch)
    candidate_multiplier = int(cfg["selection"]["candidate_multiplier"]) if optiselect else 1
    actual_candidate_tokens = int(iterations * effective_batch * candidate_multiplier) if optiselect else 0

    result = {
        "run_key": run_key,
        "optimizer": opt_key,
        "repo_opt_name": repo_opt_name,
        "optiselect": optiselect,
        "distributed": {
            "enabled": info.world_size > 1,
            "world_size": int(info.world_size),
            "global_device_batch_size": int(global_batch),
            "local_device_batch_size": int(local_batch),
            "grad_accum_steps": int(grad_accum),
        },
        "model_config": asdict(model_cfg),
        "num_params": int(num_params),
        "steps": int(iterations),
        "effective_batch_tokens": int(effective_batch),
        "effective_batch_examples": int(update_batch_examples(cfg)),
        "target_update_tokens": int(target_update_tokens_for_run(optiselect, cfg)),
        "actual_update_tokens": actual_update_tokens,
        "actual_candidate_tokens": actual_candidate_tokens,
        "actual_processed_tokens": int(actual_update_tokens + actual_candidate_tokens),
        "eval_batches": int(eval_batches),
        "eval_tokens": int(eval_batches * effective_batch_tokens(cfg)),
        "final_eval_batches": int(final_eval_batches),
        "final_eval_tokens": int(final_eval_batches * effective_batch_tokens(cfg)),
        "training_time_sec": float(total_training_time),
        "wall_time_sec": float(time.time() - wall_start),
        "final_val_loss": float(final_loss),
        "final_val_ppl": float(final_ppl),
        "peak_vram_gb": float(peak_vram),
        "history": history,
        "selection_summary": engine.get_selection_summary() if engine is not None else None,
        "selection_doc_score_summary": summarize_doc_score_stats(selection_doc_score_stats) if optiselect else None,
        "optimizer_config": opt_cfg,
        "dataset_metadata": data_metadata,
        "selection_proxy": {
            "source": cfg["selection"].get("proxy_source", "dclm_edu_heldout"),
            "unit": cfg["selection"].get("proxy_unit", "documents"),
            "tasks": cfg["selection"].get("proxy_tasks") if cfg["selection"].get("proxy_source") == "downstream" else None,
            "last_proxy_counts": proxy_meta.get("counts") if isinstance(proxy_meta, dict) else None,
            "last_proxy_real_target_tokens": proxy_meta.get("real_target_tokens") if isinstance(proxy_meta, dict) else None,
            "val_proxy_size": int(cfg["selection"]["val_proxy_size"]),
            "val_proxy_refresh": int(cfg["selection"]["val_proxy_refresh"]),
            "use_countsketch": bool(cfg["selection"].get("use_countsketch", True)),
            "sketch_dim": int(cfg["selection"]["sketch_dim"]),
            "countsketch_token_block": int(cfg["selection"].get("countsketch_token_block", 128)),
            "global_selection": True,
        } if optiselect else None,
    }

    if info.is_master:
        run_dir = results_dir / run_key
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(result, indent=2))
        print(f"FINAL {run_key}: val_loss={final_loss:.4f}, ppl={final_ppl:.1f}, steps={iterations}", flush=True)

    if engine is not None:
        engine.detach()
    del ddp_model, raw_model, opt
    torch.cuda.empty_cache()
    gc.collect()
    barrier(info)
    return result if info.is_master else None


def main():
    args = parse_args()
    info = init_distributed()
    cfg, config_path, results_dir = load_cfg(args, info)
    validate_cfg(cfg, info)

    seed_everything(int(cfg["training"]["seed"]))
    Path(results_dir).mkdir(parents=True, exist_ok=True) if info.is_master else None
    barrier(info)

    rank_print(info, f"Repo: {find_repo_root()}")
    rank_print(info, f"Config: {config_path}")
    rank_print(info, f"Results: {results_dir}")
    rank_print(info, f"DDP world_size={info.world_size}; run_keys={cfg['experiment']['run_keys']}")
    rank_print(info, f"Global device_batch_size={cfg['training']['device_batch_size']} -> local={int(cfg['training']['device_batch_size']) // info.world_size}")
    rank_print(info, "Selection proxy:")
    rank_print(info, f"  source={cfg['selection'].get('proxy_source')} size={cfg['selection']['val_proxy_size']} {cfg['selection'].get('proxy_unit', 'documents')}")
    if cfg["selection"].get("proxy_source") == "downstream":
        rank_print(info, "  tasks=" + ", ".join(cfg["selection"].get("proxy_tasks", [])))
    rank_print(info, f"  refresh={cfg['selection']['val_proxy_refresh']} steps, candidate_multiplier={cfg['selection']['candidate_multiplier']}")
    rank_print(
        info,
        "  "
        f"use_countsketch={bool(cfg['selection'].get('use_countsketch', True))}, "
        f"sketch_dim={cfg['selection']['sketch_dim']}, "
        f"token_block={cfg['selection'].get('countsketch_token_block', 128)}, "
        f"candidate_chunk_size={cfg['selection'].get('candidate_chunk_size')}",
    )

    train_data, val_data, train_lens, val_lens, train_scores, _val_scores, bos_token_id, vocab_size, data_metadata = load_dclm_data(cfg, info)
    all_results = {}
    for run_key in cfg["experiment"]["run_keys"]:
        result = run_experiment(
            run_key,
            copy.deepcopy(cfg),
            info,
            train_data,
            val_data,
            train_lens,
            train_scores,
            val_lens,
            bos_token_id,
            vocab_size,
            data_metadata,
            Path(results_dir),
        )
        if info.is_master and result is not None:
            all_results[run_key] = result

    if info.is_master:
        summary_name = args.summary_name or "demo_dclm_ddp_summary.json"
        summary_path = Path(results_dir) / summary_name
        summary_path.write_text(json.dumps(all_results, indent=2))
        print(f"Wrote sweep summary: {summary_path}", flush=True)
    barrier(info)
    if info.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
