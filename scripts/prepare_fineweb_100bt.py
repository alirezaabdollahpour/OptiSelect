#!/usr/bin/env python3
"""Download pre-tokenized FineWeb100B GPT-2 tokens for this benchmark.

The training code in src/data expects two raw uint16 memmap files:

    <datasets_dir>/fineweb-100BT/train.bin
    <datasets_dir>/fineweb-100BT/val.bin

Instead of downloading HuggingFaceFW/fineweb sample-100BT and tokenizing
locally, this script downloads the already-tokenized GPT-2 chunks from
kjj0/fineweb100B-gpt2 and assembles them into the expected files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from huggingface_hub import snapshot_download
from tqdm import tqdm


SOURCE_REPO = "kjj0/fineweb100B-gpt2"
SOURCE_REPO_TYPE = "dataset"
DEFAULT_NUM_CHUNKS = 1028
TRAIN_CHUNK_TOKENS = 100_000_000
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype(np.int32).itemsize
HEADER_MAGIC = 20240520
HEADER_VERSION = 1
UINT16_BYTES = np.dtype(np.uint16).itemsize
EXPECTED_CHUNK_BYTES = TRAIN_CHUNK_TOKENS * UINT16_BYTES
MIN_RECOMMENDED_FREE_BYTES = 430 * 1024**3


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_datasets_dir = repo_root.parent / "datasets"

    parser = argparse.ArgumentParser(
        description=(
            "Download kjj0/fineweb100B-gpt2 chunks and assemble "
            "datasets/fineweb-100BT/{train,val}.bin."
        )
    )
    parser.add_argument(
        "num_chunks_pos",
        nargs="?",
        type=int,
        help=(
            "Optional positional train chunk count, matching the common "
            "download script style. Overrides --num_chunks if provided."
        ),
    )
    parser.add_argument(
        "--datasets_dir",
        type=Path,
        default=default_datasets_dir,
        help=(
            "Base dataset directory. Output goes to "
            "<datasets_dir>/fineweb-100BT."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override the exact output directory. Usually leave unset.",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help="Number of 100M-token training chunks to download and assemble.",
    )
    parser.add_argument(
        "--download_workers",
        type=int,
        default=16,
        help="Parallel workers used by huggingface_hub.snapshot_download.",
    )
    parser.add_argument(
        "--hf_transfer",
        action="store_true",
        help=(
            "Enable HF_HUB_ENABLE_HF_TRANSFER=1. Requires installing "
            "huggingface_hub[hf_transfer] or hf_transfer."
        ),
    )
    parser.add_argument(
        "--keep_chunks",
        action="store_true",
        default=True,
        help="Keep downloaded chunk files after assembling. Default: true.",
    )
    parser.add_argument(
        "--remove_chunks_after_assemble",
        dest="keep_chunks",
        action="store_false",
        help="Remove downloaded chunk files after train.bin/val.bin are ready.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing final/temp files before assembling.",
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only download chunks; do not assemble train.bin/val.bin.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Use existing files in <output_dir>/chunks and only validate/assemble.",
    )
    parser.add_argument(
        "--force_redownload",
        action="store_true",
        help="Force huggingface_hub to re-download files it would otherwise reuse.",
    )
    return parser.parse_args()


def output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    return args.datasets_dir.expanduser().resolve() / "fineweb-100BT"


def train_chunk_name(index: int) -> str:
    return f"fineweb_train_{index:06d}.bin"


def val_chunk_name() -> str:
    return "fineweb_val_000000.bin"


def requested_filenames(num_chunks: int) -> List[str]:
    return [val_chunk_name()] + [train_chunk_name(i) for i in range(1, num_chunks + 1)]


def ensure_enough_space(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < MIN_RECOMMENDED_FREE_BYTES:
        print(
            "WARNING: free space under "
            f"{path} is {free / 1024**3:.1f} GiB. Keeping chunks and assembling "
            "a contiguous train.bin may use roughly 410 GiB for the full 100B "
            "download.",
            file=sys.stderr,
        )


def remove_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def bytes_to_tokens(path: Path) -> int:
    return path.stat().st_size // UINT16_BYTES


def validate_chunk(path: Path, *, expected_tokens: int | None = None) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Missing downloaded chunk: {path}")
    size = path.stat().st_size
    if size < HEADER_BYTES:
        raise ValueError(f"{path} is too small to contain the {HEADER_BYTES}-byte header")

    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    if int(header[0]) != HEADER_MAGIC or int(header[1]) != HEADER_VERSION:
        raise ValueError(
            f"{path} has unexpected header magic/version "
            f"{int(header[0])}/{int(header[1])}"
        )

    payload_tokens = int(header[2])
    payload_bytes = payload_tokens * UINT16_BYTES
    expected_size = HEADER_BYTES + payload_bytes
    if size != expected_size:
        raise ValueError(
            f"{path} is {size} bytes, expected {expected_size} bytes from "
            f"header token count {payload_tokens:,}"
        )

    if expected_tokens is not None and payload_tokens != expected_tokens:
        raise ValueError(
            f"{path} contains {payload_tokens:,} tokens, expected "
            f"{expected_tokens:,}"
        )
    if payload_tokens <= 0:
        raise ValueError(f"{path} reports non-positive token count {payload_tokens}")
    return payload_tokens


def validate_final_train(path: Path, expected_tokens: int) -> None:
    expected_size = expected_tokens * UINT16_BYTES
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path} is {actual_size} bytes, expected {expected_size} bytes for "
            f"{expected_tokens:,} tokens. Use --overwrite to rebuild it."
        )


def download_chunks(args: argparse.Namespace, chunks_dir: Path, filenames: List[str]) -> Path:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if args.hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print(
        f"Downloading {len(filenames)} files from {SOURCE_REPO} into {chunks_dir} "
        f"with {args.download_workers} workers"
    )
    return Path(
        snapshot_download(
            repo_id=SOURCE_REPO,
            repo_type=SOURCE_REPO_TYPE,
            local_dir=chunks_dir,
            allow_patterns=filenames,
            max_workers=args.download_workers,
            force_download=args.force_redownload,
        )
    )


def copy_payload(src: Path, dst: Path) -> int:
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    remove_if_exists(tmp_path)
    validate_chunk(src)
    with src.open("rb") as src_file, tmp_path.open("wb") as dst_file:
        src_file.seek(HEADER_BYTES)
        shutil.copyfileobj(src_file, dst_file, length=64 * 1024 * 1024)
    tmp_path.replace(dst)
    return bytes_to_tokens(dst)


def existing_complete_chunks(train_tmp_path: Path, chunks: List[Path]) -> int:
    if not train_tmp_path.exists():
        return 0
    size = train_tmp_path.stat().st_size
    if size % EXPECTED_CHUNK_BYTES != 0:
        raise ValueError(
            f"Partial assembly file {train_tmp_path} has size {size}, which is "
            f"not a whole number of {EXPECTED_CHUNK_BYTES}-byte chunks. Delete "
            "it or run with --overwrite."
        )
    complete = size // EXPECTED_CHUNK_BYTES
    if complete > len(chunks):
        raise ValueError(
            f"Partial assembly file {train_tmp_path} contains {complete} chunks, "
            f"but only {len(chunks)} were requested."
        )
    return int(complete)


def append_chunks(
    chunks: List[Path],
    train_path: Path,
    overwrite: bool,
    payload_tokens: Dict[Path, int],
) -> int:
    train_tmp_path = train_path.with_suffix(train_path.suffix + ".tmp")
    if overwrite:
        remove_if_exists(train_tmp_path)

    start_index = existing_complete_chunks(train_tmp_path, chunks)
    if start_index:
        print(f"Resuming train.bin assembly after {start_index:,} chunks")

    mode = "ab" if start_index else "wb"
    with train_tmp_path.open(mode) as out_file:
        for chunk_path in tqdm(
            chunks[start_index:],
            initial=start_index,
            total=len(chunks),
            unit="chunk",
            desc="Assembling train.bin",
            dynamic_ncols=True,
        ):
            validate_chunk(chunk_path, expected_tokens=payload_tokens[chunk_path])
            with chunk_path.open("rb") as in_file:
                in_file.seek(HEADER_BYTES)
                shutil.copyfileobj(in_file, out_file, length=64 * 1024 * 1024)
            out_file.flush()

    train_tmp_path.replace(train_path)
    return bytes_to_tokens(train_path)


def validate_downloaded_chunks(chunks_dir: Path, num_chunks: int) -> Dict[Path, int]:
    val_path = chunks_dir / val_chunk_name()
    validate_chunk(val_path)

    train_chunks = [chunks_dir / train_chunk_name(i) for i in range(1, num_chunks + 1)]
    payload_tokens = {}
    for index, chunk_path in tqdm(
        list(enumerate(train_chunks, start=1)),
        unit="chunk",
        desc="Validating chunks",
        dynamic_ncols=True,
    ):
        expected_tokens = TRAIN_CHUNK_TOKENS
        if index == DEFAULT_NUM_CHUNKS:
            expected_tokens = None
        payload_tokens[chunk_path] = validate_chunk(
            chunk_path,
            expected_tokens=expected_tokens,
        )
    return payload_tokens


def remove_chunks(chunks_dir: Path, filenames: Iterable[str]) -> None:
    for name in filenames:
        remove_if_exists(chunks_dir / name)


def main() -> None:
    args = parse_args()
    if args.num_chunks_pos is not None:
        args.num_chunks = args.num_chunks_pos
    if not 1 <= args.num_chunks <= DEFAULT_NUM_CHUNKS:
        raise ValueError(f"--num_chunks must be in [1, {DEFAULT_NUM_CHUNKS}]")
    if args.download_workers <= 0:
        raise ValueError("--download_workers must be positive")

    out_dir = output_dir_from_args(args)
    chunks_dir = out_dir / "chunks"
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    metadata_path = out_dir / "metadata.json"
    filenames = requested_filenames(args.num_chunks)

    ensure_enough_space(out_dir)

    if args.overwrite:
        for path in (
            train_path,
            val_path,
            train_path.with_suffix(train_path.suffix + ".tmp"),
            val_path.with_suffix(val_path.suffix + ".tmp"),
            metadata_path,
        ):
            remove_if_exists(path)
    elif train_path.exists() and val_path.exists():
        print(f"FineWeb100B GPT-2 files are already prepared in {out_dir}")
        print(f"train tokens: {bytes_to_tokens(train_path):,}")
        print(f"val tokens:   {bytes_to_tokens(val_path):,}")
        return

    if args.skip_download:
        print(f"Skipping download; using existing chunks in {chunks_dir}")
    else:
        download_chunks(args, chunks_dir, filenames)
    payload_tokens = validate_downloaded_chunks(chunks_dir, args.num_chunks)
    train_chunks = list(payload_tokens.keys())
    expected_train_tokens = sum(payload_tokens.values())

    if args.download_only:
        print(f"Downloaded chunks are ready in {chunks_dir}")
        return

    if not val_path.exists() or args.overwrite:
        copy_payload(chunks_dir / val_chunk_name(), val_path)

    if train_path.exists() and not args.overwrite:
        validate_final_train(train_path, expected_train_tokens)
        train_tokens = bytes_to_tokens(train_path)
    else:
        train_tokens = append_chunks(
            train_chunks,
            train_path,
            overwrite=args.overwrite,
            payload_tokens=payload_tokens,
        )

    val_tokens = bytes_to_tokens(val_path)

    metadata = {
        "source_repo": SOURCE_REPO,
        "source_repo_type": SOURCE_REPO_TYPE,
        "tokenizer": "gpt2",
        "dtype": "uint16",
        "num_train_chunks": args.num_chunks,
        "train_chunk_tokens": TRAIN_CHUNK_TOKENS,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_file": str(train_path),
        "val_file": str(val_path),
        "chunks_dir": str(chunks_dir),
        "kept_chunks": args.keep_chunks,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
    }
    atomic_write_json(metadata_path, metadata)

    if not args.keep_chunks:
        remove_chunks(chunks_dir, filenames)

    print("Done.")
    print(f"train.bin: {train_tokens:,} tokens")
    print(f"val.bin:   {val_tokens:,} tokens")
    print(
        "Use with training, for example:\n"
        f"  python src/main.py --config_format base --model llama "
        f"--dataset fineweb --datasets_dir {out_dir.parent}"
    )


if __name__ == "__main__":
    main()
