import os
import subprocess
import sys
from pathlib import Path


def get_fineweb_data(datasets_dir, num_proc=40):
    """Return paths to the tokenized FineWeb100B GPT-2 memmaps.

    If the files are missing, prepare them from kjj0/fineweb100B-gpt2, which
    already contains GPT-2 uint16 token chunks. This avoids the slow raw
    FineWeb download + local tokenization path.
    """
    del num_proc

    fweb_data_path = Path(datasets_dir) / "fineweb-100BT"
    train_path = fweb_data_path / "train.bin"
    val_path = fweb_data_path / "val.bin"

    if not (train_path.exists() and val_path.exists()):
        repo_root = Path(__file__).resolve().parents[2]
        prepare_script = repo_root / "scripts" / "prepare_fineweb_100bt.py"
        if not prepare_script.exists():
            raise FileNotFoundError(
                f"Missing FineWeb files in {fweb_data_path} and could not find "
                f"the preparation script at {prepare_script}"
            )

        subprocess.check_call(
            [
                sys.executable,
                str(prepare_script),
                "--datasets_dir",
                os.fspath(Path(datasets_dir)),
            ]
        )

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"FineWeb preparation did not produce both {train_path} and {val_path}"
        )

    return {
        "train": os.fspath(train_path),
        "val": os.fspath(val_path),
    }


if __name__ == "__main__":
    get_fineweb_data("./datasets/")
