#!/usr/bin/env bash
# =============================================================================
# DCLM-Edu demo launcher for demo_dclm_ddp.py
# =============================================================================
#
# This is the distributed, non-interactive counterpart to demo.ipynb.
# It never edits demo.ipynb or demo_config.yaml; every override is passed at
# launch time.  The defaults below mirror the current notebook experiment cell;
# set DEMO_RESULTS_DIR= to let demo_dclm_ddp.py create a separate *_ddpN_seedS
# directory from demo_config.yaml instead.
#
# Common local use matching the current demo.ipynb experiment cell:
#   cd /mloscratch/homes/aabdolla/llm-optimizer-benchmark/src
#   DEMO_NPROC=1 bash demo_DCLM.sh
#
# Multi-GPU use with the same notebook settings:
#   DEMO_NPROC=4 bash demo_DCLM.sh
#
# Scheduler-style use:
#   python csub.py -n demo-dclm-signsgd-downstream-ddp4 -g 4 -t 1d --train --large-shm --node-type h100 \
#     --command "cd /mloscratch/homes/aabdolla/llm-optimizer-benchmark/src && \
#       source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate && \
#       DEMO_NPROC=4 \
#       bash demo_DCLM.sh"
#
# Positional fallback:
#   bash demo_DCLM.sh "optiselect_signsgd,signsgd" 42 4
#     $1 run keys, $2 seed, $3 number of GPUs.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${SCRIPT_DIR}"

DEMO_RUN_KEYS="${DEMO_RUN_KEYS:-${1:-optiselect_signsgd,signsgd}}"
DEMO_SEED="${DEMO_SEED:-${2:-42}}"
DEMO_NPROC="${DEMO_NPROC:-${3:-1}}"

DEMO_CONFIG="${DEMO_CONFIG:-${SRC_DIR}/demo_config.yaml}"
DEMO_RESULTS_DIR="${DEMO_RESULTS_DIR-/mloscratch/homes/aabdolla/results/demo_optiselect_signsgd_downstream_proxy_cm2}"
DEMO_SUMMARY_NAME="${DEMO_SUMMARY_NAME-demo_optiselect_signsgd_downstream_proxy_summary.json}"
DEMO_VENV="${DEMO_VENV:-/mloscratch/homes/aabdolla/optiselect/.venv}"

# Training/global-batch knobs.  Defaults come from demo_config.yaml when empty.
DEMO_DEVICE_BATCH_SIZE="${DEMO_DEVICE_BATCH_SIZE:-}"
DEMO_TOTAL_BATCH_TOKENS="${DEMO_TOTAL_BATCH_TOKENS:-}"
DEMO_STANDARD_UPDATE_TOKENS="${DEMO_STANDARD_UPDATE_TOKENS:-}"
DEMO_OPTISELECT_UPDATE_TOKENS="${DEMO_OPTISELECT_UPDATE_TOKENS:-}"
DEMO_ITERATIONS="${DEMO_ITERATIONS:-}"
DEMO_SMOKE_STEPS="${DEMO_SMOKE_STEPS:-}"

# Eval/logging knobs.
DEMO_EVAL_INTERVAL="${DEMO_EVAL_INTERVAL:-}"
DEMO_EVAL_TOKENS="${DEMO_EVAL_TOKENS:-}"
DEMO_FINAL_EVAL_TOKENS="${DEMO_FINAL_EVAL_TOKENS:-}"
DEMO_LOG_INTERVAL="${DEMO_LOG_INTERVAL:-}"

# OptiSelect knobs.  Defaults mirror the current demo.ipynb experiment cell.
DEMO_CANDIDATE_MULTIPLIER="${DEMO_CANDIDATE_MULTIPLIER:-2}"
DEMO_PROXY_SOURCE="${DEMO_PROXY_SOURCE:-downstream}"
DEMO_PROXY_TASKS="${DEMO_PROXY_TASKS:-hellaswag,arc_easy,arc_challenge,openbookqa}"
DEMO_CANDIDATE_CHUNK_SIZE="${DEMO_CANDIDATE_CHUNK_SIZE:-16}"
DEMO_PROXY_BATCH_SIZE="${DEMO_PROXY_BATCH_SIZE:-32}"
DEMO_VAL_PROXY_SIZE="${DEMO_VAL_PROXY_SIZE:-8192}"
DEMO_VAL_PROXY_REFRESH="${DEMO_VAL_PROXY_REFRESH:-}"
DEMO_SKETCH_DIM="${DEMO_SKETCH_DIM:-16384}"
DEMO_COUNTSKETCH_ROW_BLOCK="${DEMO_COUNTSKETCH_ROW_BLOCK:-}"
DEMO_COUNTSKETCH_TOKEN_BLOCK="${DEMO_COUNTSKETCH_TOKEN_BLOCK:-128}"
DEMO_TEMPERATURE="${DEMO_TEMPERATURE:-}"
DEMO_REDUNDANCY_WEIGHT="${DEMO_REDUNDANCY_WEIGHT:-}"
DEMO_USE_COUNTSKETCH="${DEMO_USE_COUNTSKETCH:-1}"

DEMO_NO_COMPILE="${DEMO_NO_COMPILE:-0}"
DEMO_TORCHRUN_STANDALONE="${DEMO_TORCHRUN_STANDALONE:-1}"
DEMO_DRY_RUN="${DEMO_DRY_RUN:-0}"

if ! [[ "${DEMO_NPROC}" =~ ^[0-9]+$ ]] || [ "${DEMO_NPROC}" -lt 1 ]; then
    echo "[FATAL] DEMO_NPROC must be a positive integer, got: ${DEMO_NPROC}" >&2
    exit 2
fi

cd "${SRC_DIR}"

if [ -d "${DEMO_VENV}" ] && [ -f "${DEMO_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${DEMO_VENV}/bin/activate"
fi

export PYTHONPATH="${SRC_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mloscratch/homes/aabdolla/datasets/hf_cache/home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mloscratch/homes/aabdolla/datasets/hf_cache/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mloscratch/homes/aabdolla/datasets/hf_cache/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [ "${DEMO_DRY_RUN}" != "1" ] && [ "${DEMO_NPROC}" -gt 1 ]; then
    VISIBLE_GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
    if [ "${VISIBLE_GPU_COUNT}" -lt "${DEMO_NPROC}" ]; then
        echo "[FATAL] Requested DEMO_NPROC=${DEMO_NPROC}, but only ${VISIBLE_GPU_COUNT} CUDA device(s) are visible." >&2
        exit 3
    fi
fi

python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${DEMO_CONFIG}").read_text())
run_keys = [x.strip() for x in "${DEMO_RUN_KEYS}".split(",") if x.strip()]
missing = [k for k in run_keys if k not in cfg["runs"]]
if missing:
    raise SystemExit(f"[FATAL] Unknown run key(s): {missing}; valid keys are {sorted(cfg['runs'])}")
supported_opts = {"adamw", "d-muon", "sgd", "signsgd", "ademamix", "sophia"}
unsupported = [k for k in run_keys if cfg["runs"][k]["optimizer"] not in supported_opts]
if unsupported:
    raise SystemExit(f"[FATAL] Unsupported optimizer in run key(s): {unsupported}; supported: {sorted(supported_opts)}")
print("[OK] Run keys:", ",".join(run_keys))
PY

ARGS=(
    --config "${DEMO_CONFIG}"
    --run-keys "${DEMO_RUN_KEYS}"
    --seed "${DEMO_SEED}"
)

append_arg() {
    local value="$1"
    local flag="$2"
    if [ -n "${value}" ]; then
        ARGS+=("${flag}" "${value}")
    fi
}

append_arg "${DEMO_RESULTS_DIR}" --results-dir
append_arg "${DEMO_SUMMARY_NAME}" --summary-name
append_arg "${DEMO_DEVICE_BATCH_SIZE}" --device-batch-size
append_arg "${DEMO_TOTAL_BATCH_TOKENS}" --total-batch-tokens
append_arg "${DEMO_STANDARD_UPDATE_TOKENS}" --standard-update-tokens
append_arg "${DEMO_OPTISELECT_UPDATE_TOKENS}" --optiselect-update-tokens
append_arg "${DEMO_ITERATIONS}" --iterations
append_arg "${DEMO_SMOKE_STEPS}" --smoke-steps
append_arg "${DEMO_EVAL_INTERVAL}" --eval-interval
append_arg "${DEMO_EVAL_TOKENS}" --eval-tokens
append_arg "${DEMO_FINAL_EVAL_TOKENS}" --final-eval-tokens
append_arg "${DEMO_LOG_INTERVAL}" --log-interval
append_arg "${DEMO_CANDIDATE_MULTIPLIER}" --candidate-multiplier
append_arg "${DEMO_PROXY_SOURCE}" --proxy-source
append_arg "${DEMO_PROXY_TASKS}" --proxy-tasks
append_arg "${DEMO_CANDIDATE_CHUNK_SIZE}" --candidate-chunk-size
append_arg "${DEMO_PROXY_BATCH_SIZE}" --proxy-batch-size
append_arg "${DEMO_VAL_PROXY_SIZE}" --val-proxy-size
append_arg "${DEMO_VAL_PROXY_REFRESH}" --val-proxy-refresh
append_arg "${DEMO_SKETCH_DIM}" --sketch-dim
append_arg "${DEMO_COUNTSKETCH_ROW_BLOCK}" --countsketch-row-block
append_arg "${DEMO_COUNTSKETCH_TOKEN_BLOCK}" --countsketch-token-block
append_arg "${DEMO_TEMPERATURE}" --temperature
append_arg "${DEMO_REDUNDANCY_WEIGHT}" --redundancy-weight
append_arg "${DEMO_USE_COUNTSKETCH}" --use-countsketch

if [ "${DEMO_NO_COMPILE}" = "1" ]; then
    ARGS+=(--no-compile)
fi

echo "================================================================"
echo "  DCLM-Edu demo DDP launch"
echo "  Repo:        ${REPO_ROOT}"
echo "  Config:      ${DEMO_CONFIG}"
echo "  Run keys:    ${DEMO_RUN_KEYS}"
echo "  Seed:        ${DEMO_SEED}"
echo "  GPUs:        ${DEMO_NPROC}"
echo "  Results dir: ${DEMO_RESULTS_DIR:-<auto _ddp${DEMO_NPROC}_seed${DEMO_SEED}>}"
echo "  Summary:     ${DEMO_SUMMARY_NAME:-<default demo_dclm_ddp_summary.json>}"
if [ -n "${DEMO_PROXY_SOURCE}" ]; then
    echo "  Proxy:       ${DEMO_PROXY_SOURCE}${DEMO_PROXY_TASKS:+ (${DEMO_PROXY_TASKS})}"
fi
echo "================================================================"

if [ "${DEMO_DRY_RUN}" = "1" ]; then
    printf 'torchrun'
    if [ "${DEMO_TORCHRUN_STANDALONE}" = "1" ]; then
        printf ' --standalone --nnodes=1'
    else
        printf ' --nnodes=1'
    fi
    printf ' --nproc_per_node=%q demo_dclm_ddp.py' "${DEMO_NPROC}"
    for arg in "${ARGS[@]}"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
    exit 0
fi

if [ "${DEMO_TORCHRUN_STANDALONE}" = "1" ]; then
    exec torchrun --standalone --nnodes=1 --nproc_per_node="${DEMO_NPROC}" demo_dclm_ddp.py "${ARGS[@]}"
else
    exec torchrun --nnodes=1 --nproc_per_node="${DEMO_NPROC}" demo_dclm_ddp.py "${ARGS[@]}"
fi
