#!/bin/bash
# =============================================================================
# OptiSelect FineWeb100B 720M Benchmark Dispatcher
# =============================================================================
#
# FineWeb100B:
#   Tokenized GPT-2 uint16 memmaps under:
#     /mloscratch/homes/aabdolla/datasets/fineweb-100BT/{train,val}.bin
#
# Model:
#   Llama, n_embd=2048, n_head=16, n_layer=12, seq_len=512.
#   Parameter count from the local model definition:
#     vocab*d + layers*(4*d*d + 3*d*mlp_hidden + 2*d) + final_norm
#     d=2048, vocab=50304, mlp_hidden=5632 -> 719,685,632 params.
#
# Protocol:
#   Standard:  48,000 optimizer steps x 1,984 examples x 512 tokens
#              = 48.758B update tokens.
#   Selection: same optimizer steps and selected/update tokens. Each step also
#              scores 2B candidates, so compute-tracked processed tokens are
#              3x the standard update-token count.
#
# Primary plots supported by the generated metadata:
#   1. validation loss vs update_tokens_B
#   2. downstream accuracy vs estimated_training_flops
#
# Launch examples:
#   cd /mloscratch/homes/aabdolla/llm-optimizer-benchmark/src
#   bash run_fineweb100b_720m_split.sh 0 0 8
#
# Parallel split launch from your scheduler wrapper:
#   for i in 1 2 3 4 5; do
#     python csub.py -n fineweb720m-$i -g 8 -t 7d --train \
#       --command "cd /mloscratch/homes/aabdolla/llm-optimizer-benchmark/src && \
#         source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate && \
#         export PYTHONPATH=/mloscratch/homes/aabdolla/GhostSuite:/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src:\$PYTHONPATH && \
#         bash run_fineweb100b_720m_split.sh \$i 0 8"
#   done
#
# Optional downstream eval after a split finishes:
#   RUN_DOWNSTREAM=1 bash run_fineweb100b_720m_split.sh 1 0 8
#
# Optional Hugging Face Hub tracking:
#   huggingface-cli login
#   HF_TRACK=1 bash run_fineweb100b_720m_split.sh 1 0 8
#
# By default this uploads summaries, logs, curve JSON, downstream JSON, and a
# run card to this private dataset repo:
#   AlirezaAbdollahpoorrostam/fineweb100b-720m-runs
# Set HF_UPLOAD_CHECKPOINTS=1 only if you intentionally want to push large
# checkpoint folders.
#
# Optional Weights & Biases tracking:
#   wandb login
#   WANDB_TRACK=1 bash run_fineweb100b_720m_split.sh 1 0 8
#
# For a full 100B-update-token pass instead of the 720M reference 48.8B-token
# run, override ITERS=98425.
# =============================================================================

# Do NOT use set -e; one failed optimizer should not kill the whole split.

SPLIT=${1:-0}      # 0 = all sequential; 1-5 = parallel split
SEED=${2:-0}
NPROC=${3:-8}     # GPUs per run

SRC_DIR="/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src"
DATASETS_DIR="/mloscratch/homes/aabdolla/datasets"
RESULTS_DIR=${RESULTS_DIR:-"/mloscratch/homes/aabdolla/results/fineweb100b_720m_seed${SEED}"}

PROXY_SOURCE=${PROXY_SOURCE:-train}
PROXY_TASKS=${PROXY_TASKS:-hellaswag,arc_easy,arc_challenge,piqa,sciq}
RUN_DOWNSTREAM=${RUN_DOWNSTREAM:-0}
DOWNSTREAM_TASKS=${DOWNSTREAM_TASKS:-hellaswag,arc_easy,arc_challenge,piqa,sciq}

HF_TRACK=${HF_TRACK:-0}
HF_USERNAME=${HF_USERNAME:-AlirezaAbdollahpoorrostam}
HF_CONTACT_EMAIL=${HF_CONTACT_EMAIL:-alireza.abdollahpoorrostam@epfl.ch}
HF_REPO_ID=${HF_REPO_ID:-"${HF_USERNAME}/fineweb100b-720m-runs"}
HF_REPO_TYPE=${HF_REPO_TYPE:-dataset}
HF_PRIVATE=${HF_PRIVATE:-1}
HF_UPLOAD_CHECKPOINTS=${HF_UPLOAD_CHECKPOINTS:-0}
HF_TRACK_EVERY_RUN=${HF_TRACK_EVERY_RUN:-1}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

WANDB_TRACK=${WANDB_TRACK:-0}
WANDB_PROJECT=${WANDB_PROJECT:-fineweb100b-720m-optimizer-benchmark}
WANDB_ENTITY=${WANDB_ENTITY:-alirezaabdollahpoorrostam-epfl}
WANDB_MODE=${WANDB_MODE:-online}
export WANDB_MODE

cd "$SRC_DIR"
source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate
export PYTHONPATH="/mloscratch/homes/aabdolla/GhostSuite:${SRC_DIR}:$PYTHONPATH"
export HF_HOME=/mloscratch/homes/aabdolla/.hf_cache
export HF_DATASETS_CACHE=/mloscratch/homes/aabdolla/.hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SEED RESULTS_DIR PROXY_SOURCE PROXY_TASKS

mkdir -p "$RESULTS_DIR" logs

if [ ! -f "${DATASETS_DIR}/fineweb-100BT/train.bin" ] || [ ! -f "${DATASETS_DIR}/fineweb-100BT/val.bin" ]; then
    echo "[FATAL] FineWeb100B not found at ${DATASETS_DIR}/fineweb-100BT/{train,val}.bin"
    echo "        Prepare it with: python /mloscratch/homes/aabdolla/llm-optimizer-benchmark/scripts/prepare_fineweb_100bt.py --datasets_dir ${DATASETS_DIR}"
    exit 1
fi

python - << PYTHON_CHECK
import os
import numpy as np
import torch
base = "${DATASETS_DIR}/fineweb-100BT"
for split in ("train", "val"):
    path = os.path.join(base, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    print(f"[OK] FineWeb100B {split}: {len(data):,} tokens ({os.path.getsize(path)/1e9:.1f} GB)")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"[GPU] {torch.cuda.device_count()} visible GPU(s); GPU0={props.name}, {props.total_memory/1e9:.1f} GB")
else:
    print("[WARN] CUDA not available")
PYTHON_CHECK

# ----------------------------------------------------------------------
# 720M reference configuration
# ----------------------------------------------------------------------
N_EMBD=2048
N_HEAD=16
N_LAYER=12
VOCAB_SIZE=50304
PARAM_COUNT=719685632

SEQ_LEN=512
GLOBAL_EXAMPLES_PER_STEP=1984

# Use a factorization that preserves the scripts/720m effective batch
# (62 x 32 = 1,984 examples) while giving 8-GPU DDP a smaller local microbatch:
# with NPROC=8, the DDP adapter turns 16 x 124 into local 8 x 31.
BATCH_SIZE=${BATCH_SIZE:-16}
ACC_STEPS=${ACC_STEPS:-124}

SOPHIA_BATCH_SIZE=${SOPHIA_BATCH_SIZE:-8}
SOPHIA_ACC_STEPS=${SOPHIA_ACC_STEPS:-248}
SOPHIA_BS=$(( SOPHIA_BATCH_SIZE * SOPHIA_ACC_STEPS ))

ITERS=${ITERS:-48000}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
LOG_INTERVAL=${LOG_INTERVAL:-100}
EVAL_BATCHES=${EVAL_BATCHES:-64}
LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-$ITERS}
PERMANENT_CKPT_INTERVAL=${PERMANENT_CKPT_INTERVAL:-0}

CAND_MULT=${CAND_MULT:-2}
SEL_TEMP=${SEL_TEMP:-0.1}
SEL_SKETCH=${SEL_SKETCH:-1024}
SEL_REDUNDANCY=${SEL_REDUNDANCY:-1.0}
VAL_PROXY_SIZE=${VAL_PROXY_SIZE:-4096}
VAL_PROXY_REFRESH=${VAL_PROXY_REFRESH:-5000}

MODEL_ARGS="--config_format base --model llama --n_embd ${N_EMBD} --n_head ${N_HEAD} --n_layer ${N_LAYER}"
DATA_ARGS="--dataset fineweb --datasets_dir ${DATASETS_DIR}"
COMMON_ARGS="--iterations ${ITERS} --warmup_steps ${WARMUP_STEPS} --scheduler cos"
COMMON_ARGS="${COMMON_ARGS} --grad_clip 0.1 --weight_decay 0.1"
COMMON_ARGS="${COMMON_ARGS} --dropout 0.0 --dtype bfloat16 --device cuda:0"
COMMON_ARGS="${COMMON_ARGS} --distributed_backend nccl"
COMMON_ARGS="${COMMON_ARGS} --latest_ckpt_interval ${LATEST_CKPT_INTERVAL}"
COMMON_ARGS="${COMMON_ARGS} --permanent_ckpt_interval ${PERMANENT_CKPT_INTERVAL}"
if [ "$WANDB_TRACK" = "1" ]; then
    COMMON_ARGS="${COMMON_ARGS} --wandb --wandb_project ${WANDB_PROJECT}"
    if [ -n "$WANDB_ENTITY" ]; then
        COMMON_ARGS="${COMMON_ARGS} --wandb_entity ${WANDB_ENTITY}"
    fi
fi
EVAL_ARGS="--eval_interval ${EVAL_INTERVAL} --log_interval ${LOG_INTERVAL} --eval_batches ${EVAL_BATCHES}"
RESULTS_ARGS="--results_base_folder ${RESULTS_DIR}"

SEL_ARGS="--selection --candidate_multiplier ${CAND_MULT}"
SEL_ARGS="${SEL_ARGS} --selection_temperature ${SEL_TEMP}"
SEL_ARGS="${SEL_ARGS} --selection_sketch_dim ${SEL_SKETCH}"
SEL_ARGS="${SEL_ARGS} --selection_redundancy_weight ${SEL_REDUNDANCY}"
SEL_ARGS="${SEL_ARGS} --val_proxy_size ${VAL_PROXY_SIZE}"
SEL_ARGS="${SEL_ARGS} --val_proxy_refresh ${VAL_PROXY_REFRESH}"
SEL_ARGS="${SEL_ARGS} --val_proxy_source ${PROXY_SOURCE}"
SEL_ARGS="${SEL_ARGS} --val_proxy_tasks ${PROXY_TASKS}"

BATCH="--batch_size ${BATCH_SIZE} --sequence_length ${SEQ_LEN} --acc_steps ${ACC_STEPS}"
SOPHIA_BATCH="--batch_size ${SOPHIA_BATCH_SIZE} --sequence_length ${SEQ_LEN} --acc_steps ${SOPHIA_ACC_STEPS}"

FAILED_RUNS=()
COMPLETED_RUNS=()
SPLIT_OPTIMIZERS=()

upload_hf_tracking() {
    local OPT_CSV=$1
    local DOWNSTREAM_PATH=${2:-}
    local CURVES_PATH=${3:-}

    if [ "$HF_TRACK" != "1" ]; then
        return 0
    fi
    if [ -z "$HF_REPO_ID" ]; then
        echo "[HF] HF_TRACK=1 but HF_REPO_ID is empty; skipping Hugging Face upload."
        return 0
    fi
    if [ -z "$OPT_CSV" ]; then
        echo "[HF] No optimizer names supplied; skipping Hugging Face upload."
        return 0
    fi

    echo ""
    echo "================================================================"
    echo "  Uploading tracking artifacts to Hugging Face Hub"
    echo "  Repo: ${HF_REPO_ID} (type=${HF_REPO_TYPE}, private=${HF_PRIVATE})"
    echo "  Optimizers: ${OPT_CSV}"
    echo "  Checkpoints: ${HF_UPLOAD_CHECKPOINTS}"
    echo "================================================================"
    python hf_tracking.py \
        --repo-id "$HF_REPO_ID" \
        --repo-type "$HF_REPO_TYPE" \
        --private "$HF_PRIVATE" \
        --results-dir "$RESULTS_DIR" \
        --src-dir "$SRC_DIR" \
        --split "$SPLIT" \
        --seed "$SEED" \
        --dataset fineweb100b \
        --model-size 720m \
        --owner "$HF_USERNAME" \
        --contact-email "$HF_CONTACT_EMAIL" \
        --proxy-source "$PROXY_SOURCE" \
        --optimizers "$OPT_CSV" \
        --upload-checkpoints "$HF_UPLOAD_CHECKPOINTS" \
        --curves-file "$CURVES_PATH" \
        --downstream-file "$DOWNSTREAM_PATH"
}

run_experiment() {
    local OPT_NAME=$1
    local MODE=$2
    local OPT_FLAG=$3
    local OPT_EXTRA=$4
    local BATCH_OVERRIDE=${5:-$BATCH}

    local EXP_NAME
    if [ "$PROXY_SOURCE" = "train" ]; then
        EXP_NAME="${MODE}_fineweb100b_720m_${OPT_NAME}_seed${SEED}"
    else
        EXP_NAME="${MODE}_fineweb100b_720m_proxy-${PROXY_SOURCE}_${OPT_NAME}_seed${SEED}"
    fi
    local LOG_FILE="logs/${EXP_NAME}.log"

    if [ -f "${RESULTS_DIR}/${EXP_NAME}/summary.json" ]; then
        if python - << PYTHON_SKIP 2>/dev/null
import json, os
p = "${RESULTS_DIR}/${EXP_NAME}/summary.json"
d = json.load(open(p))
has_args = isinstance(d.get("args"), dict)
has_final = d.get("final_val_loss") is not None or d.get("val_loss") is not None
has_ckpt = os.path.exists("${RESULTS_DIR}/${EXP_NAME}/final.pt/main.pt") or os.path.exists("${RESULTS_DIR}/${EXP_NAME}/ckpts/latest/main.pt")
assert has_args and (has_final or has_ckpt)
PYTHON_SKIP
        then
            echo "[SKIP] ${EXP_NAME} -- already has summary/checkpoint"
            COMPLETED_RUNS+=("$EXP_NAME")
            return 0
        fi
    fi

    rm -f "$LOG_FILE"

    local MODE_EXTRA=""
    if [ "$MODE" = "selection" ]; then
        MODE_EXTRA="$SEL_ARGS"
    fi

    OPT_EXTRA="${OPT_EXTRA//__ITERS__/$ITERS}"

    echo ""
    echo "================================================================"
    echo "  ${OPT_NAME} | ${MODE} | FineWeb100B 720M | seed=${SEED} | GPUs=${NPROC}"
    echo "  Iters=${ITERS} warmup=${WARMUP_STEPS} batch=${BATCH_OVERRIDE}"
    echo "  Results=${RESULTS_DIR}/${EXP_NAME}"
    if [ "$MODE" = "selection" ]; then
        echo "  Selection: Btilde/B=${CAND_MULT}, tau=${SEL_TEMP}, lambda_r=${SEL_REDUNDANCY}, proxy=${PROXY_SOURCE}"
    fi
    echo "  Started: $(date)"
    echo "================================================================"

    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} main.py \
        $MODEL_ARGS \
        $DATA_ARGS \
        $BATCH_OVERRIDE \
        $COMMON_ARGS \
        $EVAL_ARGS \
        $RESULTS_ARGS \
        $OPT_FLAG \
        $OPT_EXTRA \
        $MODE_EXTRA \
        --experiment_name "$EXP_NAME" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_FILE"

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -eq 0 ]; then
        echo ">>> [OK] ${EXP_NAME} at $(date)"
        COMPLETED_RUNS+=("$EXP_NAME")
        if [ "$HF_TRACK_EVERY_RUN" = "1" ]; then
            upload_hf_tracking "$OPT_NAME"
        fi
    else
        echo ">>> [FAIL] ${EXP_NAME} exit ${EXIT_CODE}"
        FAILED_RUNS+=("$EXP_NAME")
    fi

    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null
    return 0
}

run_adamw() {
    run_experiment "adamw" "$1" "--opt adamw" \
        "--lr 1e-3 --beta1 0.9 --beta2 0.999"
}

run_ademamix() {
    run_experiment "ademamix" "$1" "--opt ademamix" \
        "--lr 1e-3 --beta1 0.9 --beta2 0.999 --adema_beta3 0.999 --adema_alpha 8.0 --adema_beta3_warmup __ITERS__ --adema_alpha_warmup __ITERS__"
}

run_dmuon() {
    run_experiment "d-muon" "$1" "--opt d-muon" \
        "--lr 1e-3 --beta1 0.9 --beta2 0.99 --momentum 0.95 --nesterov True --muon_ns_steps 5"
}

run_mars() {
    run_experiment "mars" "$1" "--opt mars" \
        "--lr 1e-3 --mars_lr 3e-3 --beta1 0.8 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99 --mars_vr_gamma 0.025"
}

run_sophiag() {
    run_experiment "sophiag" "$1" "--opt sophiag" \
        "--lr 5e-4 --beta1 0.95 --beta2 0.99 --sophia_rho 0.04 --precondition_frequency 10 --sophia_bs ${SOPHIA_BS}" \
        "$SOPHIA_BATCH"
}

run_soap() {
    run_experiment "soap" "$1" "--opt soap" \
        "--lr 1e-3 --beta1 0.95 --beta2 0.95 --precondition_frequency 10"
}

run_lion() {
    run_experiment "lion" "$1" "--opt lion" \
        "--lr 2e-4 --beta1 0.9 --beta2 0.99"
}

run_signum() {
    run_experiment "signum" "$1" "--opt signum" \
        "--lr 2e-4 --momentum 0.95 --nesterov True"
}

run_adopt() {
    run_experiment "adopt" "$1" "--opt adopt" \
        "--lr 1e-3 --beta1 0.95 --beta2 0.999"
}

run_sgd() {
    run_experiment "sgd" "$1" "--opt sgd" \
        "--lr 3e-2 --momentum 0.9"
}

run_pair() {
    local FN=$1
    local OPT=$2
    SPLIT_OPTIMIZERS+=("$OPT")
    $FN "standard"
    $FN "selection"
}

echo ""
echo "================================================================"
echo "  OptiSelect FineWeb100B 720M Split ${SPLIT}"
echo "  Model: Llama n_embd=${N_EMBD}, n_head=${N_HEAD}, n_layer=${N_LAYER}"
echo "  Params: ${PARAM_COUNT}"
UPDATE_TOKENS_B=$(python - << PY
print(f"{${ITERS} * ${GLOBAL_EXAMPLES_PER_STEP} * ${SEQ_LEN} / 1e9:.3f}B")
PY
)
echo "  Update tokens: ${UPDATE_TOKENS_B}"
echo "  Seed: ${SEED} | DDP GPUs: ${NPROC} | Results: ${RESULTS_DIR}"
echo "  Started: $(date)"
echo "================================================================"

case $SPLIT in
    0)
        run_pair run_adamw    "adamw"
        run_pair run_ademamix "ademamix"
        run_pair run_dmuon    "d-muon"
        run_pair run_mars     "mars"
        run_pair run_sophiag  "sophiag"
        run_pair run_soap     "soap"
        run_pair run_lion     "lion"
        run_pair run_signum   "signum"
        run_pair run_adopt    "adopt"
        run_pair run_sgd      "sgd"
        ;;
    1)
        run_pair run_adamw    "adamw"
        run_pair run_ademamix "ademamix"
        ;;
    2)
        run_pair run_dmuon "d-muon"
        run_pair run_mars  "mars"
        ;;
    3)
        run_pair run_sophiag "sophiag"
        run_pair run_soap    "soap"
        ;;
    4)
        run_pair run_lion   "lion"
        run_pair run_signum "signum"
        ;;
    5)
        run_pair run_adopt "adopt"
        run_pair run_sgd   "sgd"
        ;;
    *)
        echo "[FATAL] Unknown split ${SPLIT}; use 0,1,2,3,4,5"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "  Split ${SPLIT} finished at $(date)"
echo "  Completed: ${#COMPLETED_RUNS[@]} | Failed: ${#FAILED_RUNS[@]}"
echo "================================================================"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "Failed runs:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "  - $run"
        tail -5 "logs/${run}.log" 2>/dev/null | sed 's/^/      /'
    done
fi

export ITERS GLOBAL_EXAMPLES_PER_STEP SEQ_LEN PARAM_COUNT CAND_MULT

CURVES_FILE="${SRC_DIR}/logs/fineweb100b_720m_curves_seed${SEED}$([ "$PROXY_SOURCE" != "train" ] && echo "_proxy-${PROXY_SOURCE}").json"

python - << 'PYTHON_COLLECT'
import json
import math
import os
import re

src_dir = "/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src"
log_dir = os.path.join(src_dir, "logs")
results_dir = os.environ["RESULTS_DIR"]
seed = int(os.environ.get("SEED", 0))
proxy_source = os.environ.get("PROXY_SOURCE", "train")
iters = int(os.environ.get("ITERS", 48000))
examples_per_step = int(os.environ.get("GLOBAL_EXAMPLES_PER_STEP", 1984))
seq_len = int(os.environ.get("SEQ_LEN", 512))
param_count = int(os.environ.get("PARAM_COUNT", 719685632))
cand_mult = int(os.environ.get("CAND_MULT", 2))

optimizers = ["adamw", "ademamix", "d-muon", "mars", "sophiag",
              "soap", "lion", "signum", "adopt", "sgd"]
modes = ["standard", "selection"]
proxy_tag = "" if proxy_source == "train" else f"proxy-{proxy_source}_"

def run_name(mode, opt):
    return f"{mode}_fineweb100b_720m_{proxy_tag}{opt}_seed{seed}"

def parse_log_curve(path):
    curve = []
    if not os.path.exists(path):
        return curve
    pat = re.compile(r">Eval: Iter=(\d+).*val_loss=([0-9.]+)\s+val_pp=([0-9.]+)\s+val_acc=([0-9.]+)")
    with open(path, errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            step = int(m.group(1))
            curve.append({
                "iter": step,
                "val_loss": float(m.group(2)),
                "val_pp": float(m.group(3)),
                "val_acc": float(m.group(4)),
            })
    return curve

def enrich(mode, points):
    processed_factor = 1 if mode == "standard" else 1 + cand_mult
    out = []
    for p in points:
        update_tokens = p["iter"] * examples_per_step * seq_len
        processed_tokens = update_tokens * processed_factor
        q = dict(p)
        q["update_tokens"] = update_tokens
        q["update_tokens_B"] = update_tokens / 1e9
        q["processed_tokens"] = processed_tokens
        q["processed_tokens_B"] = processed_tokens / 1e9
        q["estimated_training_flops"] = 6 * param_count * processed_tokens
        q["estimated_training_flops_EF"] = q["estimated_training_flops"] / 1e18
        out.append(q)
    return out

payload = {
    "metadata": {
        "dataset": "fineweb100b",
        "model_size": "720m",
        "model": "llama",
        "n_embd": 2048,
        "n_head": 16,
        "n_layer": 12,
        "param_count": param_count,
        "seq_len": seq_len,
        "iterations": iters,
        "examples_per_step": examples_per_step,
        "update_tokens_total": iters * examples_per_step * seq_len,
        "update_tokens_total_B": iters * examples_per_step * seq_len / 1e9,
        "selection_candidate_multiplier": cand_mult,
        "flops_rule": "6 * param_count * processed_tokens; selection processed_tokens=(1+candidate_multiplier)*update_tokens",
        "proxy_source": proxy_source,
    },
    "runs": {},
}

complete = 0
for opt in optimizers:
    payload["runs"][opt] = {}
    for mode in modes:
        name = run_name(mode, opt)
        summary_path = os.path.join(results_dir, name, "summary.json")
        log_path = os.path.join(log_dir, f"{name}.log")
        points = []
        final = {}

        if os.path.exists(summary_path):
            try:
                d = json.load(open(summary_path))
                hist = d.get("history") or []
                for h in hist:
                    if h.get("val_loss") is not None and h.get("iter") is not None:
                        points.append({
                            "iter": int(h["iter"]),
                            "val_loss": float(h["val_loss"]),
                            "val_pp": float(h.get("val_pp", math.nan)),
                            "val_acc": float(h.get("val_acc", math.nan)),
                            "selection_entropy": h.get("entropy"),
                        })
                for k in ("final_val_loss", "final_val_pp", "final_val_acc", "best_val_loss"):
                    if k in d:
                        final[k] = d[k]
                if d.get("final_val_loss") is not None and not any(p["iter"] == iters for p in points):
                    points.append({
                        "iter": iters,
                        "val_loss": float(d["final_val_loss"]),
                        "val_pp": float(d.get("final_val_pp", math.nan)),
                        "val_acc": float(d.get("final_val_acc", math.nan)),
                    })
            except Exception as exc:
                final["summary_error"] = repr(exc)

        if not points:
            points = parse_log_curve(log_path)

        points = sorted({p["iter"]: p for p in points}.values(), key=lambda x: x["iter"])
        if points:
            complete += 1
            final.setdefault("final_val_loss", points[-1].get("val_loss"))
            final.setdefault("final_val_pp", points[-1].get("val_pp"))
            final.setdefault("final_val_acc", points[-1].get("val_acc"))

        payload["runs"][opt][mode] = {
            "name": name,
            "summary_path": summary_path if os.path.exists(summary_path) else None,
            "log_path": log_path if os.path.exists(log_path) else None,
            "final": final,
            "curve": enrich(mode, points),
        }

out = os.path.join(log_dir, f"fineweb100b_720m_curves_seed{seed}" + ("" if proxy_source == "train" else f"_proxy-{proxy_source}") + ".json")
with open(out, "w") as f:
    json.dump(payload, f, indent=2)

print(f"[COLLECTOR] Wrote {out}")
print(f"[COLLECTOR] Curves found for {complete}/{len(optimizers)*len(modes)} expected runs")
print(f"[COLLECTOR] Final update-token budget: {payload['metadata']['update_tokens_total_B']:.3f}B")
PYTHON_COLLECT

DOWNSTREAM_OUT=""
if [ "$RUN_DOWNSTREAM" = "1" ]; then
    OPT_CSV=$(IFS=, ; echo "${SPLIT_OPTIMIZERS[*]}")
    DOWNSTREAM_OUT="${RESULTS_DIR}/downstream_eval_${DOWNSTREAM_TASKS//,/+}_split${SPLIT}.json"
    echo ""
    echo "================================================================"
    echo "  Running downstream eval for split ${SPLIT}: ${OPT_CSV}"
    echo "================================================================"
    python eval_downstream.py \
        --results-dir "$RESULTS_DIR" \
        --dataset fineweb100b_720m \
        --tasks "$DOWNSTREAM_TASKS" \
        --optimizers "$OPT_CSV" \
        --modes standard,selection \
        --out "$DOWNSTREAM_OUT"
fi

if [ "$HF_TRACK" = "1" ]; then
    OPT_CSV=$(IFS=, ; echo "${SPLIT_OPTIMIZERS[*]}")
    upload_hf_tracking "$OPT_CSV" "$DOWNSTREAM_OUT" "$CURVES_FILE"
fi

echo ""
echo "Done. Results: ${RESULTS_DIR}"
echo "Curves: ${CURVES_FILE}"
