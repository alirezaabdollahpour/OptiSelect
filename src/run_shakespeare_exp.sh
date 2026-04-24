#!/bin/bash
# =============================================================================
# OptiSelect Shakespeare-Char Benchmark — Publication Ready (v3)
# =============================================================================
#
# Shakespeare (tiny-shakespeare): ~1.1M characters from Shakespeare's plays,
# tokenized at the CHARACTER level (~95 unique tokens, not GPT-2 BPE).
# Train: ~880K chars, Val: ~220K chars.
#
# ---- Purpose ----
#
# Shakespeare serves a specific role in the OptiSelect paper:
#   - Rapid iteration / debugging benchmark (runs complete in 5-15 min)
#   - Tiny model (~2M params) isolates optimizer effects from capacity effects
#   - Character-level vocabulary eliminates BPE confounds
#   - Demonstrates robustness of findings to very different token distributions
#
# This is NOT a primary paper table; it's supplementary evidence. Expect:
#   - High val accuracy (~55-65%) due to small vocab (95 tokens)
#   - Some optimizer differences (Lion/Signum will still sign-collapse)
#   - Small absolute Δ between std and selection (~0.5-1.5 pp)
#
# ---- Paper Alignment ----
#
# Implementation matches:
#   - Frozen-state operator O_t (Paper Section 4.1, Table 1) for all 10 optimizers
#   - 4,096-document validation proxy SCALED to 512 for small val set
#   - Redundancy penalty λ_r=1.0 (Paper Eq. 4)
#   - MARS scoring via raw-gradient c_t (Paper Remark 4)
#   - Muon right-preconditioner (M^T M + εI)^{-1/2} (Paper Eq. 15, Remark 3)
#   - SOAP rotated Ghost factors (Paper Eq. 18, 32)
#   - Sophia linearized clip scoring (Paper Remark)
#
# ---- Model Sizing ----
#
# Character-level vocab = 95 (not 50,257 like GPT-2 BPE).
# Tiny-Shakespeare has 1.1M tokens, so we need a model that doesn't
# severely overfit. We use:
#   - n_embd=256, n_head=8, n_layer=4 → ~2M parameters
#   - With vocab=95 and tied embeddings: embedding matrix is negligible
#
# Training budget:
#   - 3000 iters × 64 batch × 256 seq = 49M tokens total
#   - Each char appears ~44× on average → near-memorization regime
#   - Selection mode: 3000 iters × 2× candidates = same compute
#
# ---- Memory Safety ----
#
# Shakespeare is very light on memory. Even at batch=256 × seq=256,
# a 2M model fits easily in 4GB. However, we still apply:
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (defensive)
#   - OOM detection with batch halving retry (inherited from other scripts)
#   - GPU cache release between runs
#
# ---- Expected runtime ----
#   ~5-10 min per run × 20 runs = ~2-3 hours total on a single A100
#   (Shakespeare is fast enough that splitting across GPUs is unnecessary)
#
# Usage:
#   bash run_shakespeare_exp.sh [SEED] [NPROC] [TAG]
#
# Each optimizer run uses DDP across NPROC GPUs (default 4). The nccl
# backend auto-divides batch_size*acc_steps across ranks, so effective
# batch is preserved — runs are just ~NPROC× faster.
#
# TAG: optional run label that isolates the results directory and log
# filenames from other Shakespeare runs. Use this when sweeping a
# hyperparameter (e.g., TAG=refresh100 for VAL_PROXY_REFRESH=100).
# Empty TAG preserves the legacy untagged layout.
# =============================================================================

# Do NOT use set -e — per-run failures must not cascade

SEED=${1:-0}
NPROC=${2:-4}   # GPUs per run (DDP world size)
TAG=${3:-}      # optional suffix for isolating this variant's outputs

# ---- Paths ----
SRC_DIR="/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src"
DATASETS_DIR="/mloscratch/homes/aabdolla/datasets"
if [ -n "$TAG" ]; then
    RESULTS_DIR="/mloscratch/homes/aabdolla/results/shakespeare_exp_${TAG}"
else
    RESULTS_DIR="/mloscratch/homes/aabdolla/results/shakespeare_exp"
fi

cd "$SRC_DIR"
source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate
export PYTHONPATH="/mloscratch/homes/aabdolla/GhostSuite:${SRC_DIR}:$PYTHONPATH"
export HF_HOME=/mloscratch/homes/aabdolla/.hf_cache
export HF_DATASETS_CACHE=/mloscratch/homes/aabdolla/.hf_cache/datasets

# Memory safety environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SEED RESULTS_DIR TAG

mkdir -p "$RESULTS_DIR" logs


# ======================================================================
# Download Shakespeare if needed (it's tiny, ~1 MB)
# ======================================================================
if [ ! -f "${DATASETS_DIR}/shakespeare/train.npy" ]; then
    echo "[INFO] Shakespeare not found. Downloading and tokenizing..."
    python -c "
from data.shakespeare import get_shakespeare_data
paths = get_shakespeare_data('${DATASETS_DIR}')
print(f'[OK] Shakespeare ready at {paths}')
"
fi

# ---- Verify data and report GPU ----
python -c "
import numpy as np, os, torch
p = '${DATASETS_DIR}/shakespeare/train.npy'
tp = '${DATASETS_DIR}/shakespeare/test.npy'
if os.path.exists(p):
    data = np.memmap(p, dtype=np.uint16, mode='r')
    print(f'[OK] Shakespeare train: {len(data):,} chars')
if os.path.exists(tp):
    data2 = np.memmap(tp, dtype=np.uint16, mode='r')
    print(f'[OK] Shakespeare val:   {len(data2):,} chars')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'[GPU] {props.name} — {props.total_memory/1e9:.1f} GB total')
    torch.cuda.empty_cache()
"


# ======================================================================
# Fixed parameters (Shakespeare-specific)
# ======================================================================
# Model: tiny ~2M param Llama for character-level training
# n_embd=256, n_head=8, n_layer=4 — 8 heads × 32 dim/head = 256
MODEL_ARGS="--model llama --n_embd 256 --n_head 8 --n_layer 4"
MODEL_ARGS="${MODEL_ARGS} --vocab_size 96"  # 95 valid chars + 1 for safety

# Data
DATA_ARGS="--dataset shakespeare-char --datasets_dir ${DATASETS_DIR}"
DATA_ARGS="${DATA_ARGS} --tokenizer gpt2"  # tokenizer arg is required; actual
                                           # tokenization uses char map internally

# Training parameters
ITERS=3000                  # 3K iters = fast validation
WARMUP_STEPS=60             # 2% of iterations
BATCH_SIZE=64
SEQ_LEN=256                 # shorter than web datasets (shakespeare lines are short)
ACC_STEPS=1
EVAL_INTERVAL=200
LOG_INTERVAL=50
EVAL_BATCHES=32
GRAD_CLIP=1.0
WEIGHT_DECAY=0.1

# Selection hyperparameters (Paper Appendix C, scaled for small dataset)
CAND_MULT=2
SEL_TEMP=0.1
SEL_SKETCH=1024
SEL_REDUNDANCY=1.0
VAL_PROXY_SIZE=512          # Paper uses 4,096; Shakespeare val = 220K chars ≈ 860 docs at seq=256
VAL_PROXY_REFRESH=50      # Refresh more often than paper's 5000 since training is short

# Batch configs
BATCH="--batch_size ${BATCH_SIZE} --sequence_length ${SEQ_LEN} --acc_steps ${ACC_STEPS}"
SOPHIA_BATCH="--batch_size 32 --sequence_length ${SEQ_LEN} --acc_steps 2"
SOAP_BATCH="$BATCH"  # SOAP fine at this scale

# Common args
COMMON="--scheduler cos --grad_clip ${GRAD_CLIP} --weight_decay ${WEIGHT_DECAY}"
COMMON="${COMMON} --dropout 0.0 --dtype bfloat16 --device cuda:0"
COMMON="${COMMON} --iterations ${ITERS} --warmup_steps ${WARMUP_STEPS}"
COMMON="${COMMON} --distributed_backend nccl"

EVAL_ARGS="--eval_interval ${EVAL_INTERVAL} --log_interval ${LOG_INTERVAL} --eval_batches ${EVAL_BATCHES}"
RESULTS_ARGS="--results_base_folder ${RESULTS_DIR}"

# Selection args
SEL_ARGS="--selection --candidate_multiplier ${CAND_MULT}"
SEL_ARGS="${SEL_ARGS} --selection_temperature ${SEL_TEMP}"
SEL_ARGS="${SEL_ARGS} --selection_sketch_dim ${SEL_SKETCH}"
SEL_ARGS="${SEL_ARGS} --selection_redundancy_weight ${SEL_REDUNDANCY}"
SEL_ARGS="${SEL_ARGS} --val_proxy_size ${VAL_PROXY_SIZE}"
SEL_ARGS="${SEL_ARGS} --val_proxy_refresh ${VAL_PROXY_REFRESH}"

SOPHIA_BS=32

FAILED_RUNS=()
COMPLETED_RUNS=()


# ======================================================================
#  Runner with OOM detection and automatic retry
# ======================================================================
run_experiment() {
    local OPT_NAME=$1
    local MODE=$2
    local OPT_FLAG=$3
    local OPT_EXTRA=$4
    local BATCH_OVR=${5:-$BATCH}

    local EXP_NAME
    if [ -n "$TAG" ]; then
        EXP_NAME="${MODE}_shakespeare_${TAG}_${OPT_NAME}_seed${SEED}"
    else
        EXP_NAME="${MODE}_shakespeare_${OPT_NAME}_seed${SEED}"
    fi
    local LOG_FILE="logs/${EXP_NAME}.log"

    # Skip only if summary has valid final_val_loss
    if [ -f "${RESULTS_DIR}/${EXP_NAME}/summary.json" ]; then
        if python -c "
import json
d = json.load(open('${RESULTS_DIR}/${EXP_NAME}/summary.json'))
assert 'final_val_loss' in d and d['final_val_loss'] is not None
" 2>/dev/null; then
            echo "[SKIP] ${EXP_NAME} — already completed"
            COMPLETED_RUNS+=("$EXP_NAME")
            return 0
        fi
    fi

    rm -f "$LOG_FILE"

    local SEL_FLAG=""
    if [ "$MODE" == "selection" ]; then
        SEL_FLAG="$SEL_ARGS"
    fi

    echo ""
    echo "================================================================"
    echo "  ${OPT_NAME} | ${MODE} | seed=${SEED} | GPUs=${NPROC}"
    echo "  Iters: ${ITERS} | Batch: $(echo $BATCH_OVR)"
    if [ "$MODE" == "selection" ]; then
        echo "  Selection: B̃/B=${CAND_MULT}, τ=${SEL_TEMP}, λ_r=${SEL_REDUNDANCY}"
        echo "             Val proxy: ${VAL_PROXY_SIZE} docs, refresh every ${VAL_PROXY_REFRESH} steps"
    fi
    echo "  Started: $(date)"
    echo "================================================================"

    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} main.py \
        $MODEL_ARGS $DATA_ARGS $BATCH_OVR \
        $COMMON $EVAL_ARGS $RESULTS_ARGS \
        $OPT_FLAG $OPT_EXTRA $SEL_FLAG \
        --experiment_name "$EXP_NAME" --seed $SEED \
        2>&1 | tee "$LOG_FILE"

    local EXIT_CODE=${PIPESTATUS[0]}

    # ===== OOM detection: retry with halved batch =====
    if [ $EXIT_CODE -ne 0 ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "!!! [OOM] Detected OOM for ${EXP_NAME}. Retrying with halved batch..."

        local CUR_BS=$(echo "$BATCH_OVR" | grep -oP '(?<=--batch_size )\d+')
        local CUR_AS=$(echo "$BATCH_OVR" | grep -oP '(?<=--acc_steps )\d+')
        local NEW_BS=$((CUR_BS / 2))
        local NEW_AS=$((CUR_AS * 2))

        if [ $NEW_BS -lt 4 ]; then
            echo "!!! [FATAL] Cannot halve further; NEW_BS=${NEW_BS} < 4"
            FAILED_RUNS+=("$EXP_NAME")
            return 1
        fi

        local RETRY_BATCH="--batch_size ${NEW_BS} --sequence_length ${SEQ_LEN} --acc_steps ${NEW_AS}"
        echo "    Retry config: ${RETRY_BATCH}"

        rm -rf "${RESULTS_DIR}/${EXP_NAME}"
        mv "$LOG_FILE" "${LOG_FILE}.oom-attempt1"

        local RETRY_EXTRA="$OPT_EXTRA"
        if [ "$OPT_NAME" == "sophiag" ]; then
            RETRY_EXTRA=$(echo "$OPT_EXTRA" | sed "s/--sophia_bs [0-9]*/--sophia_bs ${NEW_BS}/")
        fi

        torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} main.py \
            $MODEL_ARGS $DATA_ARGS $RETRY_BATCH \
            $COMMON $EVAL_ARGS $RESULTS_ARGS \
            $OPT_FLAG $RETRY_EXTRA $SEL_FLAG \
            --experiment_name "$EXP_NAME" --seed $SEED \
            2>&1 | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}

        # Second OOM attempt
        if [ $EXIT_CODE -ne 0 ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_FILE" 2>/dev/null; then
            echo "!!! [OOM] Second attempt. Halving again..."
            local NEW_BS2=$((NEW_BS / 2))
            local NEW_AS2=$((NEW_AS * 2))
            if [ $NEW_BS2 -lt 4 ]; then
                echo "!!! [FATAL] Cannot halve further; NEW_BS=${NEW_BS2} < 4"
                FAILED_RUNS+=("$EXP_NAME")
                return 1
            fi
            local RETRY_BATCH2="--batch_size ${NEW_BS2} --sequence_length ${SEQ_LEN} --acc_steps ${NEW_AS2}"
            local RETRY_EXTRA2="$OPT_EXTRA"
            if [ "$OPT_NAME" == "sophiag" ]; then
                RETRY_EXTRA2=$(echo "$OPT_EXTRA" | sed "s/--sophia_bs [0-9]*/--sophia_bs ${NEW_BS2}/")
            fi
            rm -rf "${RESULTS_DIR}/${EXP_NAME}"
            mv "$LOG_FILE" "${LOG_FILE}.oom-attempt2"

            torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} main.py \
                $MODEL_ARGS $DATA_ARGS $RETRY_BATCH2 \
                $COMMON $EVAL_ARGS $RESULTS_ARGS \
                $OPT_FLAG $RETRY_EXTRA2 $SEL_FLAG \
                --experiment_name "$EXP_NAME" --seed $SEED \
                2>&1 | tee "$LOG_FILE"
            EXIT_CODE=${PIPESTATUS[0]}
        fi
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo ">>> [OK] ${EXP_NAME} at $(date)"
        COMPLETED_RUNS+=("$EXP_NAME")
    else
        echo ">>> [FAIL] ${EXP_NAME} exit ${EXIT_CODE}"
        FAILED_RUNS+=("$EXP_NAME")
    fi

    # Release GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    return 0
}


# ======================================================================
echo ""
echo "================================================================"
echo "  OptiSelect Shakespeare-Char Benchmark (v3)"
echo "  10 optimizers × 2 modes = 20 runs | Seed: ${SEED} | DDP GPUs: ${NPROC}"
echo "  Model: ~2M params (Llama: n_embd=256, n_head=8, n_layer=4)"
echo "  Vocab: 95 chars (character-level, not BPE)"
echo "  Training: ${ITERS} iters × 16K tok ≈ 49M tokens"
echo "  Alloc: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  Started: $(date)"
echo "================================================================"


# ======================================================================
# All 10 optimizers (Paper Appendix C hyperparameters)
# Each block documents the frozen-state operator reference
# ======================================================================

# -----------------------------------------------------------
# 1. AdamW — Paper Eq. 7
# -----------------------------------------------------------
run_experiment "adamw" "standard" \
    "--opt adamw" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.999"

run_experiment "adamw" "selection" \
    "--opt adamw" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.999"


# -----------------------------------------------------------
# 2. AdEMAMix — Paper Eq. 8, H1
# Operator identical to AdamW; slow momentum stabilizes influence
# -----------------------------------------------------------
run_experiment "ademamix" "standard" \
    "--opt ademamix" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.999 --adema_beta3 0.9999 --adema_alpha 0.8"

run_experiment "ademamix" "selection" \
    "--opt ademamix" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.999 --adema_beta3 0.9999 --adema_alpha 0.8"


# -----------------------------------------------------------
# 3. D-Muon — Paper Eq. 15, Remark 3
# -----------------------------------------------------------
run_experiment "d-muon" "standard" \
    "--opt d-muon" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"

run_experiment "d-muon" "selection" \
    "--opt d-muon" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"


# -----------------------------------------------------------
# 4. MARS — Paper Remark 4
# -----------------------------------------------------------
run_experiment "mars" "standard" \
    "--opt mars" \
    "--lr 1e-3 --mars_lr 3e-3 --beta1 0.9 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99 --mars_vr_gamma 0.025"

run_experiment "mars" "selection" \
    "--opt mars" \
    "--lr 1e-3 --mars_lr 3e-3 --beta1 0.9 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99 --mars_vr_gamma 0.025"


# -----------------------------------------------------------
# 5. Sophia — Paper Eq. 11, H3 / Proposition 1
# -----------------------------------------------------------
run_experiment "sophiag" "standard" \
    "--opt sophiag" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.999 --sophia_rho 0.04 --precondition_frequency 10 --sophia_bs ${SOPHIA_BS}" \
    "$SOPHIA_BATCH"

run_experiment "sophiag" "selection" \
    "--opt sophiag" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.999 --sophia_rho 0.04 --precondition_frequency 10 --sophia_bs ${SOPHIA_BS}" \
    "$SOPHIA_BATCH"


# -----------------------------------------------------------
# 6. SOAP — Paper Eq. 18, 32
# β₂=0.95 per Paper Appendix C
# -----------------------------------------------------------
run_experiment "soap" "standard" \
    "--opt soap" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.95 --precondition_frequency 10" \
    "$SOAP_BATCH"

run_experiment "soap" "selection" \
    "--opt soap" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.95 --precondition_frequency 10" \
    "$SOAP_BATCH"


# -----------------------------------------------------------
# 7. Lion — Paper Eq. 13-14, H2 / Theorem 1
# Expected: smallest selection synergy (sign collapse)
# lr=3e-4 per Chen et al. [4]
# -----------------------------------------------------------
run_experiment "lion" "standard" \
    "--opt lion" \
    "--lr 3e-4 --beta1 0.9 --beta2 0.99"

run_experiment "lion" "selection" \
    "--opt lion" \
    "--lr 3e-4 --beta1 0.9 --beta2 0.99"


# -----------------------------------------------------------
# 8. Signum — Paper Section 4.3.2
# O_t^Signum(x) = sign(m_{t-1}) [constant in x]
# -----------------------------------------------------------
run_experiment "signum" "standard" \
    "--opt signum" \
    "--lr 3e-4 --momentum 0.9"

run_experiment "signum" "selection" \
    "--opt signum" \
    "--lr 3e-4 --momentum 0.9"


# -----------------------------------------------------------
# 9. ADOPT — Paper Section 4.3.1 (lagged variance)
# -----------------------------------------------------------
run_experiment "adopt" "standard" \
    "--opt adopt" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.999"

run_experiment "adopt" "selection" \
    "--opt adopt" \
    "--lr 1e-3 --beta1 0.9 --beta2 0.999"


# -----------------------------------------------------------
# 10. SGD — baseline, identity operator
# -----------------------------------------------------------
run_experiment "sgd" "standard" \
    "--opt sgd" \
    "--lr 3e-2 --momentum 0.9"

run_experiment "sgd" "selection" \
    "--opt sgd" \
    "--lr 3e-2 --momentum 0.9"


# ======================================================================
#  Summary and results collection
# ======================================================================
echo ""
echo "================================================================"
echo "  EXPERIMENT SUMMARY"
echo "  Finished: $(date)"
echo "  Completed: ${#COMPLETED_RUNS[@]} | Failed: ${#FAILED_RUNS[@]}"
echo "================================================================"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "  Failed runs:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "    - $run"
        echo "      Last 5 lines:"
        tail -5 "logs/${run}.log" 2>/dev/null | sed 's/^/      /'
    done
fi

echo ""
echo "Collecting results..."

python - << 'PYTHON_COLLECTOR'
import os, json

SEED = int(os.environ.get("SEED", "0"))
RESULTS_DIR = os.environ.get("RESULTS_DIR",
    "/mloscratch/homes/aabdolla/results/shakespeare_exp")
LOG_DIR = "logs"

optimizers = ["adamw", "ademamix", "d-muon", "mars", "sophiag", "soap",
              "lion", "signum", "adopt", "sgd"]
display = {
    "adamw": "AdamW", "ademamix": "AdEMAMix", "d-muon": "D-Muon",
    "mars": "MARS", "sophiag": "Sophia", "soap": "SOAP",
    "lion": "Lion", "signum": "Signum", "adopt": "ADOPT", "sgd": "SGD",
}
modes = ["standard", "selection"]

results = {}
n_found = 0
for opt in optimizers:
    results[opt] = {}
    for mode in modes:
        p = os.path.join(RESULTS_DIR,
                         f"{mode}_shakespeare_{opt}_seed{SEED}",
                         "summary.json")
        r = {"val_loss": None, "val_pp": None, "val_acc": None, "entropy": None}
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                if d.get("final_val_loss") is not None:
                    r["val_loss"] = d.get("final_val_loss")
                    r["val_pp"]   = d.get("final_val_pp")
                    r["val_acc"]  = d.get("final_val_acc")
                    ssum = d.get("selection_summary", {}) or {}
                    r["entropy"] = ssum.get("mean_entropy")
                    n_found += 1
            except Exception:
                pass
        results[opt][mode] = r

print()
print("=" * 130)
print(f"  Shakespeare-Char Results: Standard vs OptiSelect (seed {SEED})")
print(f"  Model: ~2M params (Llama, char-level vocab=95)")
print(f"  Training: 3000 iters ≈ 49M tokens | Found {n_found}/20 runs")
print("=" * 130)
print()
hdr = f"{'Optimizer':<12} | {'Std Loss':>10} {'Std PP':>10} {'Std Acc%':>10} | {'Sel Loss':>10} {'Sel PP':>10} {'Sel Acc%':>10} | {'ΔLoss':>8} {'ΔAcc':>8} {'H_sel':>6}"
print(hdr)
print("-" * len(hdr))

for opt in optimizers:
    s = results[opt]["standard"]
    x = results[opt]["selection"]
    name = display.get(opt, opt)
    sl = f"{s['val_loss']:.3f}" if s['val_loss'] else "    —"
    sp = f"{s['val_pp']:.2f}"  if s['val_pp']  else "    —"
    sa = f"{100*s['val_acc']:.2f}" if s['val_acc'] else "    —"
    xl = f"{x['val_loss']:.3f}" if x['val_loss'] else "    —"
    xp = f"{x['val_pp']:.2f}"  if x['val_pp']  else "    —"
    xa = f"{100*x['val_acc']:.2f}" if x['val_acc'] else "    —"
    dl = f"{x['val_loss']-s['val_loss']:+.3f}" if s['val_loss'] and x['val_loss'] else "    —"
    da = f"{100*(x['val_acc']-s['val_acc']):+.2f}" if s['val_acc'] and x['val_acc'] else "    —"
    en = f"{x['entropy']:.2f}" if x['entropy'] else "   —"
    print(f"{name:<12} | {sl:>10} {sp:>10} {sa:>10} | {xl:>10} {xp:>10} {xa:>10} | {dl:>8} {da:>8} {en:>6}")

print()

# LaTeX table (supplementary)
print()
print("% ====== LaTeX Table: Shakespeare-Char (Supplementary) ======")
print(r"\begin{table}[t]")
print(r"\centering")
print(f"\\caption{{Shakespeare-Char results (2M-parameter Llama, character-level vocab).")
print(f"Supplementary benchmark demonstrating robustness to non-BPE token distributions.")
print(f"Seed {SEED}.}}")
print(r"\label{tab:shakespeare_results}")
print(r"\small")
print(r"\begin{tabular}{l cc cc c c}")
print(r"\toprule")
print(r"& \multicolumn{2}{c}{Standard} & \multicolumn{2}{c}{+ OptiSelect} & & \\")
print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
print(r"Optimizer & Loss & Acc (\%) & Loss & Acc (\%) & $\Delta$Acc & $\mathcal{H}_{\mathrm{sel}}$ \\")
print(r"\midrule")

std_accs = [(o, results[o]["standard"]["val_acc"]) for o in optimizers if results[o]["standard"]["val_acc"]]
sel_accs = [(o, results[o]["selection"]["val_acc"]) for o in optimizers if results[o]["selection"]["val_acc"]]
best_std = max(std_accs, key=lambda x: x[1])[0] if std_accs else None
best_sel = max(sel_accs, key=lambda x: x[1])[0] if sel_accs else None

for opt in optimizers:
    s = results[opt]["standard"]
    x = results[opt]["selection"]
    name = display.get(opt, opt)
    if s["val_loss"]:
        sl = f"{s['val_loss']:.3f}"
        sa = f"{100*s['val_acc']:.2f}"
        if opt == best_std: sa = r"\textbf{" + sa + "}"
    else: sl = sa = "—"
    if x["val_loss"]:
        xl = f"{x['val_loss']:.3f}"
        xa = f"{100*x['val_acc']:.2f}"
        if opt == best_sel: xa = r"\textbf{" + xa + "}"
    else: xl = xa = "—"
    if s["val_acc"] and x["val_acc"]:
        da = f"{100*(x['val_acc']-s['val_acc']):+.2f}"
    else: da = "—"
    ent = f"{x['entropy']:.2f}" if x["entropy"] else "—"
    print(f"{name} & {sl} & {sa} & {xl} & {xa} & {da} & {ent} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

out = os.path.join(LOG_DIR, f"shakespeare_results_seed{SEED}.json")
with open(out, "w") as f:
    json.dump({
        "metadata": {
            "dataset": "shakespeare-char",
            "model": "llama-2M (n_embd=256, n_head=8, n_layer=4)",
            "vocab_size": 95,
            "iterations": 3000,
            "seed": SEED,
        },
        "results": results,
    }, f, indent=2)
print(f"\nSaved: {out}")

PYTHON_COLLECTOR

echo ""
echo "================================================================"
echo "  Done. Key files:"
echo "    Logs:        ${SRC_DIR}/logs/*shakespeare*.log"
echo "    Checkpoints: ${RESULTS_DIR}/"
echo "    Summary:     ${SRC_DIR}/logs/shakespeare_results_seed${SEED}.json"
echo "================================================================"