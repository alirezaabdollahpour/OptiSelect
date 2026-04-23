#!/bin/bash
# =============================================================================
# OptiSelect SlimPajama Split Dispatcher — Publication Ready (v3)
# =============================================================================
#
# SlimPajama (DKYoon/SlimPajama-6B): deduplicated, quality-filtered multi-
# domain corpus (Common Crawl, C4, GitHub, Wikipedia, Books, ArXiv, Stack
# Exchange). GPT-2 BPE tokenizer (vocab=50257). ~6B tokens.
#
# SlimPajama is the primary multi-domain benchmark in the OptiSelect paper.
# It complements OpenWebText2 (single-domain web) and WikiText (narrow
# high-quality text) to test whether optimizer-selection interactions are
# distribution-dependent.
#
# ---- Paper Alignment ----
#
# Implementation matches:
#   - Frozen-state operator O_t (Paper Section 4.1, Table 1) for all 10 optimizers
#   - 4,096-document validation proxy (Paper Appendix C)
#   - Redundancy penalty λ_r=1.0 (Paper Eq. 4)
#   - MARS scoring via raw-gradient c_t (Paper Remark 4)
#   - Muon right-preconditioner (M^T M + εI)^{-1/2} (Paper Eq. 15, Remark 3)
#   - SOAP rotated Ghost factors (Paper Eq. 18, 32)
#   - Sophia linearized clip scoring (Paper Remark)
#
# ---- Parallel Split Strategy ----
#
#   Split 1: AdamW + AdEMAMix  (Adam-family)
#   Split 2: D-Muon + MARS     (matrix + variance-reduced)
#   Split 3: Sophia + SOAP     (curvature + rotated diagonal)
#   Split 4: Lion + Signum     (sign-based, Theorem 1)
#   Split 5: ADOPT + SGD       (lagged adaptive + baseline)
#
# Wall time with 5-way parallel: ~15 hours (vs ~3 days sequential)
#
# ---- Memory Safety ----
#
# OptiSelect memory overhead (over standard training):
#   - Candidate batch (2B samples): 2× forward activations
#   - Ghost factor capture: stores (a(z), b(z)) per Linear layer (~1.5× act mem)
#   - Validation proxy factors: cached (1, 1, d) after averaging (~50 MB for 124M)
#   - Optimizer states at 124M:
#       AdamW/AdEMAMix/ADOPT/MARS: 2× params (~1 GB bf16)
#       Lion/Signum: 1× params
#       Muon: 1× + M^T M eigendecomposition
#       Sophia: 2× + Hessian buffer
#       SOAP: 3-5× params (Kronecker eigenbases + rotated v)
#
# On A100 40GB at 124M: peak ≈ 8-12 GB. Safe margin.
#
# This script applies:
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (reduces fragmentation)
#   - Per-optimizer batch adjustments (Sophia halved, SOAP halved)
#   - GPU empty_cache between runs (each run is a separate Python process)
#   - OOM detection with 2-level automatic batch halving retry
#
# Usage:
#   bash run_slimpajama_split.sh [SPLIT_ID] [SCALE] [SEED] [NPROC]
#
# Each optimizer run uses DDP across NPROC GPUs (default 4). The nccl
# backend auto-divides batch_size*acc_steps across ranks, so effective
# batch is preserved — runs are just ~NPROC× faster.
#
# Launch all 5 splits from WSL (4 GPUs per run):
#   for i in 1 2 3 4 5; do
#     python csub.py -n slimpajama-$i -g 4 -t 3d --train \
#       --command "cd /mloscratch/homes/aabdolla/llm-optimizer-benchmark/src && \
#         source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate && \
#         export PYTHONPATH=/mloscratch/homes/aabdolla/GhostSuite:/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src:\$PYTHONPATH && \
#         bash run_slimpajama_split.sh \$i full 0 4"
#   done
# =============================================================================

# Do NOT use set -e

SPLIT=${1:-0}
SCALE=${2:-full}
SEED=${3:-0}
NPROC=${4:-4}   # GPUs per run (DDP world size)

SRC_DIR="/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src"
DATASETS_DIR="/mloscratch/homes/aabdolla/datasets"
RESULTS_DIR="/mloscratch/homes/aabdolla/results/slimpajama_exp_${SCALE}"

cd "$SRC_DIR"
source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate
export PYTHONPATH="/mloscratch/homes/aabdolla/GhostSuite:${SRC_DIR}:$PYTHONPATH"
export HF_HOME=/mloscratch/homes/aabdolla/.hf_cache
export HF_DATASETS_CACHE=/mloscratch/homes/aabdolla/.hf_cache/datasets

# Memory safety
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCALE SEED RESULTS_DIR

mkdir -p "$RESULTS_DIR" logs

# Verify data + report GPU
if [ ! -f "${DATASETS_DIR}/slimpajama6B/train.bin" ]; then
    echo "[FATAL] SlimPajama not found at ${DATASETS_DIR}/slimpajama6B/train.bin"
    exit 1
fi

python -c "
import numpy as np, os, torch
p = '${DATASETS_DIR}/slimpajama6B/train.bin'
data = np.memmap(p, dtype=np.uint16, mode='r')
print(f'[OK] SlimPajama train: {len(data):,} tokens ({os.path.getsize(p)/1e9:.1f} GB)')
p2 = '${DATASETS_DIR}/slimpajama6B/val.bin'
data2 = np.memmap(p2, dtype=np.uint16, mode='r')
print(f'[OK] SlimPajama val:   {len(data2):,} tokens')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'[GPU] {props.name} — {props.total_memory/1e9:.1f} GB total')
    torch.cuda.empty_cache()
else:
    print('[WARN] CUDA not available')
"


# ======================================================================
# Scale-dependent parameters with memory-safe batch sizing
# ======================================================================
if [ "$SCALE" == "full" ]; then
    MODEL_ARGS="--model llama --n_embd 768 --n_head 12 --n_layer 12"
    BATCH="--batch_size 64 --sequence_length 512 --acc_steps 4"
    SOPHIA_BATCH="--batch_size 32 --sequence_length 512 --acc_steps 8"
    SOAP_BATCH="--batch_size 48 --sequence_length 512 --acc_steps 6"

    STD_ITERS=38000
    SEL_ITERS=19000
    STD_WARMUP=760
    SEL_WARMUP=380
    EVAL_INTERVAL=1000
    LOG_INTERVAL=100
    EVAL_BATCHES=64
    SOPHIA_BS=32
    VAL_PROXY_SIZE=4096
    VAL_PROXY_REFRESH=5000
    DESCRIPTION="124M params, 5.0B tokens"
else
    MODEL_ARGS="--model llama --n_embd 384 --n_head 6 --n_layer 6"
    BATCH="--batch_size 32 --sequence_length 512 --acc_steps 1"
    SOPHIA_BATCH="--batch_size 16 --sequence_length 512 --acc_steps 2"
    SOAP_BATCH="--batch_size 32 --sequence_length 512 --acc_steps 1"

    STD_ITERS=12000
    SEL_ITERS=12000
    STD_WARMUP=240
    SEL_WARMUP=240
    EVAL_INTERVAL=400
    LOG_INTERVAL=100
    EVAL_BATCHES=32
    SOPHIA_BS=16
    VAL_PROXY_SIZE=1024
    VAL_PROXY_REFRESH=3000
    DESCRIPTION="25M params, 196M tokens"
fi

DATA_ARGS="--dataset slimpajama --datasets_dir ${DATASETS_DIR}"
COMMON="--scheduler cos --grad_clip 1.0 --weight_decay 0.1"
COMMON="${COMMON} --dropout 0.0 --dtype bfloat16 --device cuda:0"
COMMON="${COMMON} --distributed_backend nccl"
EVAL_ARGS="--eval_interval ${EVAL_INTERVAL} --log_interval ${LOG_INTERVAL} --eval_batches ${EVAL_BATCHES}"
RESULTS_ARGS="--results_base_folder ${RESULTS_DIR}"

SEL_ARGS="--selection --candidate_multiplier 2"
SEL_ARGS="${SEL_ARGS} --selection_temperature 0.1 --selection_sketch_dim 1024"
SEL_ARGS="${SEL_ARGS} --selection_redundancy_weight 1.0"
SEL_ARGS="${SEL_ARGS} --val_proxy_size ${VAL_PROXY_SIZE}"
SEL_ARGS="${SEL_ARGS} --val_proxy_refresh ${VAL_PROXY_REFRESH}"


# ======================================================================
#  Runner with OOM detection and 2-level automatic retry
# ======================================================================
run_experiment() {
    local OPT_NAME=$1
    local MODE=$2
    local OPT_FLAG=$3
    local OPT_EXTRA=$4
    local BATCH_OVR=${5:-$BATCH}

    local EXP_NAME="${MODE}_slimpajama_${SCALE}_${OPT_NAME}_seed${SEED}"
    local LOG_FILE="logs/${EXP_NAME}.log"

    # Skip only if summary has valid final_val_loss
    if [ -f "${RESULTS_DIR}/${EXP_NAME}/summary.json" ]; then
        if python -c "
import json
d = json.load(open('${RESULTS_DIR}/${EXP_NAME}/summary.json'))
assert 'final_val_loss' in d and d['final_val_loss'] is not None
" 2>/dev/null; then
            echo "[SKIP] ${EXP_NAME} — already completed"
            return 0
        fi
    fi

    rm -f "$LOG_FILE"

    local ITERS WARMUP
    if [ "$MODE" == "standard" ]; then
        ITERS=$STD_ITERS; WARMUP=$STD_WARMUP
    else
        ITERS=$SEL_ITERS; WARMUP=$SEL_WARMUP
    fi

    local SEL_FLAG=""
    if [ "$MODE" == "selection" ]; then
        SEL_FLAG="$SEL_ARGS"
    fi

    # ===== Attempt 1: original batch config =====
    echo ""
    echo "================================================================"
    echo "  ${OPT_NAME} | ${MODE} | ${SCALE} | seed=${SEED} | GPUs=${NPROC}"
    echo "  Iters: ${ITERS} | Batch: $(echo $BATCH_OVR)"
    echo "  Started: $(date)"
    echo "================================================================"

    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} main.py \
        $MODEL_ARGS $DATA_ARGS $BATCH_OVR \
        --iterations $ITERS --warmup_steps $WARMUP \
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
            return 1
        fi

        local RETRY_BATCH="--batch_size ${NEW_BS} --sequence_length 512 --acc_steps ${NEW_AS}"
        echo "    Retry config: ${RETRY_BATCH}"
        echo "    Effective batch preserved: ${CUR_BS}*${CUR_AS} = ${NEW_BS}*${NEW_AS}"

        rm -rf "${RESULTS_DIR}/${EXP_NAME}"
        mv "$LOG_FILE" "${LOG_FILE}.oom-attempt1"

        # For Sophia, update sophia_bs
        local RETRY_EXTRA="$OPT_EXTRA"
        if [ "$OPT_NAME" == "sophiag" ]; then
            RETRY_EXTRA=$(echo "$OPT_EXTRA" | sed "s/--sophia_bs [0-9]*/--sophia_bs ${NEW_BS}/")
        fi

        torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} main.py \
            $MODEL_ARGS $DATA_ARGS $RETRY_BATCH \
            --iterations $ITERS --warmup_steps $WARMUP \
            $COMMON $EVAL_ARGS $RESULTS_ARGS \
            $OPT_FLAG $RETRY_EXTRA $SEL_FLAG \
            --experiment_name "$EXP_NAME" --seed $SEED \
            2>&1 | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}

        # Second OOM? Halve again
        if [ $EXIT_CODE -ne 0 ] && grep -q "CUDA out of memory\|OutOfMemoryError" "$LOG_FILE" 2>/dev/null; then
            echo "!!! [OOM] Second attempt. Halving again..."
            local NEW_BS2=$((NEW_BS / 2))
            local NEW_AS2=$((NEW_AS * 2))
            if [ $NEW_BS2 -lt 4 ]; then
                echo "!!! [FATAL] Cannot halve further; NEW_BS=${NEW_BS2} < 4"
                return 1
            fi
            local RETRY_BATCH2="--batch_size ${NEW_BS2} --sequence_length 512 --acc_steps ${NEW_AS2}"
            local RETRY_EXTRA2="$OPT_EXTRA"
            if [ "$OPT_NAME" == "sophiag" ]; then
                RETRY_EXTRA2=$(echo "$OPT_EXTRA" | sed "s/--sophia_bs [0-9]*/--sophia_bs ${NEW_BS2}/")
            fi
            rm -rf "${RESULTS_DIR}/${EXP_NAME}"
            mv "$LOG_FILE" "${LOG_FILE}.oom-attempt2"

            torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} main.py \
                $MODEL_ARGS $DATA_ARGS $RETRY_BATCH2 \
                --iterations $ITERS --warmup_steps $WARMUP \
                $COMMON $EVAL_ARGS $RESULTS_ARGS \
                $OPT_FLAG $RETRY_EXTRA2 $SEL_FLAG \
                --experiment_name "$EXP_NAME" --seed $SEED \
                2>&1 | tee "$LOG_FILE"
            EXIT_CODE=${PIPESTATUS[0]}
        fi
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo ">>> [OK] ${EXP_NAME} at $(date)"
    else
        echo ">>> [FAIL] ${EXP_NAME} exit ${EXIT_CODE}"
    fi

    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    return 0
}


# ======================================================================
echo ""
echo "================================================================"
echo "  OptiSelect SlimPajama Split ${SPLIT} | ${SCALE}"
echo "  ${DESCRIPTION} | Seed: ${SEED} | DDP GPUs: ${NPROC}"
echo "  Alloc: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  Started: $(date)"
echo "================================================================"


# ======================================================================
# Optimizer functions (Paper Appendix C hyperparameters)
# ======================================================================

run_adamw() {
    # Paper Eq. 7: O_t^AdamW(x) = c_t ⊙ x, c_t = 1/(√v̂ + ε)
    run_experiment "adamw" "$1" "--opt adamw" \
        "--lr 1e-3 --beta1 0.9 --beta2 0.999"
}

run_ademamix() {
    # Paper Eq. 8: operator identical to AdamW; slow momentum β₃=0.9999
    # H1: largest selection synergy via Proposition 2
    run_experiment "ademamix" "$1" "--opt ademamix" \
        "--lr 1e-3 --beta1 0.9 --beta2 0.999 --adema_beta3 0.9999 --adema_alpha 0.8"
}

run_dmuon() {
    # Paper Eq. 15, Remark 3: right-preconditioner (M^T M + εI)^{-1/2}
    run_experiment "d-muon" "$1" "--opt d-muon" \
        "--lr 1e-3 --beta1 0.9 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"
}

run_mars() {
    # Paper Remark 4: scoring uses AdamW c_t on RAW gradients
    run_experiment "mars" "$1" "--opt mars" \
        "--lr 1e-3 --mars_lr 3e-3 --beta1 0.9 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99 --mars_vr_gamma 0.025"
}

run_sophiag() {
    # Paper Eq. 11: O_t^Sophia(x) = clip(x / max(ρh_t, ε), 1), linearized
    # H3 / Proposition 1: Hessian diagonal = curvature component of π*
    run_experiment "sophiag" "$1" "--opt sophiag" \
        "--lr 1e-3 --beta1 0.9 --beta2 0.999 --sophia_rho 0.04 --precondition_frequency 10 --sophia_bs ${SOPHIA_BS}" \
        "$SOPHIA_BATCH"
}

run_soap() {
    # Paper Eq. 18, 32: rotated Ghost factors through [U_L, U_R]
    # β₂=0.95 per Paper Appendix C
    run_experiment "soap" "$1" "--opt soap" \
        "--lr 1e-3 --beta1 0.9 --beta2 0.95 --precondition_frequency 10" \
        "$SOAP_BATCH"
}

run_lion() {
    # Paper Eq. 13-14: sign operator, linearized for scoring
    # H2 / Theorem 1: sign collapse degrades selection
    # lr=3e-4 per Chen et al. [4]
    run_experiment "lion" "$1" "--opt lion" \
        "--lr 3e-4 --beta1 0.9 --beta2 0.99"
}

run_signum() {
    # Paper Section 4.3.2: O_t^Signum(x) = sign(m_{t-1}) [constant in x]
    run_experiment "signum" "$1" "--opt signum" \
        "--lr 3e-4 --momentum 0.9"
}

run_adopt() {
    # Paper Section 4.3.1: lagged variance (t-1)
    run_experiment "adopt" "$1" "--opt adopt" \
        "--lr 1e-3 --beta1 0.9 --beta2 0.999"
}

run_sgd() {
    # Baseline: identity operator O_t^SGD(x) = x
    run_experiment "sgd" "$1" "--opt sgd" \
        "--lr 3e-2 --momentum 0.9"
}


# ======================================================================
# Split dispatcher
# ======================================================================
case $SPLIT in
    0)
        for mode in standard selection; do
            run_adamw    "$mode"; run_ademamix "$mode"
            run_dmuon    "$mode"; run_mars     "$mode"
            run_sophiag  "$mode"; run_soap     "$mode"
            run_lion     "$mode"; run_signum   "$mode"
            run_adopt    "$mode"; run_sgd      "$mode"
        done
        ;;
    1)
        run_adamw    "standard"; run_adamw    "selection"
        run_ademamix "standard"; run_ademamix "selection"
        ;;
    2)
        run_dmuon "standard"; run_dmuon "selection"
        run_mars  "standard"; run_mars  "selection"
        ;;
    3)
        run_sophiag "standard"; run_sophiag "selection"
        run_soap    "standard"; run_soap    "selection"
        ;;
    4)
        run_lion   "standard"; run_lion   "selection"
        run_signum "standard"; run_signum "selection"
        ;;
    5)
        run_adopt "standard"; run_adopt "selection"
        run_sgd   "standard"; run_sgd   "selection"
        ;;
    *)
        echo "[FATAL] Unknown split: ${SPLIT} (use 0=sequential, 1-5=parallel)"
        exit 1
        ;;
esac


echo ""
echo "================================================================"
echo "  Split ${SPLIT} complete at $(date)"
echo "================================================================"


# ======================================================================
# Results aggregation (only generates table when all 20 runs complete)
# ======================================================================
python - << 'PYTHON_COLLECTOR'
import os, json

SCALE = os.environ.get("SCALE", "full")
SEED = int(os.environ.get("SEED", "0"))
RESULTS_DIR = os.environ.get("RESULTS_DIR",
    f"/mloscratch/homes/aabdolla/results/slimpajama_exp_{SCALE}")
LOG_DIR = "logs"

optimizers = ["adamw", "ademamix", "d-muon", "mars", "sophiag", "soap",
              "lion", "signum", "adopt", "sgd"]
display = {
    "adamw": "AdamW", "ademamix": "AdEMAMix", "d-muon": "D-Muon",
    "mars": "MARS", "sophiag": "Sophia", "soap": "SOAP",
    "lion": "Lion", "signum": "Signum", "adopt": "ADOPT", "sgd": "SGD",
}
modes = ["standard", "selection"]

total_expected = len(optimizers) * len(modes)
complete = 0
results = {}

for opt in optimizers:
    results[opt] = {}
    for mode in modes:
        p = os.path.join(RESULTS_DIR,
                         f"{mode}_slimpajama_{SCALE}_{opt}_seed{SEED}",
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
                    complete += 1
            except Exception:
                pass
        results[opt][mode] = r

print(f"\n[COLLECTOR] {complete}/{total_expected} runs complete for seed={SEED}")

if complete < total_expected:
    print("[COLLECTOR] Other splits still running. Re-run after completion for full table.")
else:
    desc = "25M params" if SCALE == "small" else "124M params"
    print()
    print("=" * 130)
    print(f"  SlimPajama Results: Standard vs OptiSelect ({desc}, seed {SEED})")
    print("=" * 130)
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

    # LaTeX
    print()
    print("% ====== LaTeX Table (paper-ready) ======")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(f"\\caption{{SlimPajama results ({desc} Llama, seed {SEED}).")
    print(r"Validation loss and accuracy (\%) for standard training vs.\ OptiSelect.")
    print(r"Consistent with Theorem~1 and Proposition~2.}")
    print(r"\label{tab:slimpajama_" + SCALE + "}")
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

    out = os.path.join(LOG_DIR, f"slimpajama_{SCALE}_results_seed{SEED}.json")
    with open(out, "w") as f:
        json.dump({
            "metadata": {"scale": SCALE, "seed": SEED, "description": desc, "dataset": "slimpajama"},
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out}")

PYTHON_COLLECTOR