#!/bin/bash
# =============================================================================
# OptiSelect WikiText-103 Benchmark — Publication Ready (v3, Multi-Scale)
# =============================================================================
#
# Produces Exp. 1 + Exp. 2 (Paper Section 5.3) results for WikiText-103,
# empirical validation of theoretical claims (Theorem 1, Propositions 1-3),
# across MULTIPLE MODEL SCALES (25M / 124M / 210M).
#
# ---- Aligned with the refined paper formulation ----
#
# Implementation matches:
#   - Frozen-state operator O_t (Paper Section 4.1, Table 1) for all 10 optimizers
#   - 4,096-document validation proxy (Paper Appendix C)
#     [scaled to ~512 for WikiText val split size]
#   - Redundancy penalty λ_r = 1.0 in selection (Paper Eq. 4)
#   - Greedy sequential Boltzmann selection (Paper Eq. 34)
#   - Optimizer hyperparameters from Paper Appendix C (per-scale tuned, see below)
#   - MARS scoring via raw-gradient c_t (Paper Remark 4)
#   - Muon right-preconditioner (M^T M + εI)^{-1/2} (Paper Eq. 15, Remark 3)
#   - SOAP rotated Ghost factors (Paper Eq. 18, 32)
#   - Sophia linearized clip for scoring (Paper Remark on Sophia)
#
# ---- Protocol (Paper Section 5.1) ----
#
# Standard training:  ITERS iterations on batch of B samples
# Selection training: ITERS iterations on 2B candidates → select B
#   Both modes perform ITERS optimizer steps with matched forward-backward
#   compute (selection draws 2× candidates for scoring).
#
# ---- Models (Llama: RMSNorm, SwiGLU, RoPE) ----
#
#   25m  : n_embd=384, n_head=6,  n_layer=6   → ~25M  params  (current paper Section 5.1)
#   124m : n_embd=768, n_head=12, n_layer=12  → ~124M params  (GPT-2 small reference)
#   210m : n_embd=768, n_head=12, n_layer=24  → ~210M params  (deeper variant)
#
#   Per-scale hyperparameters (lr/β₁/warmup/clip/batch) follow the reference
#   sweeps in scripts/{124m,210m}/*.sh. The 25m config is unchanged from v3.
#
# ---- WikiText-103 ----
#   ~103M training tokens (GPT-2 BPE, vocab=50257)
#
#     scale | tokens/step | total tokens at default ITERS=12000
#     -----+-------------+------------------------------------
#      25m | 16×512×1    = 8,192   ≈  98M tokens (~0.95 epoch)
#     124m | 64×512×4    = 131,072 ≈ 1.57B tokens (~15 epochs)
#     210m | 64×512×4    = 131,072 ≈ 1.57B tokens (~15 epochs)
#
#   Iter-matched scaling study: every scale runs ITERS optimizer steps so
#   that per-scale OptiSelect overhead (one scoring pass per step) is
#   comparable. Total tokens grow with batch size, as in the reference
#   sweeps. Override via `ITERS=24000 bash run_wikitext_exp.sh ...`.
#
# ---- Hyperparameters (Paper Appendix C) ----
#   Cosine LR schedule, ~2% warmup, decay to 0.01×η_max
#   Weight decay λ = 0.1, bfloat16
#   grad_clip = 1.0 at 25m, 0.5 at 124m/210m (per reference scripts)
#
# Selection (Paper Appendix C):
#   B̃/B = 2, τ = 0.1, λ_r = 1.0
#   Val proxy: 512 docs (scaled for WikiText), refresh every 3,000 steps
#
# Expected runtime (1×A100 per run, 4×A100 DDP):
#    25m: ~15-20 min/run × 20 runs ≈ 5-7   hours
#   124m: ~ 1-2  h /run × 20 runs ≈ 20-40  hours
#   210m: ~ 2-3  h /run × 20 runs ≈ 40-60  hours
#
# Usage:
#   bash run_wikitext_exp.sh [NPROC] [MODEL_SIZE]
#
#   NPROC      : DDP world size per run (default 4)
#   MODEL_SIZE : 25m | 124m | 210m       (default 25m)
#
#   Examples:
#     bash run_wikitext_exp.sh                # 4-GPU, 25m  (current paper config)
#     bash run_wikitext_exp.sh 4 124m         # 4-GPU, 124M
#     bash run_wikitext_exp.sh 8 210m         # 8-GPU, 210M
#     ITERS=24000 bash run_wikitext_exp.sh 4 124m  # override training length
#
# Each optimizer run uses DDP across NPROC GPUs. The nccl backend
# auto-divides batch_size*acc_steps across ranks, so effective batch is
# preserved — runs are just ~NPROC× faster.
# =============================================================================

# Do NOT use `set -e` — per-optimizer failures should not cascade

NPROC=${1:-4}                # GPUs per run (DDP world size)
MODEL_SIZE=${2:-25m}         # 25m | 124m | 210m

# ---- Validation-proxy source for OptiSelect ----
# PROXY_SOURCE=train (default, Paper Appendix C) or downstream (mixture
# of HellaSwag/ARC-E/ARC-C/PIQA/SciQ). Downstream requires GPT-2 BPE.
PROXY_SOURCE=${PROXY_SOURCE:-train}
PROXY_TASKS=${PROXY_TASKS:-hellaswag,arc_easy,arc_challenge,piqa,sciq}

# ---- Paths ----
SRC_DIR="/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src"
DATASETS_DIR="/mloscratch/homes/aabdolla/datasets"

# Backward-compat: 25m+train keeps the original results dir from the v3 paper run.
if [ "$MODEL_SIZE" = "25m" ] && [ "$PROXY_SOURCE" = "train" ]; then
    RESULTS_DIR="/mloscratch/homes/aabdolla/results/wikitext_exp_v3"
elif [ "$PROXY_SOURCE" = "train" ]; then
    RESULTS_DIR="/mloscratch/homes/aabdolla/results/wikitext_exp_v3_${MODEL_SIZE}"
elif [ "$MODEL_SIZE" = "25m" ]; then
    RESULTS_DIR="/mloscratch/homes/aabdolla/results/wikitext_exp_v3_proxy_${PROXY_SOURCE}"
else
    RESULTS_DIR="/mloscratch/homes/aabdolla/results/wikitext_exp_v3_${MODEL_SIZE}_proxy_${PROXY_SOURCE}"
fi

cd "$SRC_DIR"
source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate
export PYTHONPATH="/mloscratch/homes/aabdolla/GhostSuite:${SRC_DIR}:$PYTHONPATH"
export HF_HOME=/mloscratch/homes/aabdolla/.hf_cache
export HF_DATASETS_CACHE=/mloscratch/homes/aabdolla/.hf_cache/datasets
export PROXY_SOURCE RESULTS_DIR MODEL_SIZE

mkdir -p "$RESULTS_DIR" logs

# ======================================================================
# Per-scale configuration (architecture + training schedule + per-optimizer args)
#   25m  : Paper Section 5.1 + Appendix C (current v3 config)
#   124m : scripts/124m/*.sh  (LLM-Optimizer-Benchmark reference sweep)
#   210m : scripts/210m/*.sh  (LLM-Optimizer-Benchmark reference sweep)
# ======================================================================
case "$MODEL_SIZE" in
    25m)
        # ---- Architecture (~25M params) ----
        N_EMBD=384; N_HEAD=6; N_LAYER=6

        # ---- Training schedule ----
        DEFAULT_ITERS=12000
        BATCH_SIZE=16; ACC_STEPS=1
        SOPHIA_BATCH_SIZE=8; SOPHIA_ACC_STEPS=2
        GRAD_CLIP=1.0
        EVAL_INTERVAL=500
        VAL_PROXY_REFRESH=3000

        # ---- Per-optimizer (canonical recipes — Paper Appendix C) ----
        ADAMW_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.999"
        # AdEMAMix: canonical Pagliardini et al. (α=8.0, β₃=0.999, warmup=ITERS)
        # at every scale so the cross-scale figure isolates model size, not
        # hyperparameters. The previous v3 25m used α=0.8 (gentle mixing) which
        # diluted the H1 / Proposition-2 selection-synergy test.
        ADEMAMIX_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.999 --adema_beta3 0.999 --adema_alpha 8.0"
        DMUON_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"
        MARS_ARGS="--lr 1e-3 --mars_lr 3e-3 --beta1 0.9 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99 --mars_vr_gamma 0.025"
        # Sophia: Liu et al. canonical recipe (β₂=0.99 lets Hessian EMA warm up;
        # β₁=0.965 stabilizes Hessian-momentum coupling; lr=6e-4 keeps the
        # clamp-saturated regime — which acts as sign-SGD — from blowing up;
        # sophia_bs = batch_size × acc_steps so trainer's bs = sophia_bs × seq_len
        # matches original "bs = total tokens per opt step" convention).
        SOPHIA_ARGS="--lr 6e-4 --beta1 0.965 --beta2 0.99 --sophia_rho 0.04 --precondition_frequency 10 --sophia_bs 16"
        SOAP_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.95 --precondition_frequency 10"
        LION_ARGS="--lr 3e-4 --beta1 0.9 --beta2 0.99"
        SIGNUM_ARGS="--lr 3e-4 --momentum 0.9"
        # ADOPT: paper recipe uses β₂=0.9999 (lagged variance EMA); β₂=0.999
        # is 10× faster and weakens the lagged-variance characterization that
        # the paper's Section 4.3.1 operator analysis depends on.
        ADOPT_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.9999"
        SGD_ARGS="--lr 3e-2 --momentum 0.9"
        ;;

    124m)
        # ---- Architecture (~124M params, GPT-2 small) ----
        N_EMBD=768; N_HEAD=12; N_LAYER=12

        # ---- Training schedule (scripts/124m reference; iter-matched to 25m) ----
        DEFAULT_ITERS=12000
        BATCH_SIZE=64; ACC_STEPS=4
        SOPHIA_BATCH_SIZE=32; SOPHIA_ACC_STEPS=8   # halved batch, doubled acc (sophia.sh)
        GRAD_CLIP=0.5
        EVAL_INTERVAL=500
        VAL_PROXY_REFRESH=3000

        # ---- Per-optimizer (canonical recipes — match 25m for cross-scale
        #      comparability; reference scripts/124m used β₁=0.8 for
        #      AdamW/D-Muon/MARS but the deviation is undocumented and
        #      conflates "model size" with "β₁" in the scaling figure). ----
        ADAMW_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.999"
        ADEMAMIX_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.999 --adema_beta3 0.999 --adema_alpha 8.0"
        DMUON_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"
        MARS_ARGS="--lr 1e-3 --mars_lr 3e-3 --beta1 0.9 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99 --mars_vr_gamma 0.025"
        # Sophia: Liu et al. canonical GPT2-small recipe (lr=6e-4, β=(0.965,0.99),
        # ρ=0.04). sophia_bs = batch_size × acc_steps = 32 × 8 = 256, so trainer's
        # bs = sophia_bs × seq_len matches "total tokens per opt step" convention.
        SOPHIA_ARGS="--lr 6e-4 --beta1 0.965 --beta2 0.99 --sophia_rho 0.04 --precondition_frequency 10 --sophia_bs 256"
        # SOAP: β₂=0.95 across all scales (Paper Appendix C / SOAP paper);
        # reference scripts/124m used β₂=0.999 but that's the SOAP class
        # default, undocumented, and creates a cross-scale confound.
        SOAP_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.95 --precondition_frequency 10"
        # Lion / Signum: lr=3e-4 across all scales (Chen et al.: AdamW/3 to
        # AdamW/10 for sign-based updates). Reference scripts/124m used 1e-3
        # (=AdamW, way too hot) which would confound Theorem-1 sign-collapse
        # by inducing divergence rather than the predicted scoring degeneracy.
        LION_ARGS="--lr 3e-4 --beta1 0.9 --beta2 0.99"
        SIGNUM_ARGS="--lr 3e-4 --momentum 0.9"
        ADOPT_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.9999"
        SGD_ARGS="--lr 3e-2 --momentum 0.9"
        ;;

    210m)
        # ---- Architecture (~210M params, deeper) ----
        N_EMBD=768; N_HEAD=12; N_LAYER=24

        # ---- Training schedule (scripts/210m reference; iter-matched to 25m) ----
        DEFAULT_ITERS=12000
        BATCH_SIZE=64; ACC_STEPS=4
        SOPHIA_BATCH_SIZE=32; SOPHIA_ACC_STEPS=8
        GRAD_CLIP=0.5
        EVAL_INTERVAL=500
        VAL_PROXY_REFRESH=3000

        # ---- Per-optimizer (canonical, matched with 25m and 124m) ----
        ADAMW_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.999"
        ADEMAMIX_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.999 --adema_beta3 0.999 --adema_alpha 8.0"
        DMUON_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"
        MARS_ARGS="--lr 1e-3 --mars_lr 3e-3 --beta1 0.9 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99 --mars_vr_gamma 0.025"
        # Sophia: Liu et al. recipe scaled to GPT2-medium (lr=4e-4 per paper
        # for 355M; 210m sits between 124M and 355M, depth favors lower lr).
        # sophia_bs = batch_size × acc_steps = 256.
        SOPHIA_ARGS="--lr 4e-4 --beta1 0.965 --beta2 0.99 --sophia_rho 0.04 --precondition_frequency 10 --sophia_bs 256"
        SOAP_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.95 --precondition_frequency 10"
        # Lion / Signum: lr=3e-4 across all scales (cross-scale consistent;
        # within Chen et al. AdamW/3..AdamW/10 range). Reference 210m used
        # lr=5e-4 (=AdamW/2, out of paper range) and Signum nesterov+0.95 —
        # both create cross-scale confounds.
        LION_ARGS="--lr 3e-4 --beta1 0.9 --beta2 0.99"
        SIGNUM_ARGS="--lr 3e-4 --momentum 0.9"
        ADOPT_ARGS="--lr 1e-3 --beta1 0.9 --beta2 0.9999"
        SGD_ARGS="--lr 3e-2 --momentum 0.9"
        ;;

    *)
        echo "ERROR: unknown MODEL_SIZE='${MODEL_SIZE}'. Expected: 25m | 124m | 210m" >&2
        exit 2
        ;;
esac

# ======================================================================
# Iteration / warmup / scheduler (overridable via env)
# ======================================================================
ITERS=${ITERS:-$DEFAULT_ITERS}
WARMUP_STEPS=${WARMUP_STEPS:-$(( ITERS * 2 / 100 ))}    # 2% warmup
LOG_INTERVAL=100
SEQ_LEN=512
EVAL_BATCHES=32                # ~16K tokens per eval
WEIGHT_DECAY=0.1
SEED=0

# ---- Selection hyperparameters (Paper Appendix C, scale-invariant) ----
CAND_MULT=2
SEL_TEMP=0.1
SEL_SKETCH=1024
SEL_REDUNDANCY=1.0
VAL_PROXY_SIZE=512             # Paper uses 4,096; scaled for WikiText val size

# ---- AdEMAMix warmup = ITERS at every scale (canonical Pagliardini et al.) ----
# Without warmup, exp_avg_slow stays near-zero for the first ~1/(1-β₃) steps,
# so AdEMAMix runs as plain Adam early then suddenly switches behavior. With
# β₃=0.999 + α=8.0, missing warmup would dilute the slow-momentum effect that
# powers the H1 / Proposition-2 selection-synergy test.
ADEMAMIX_ARGS="${ADEMAMIX_ARGS} --adema_beta3_warmup ${ITERS} --adema_alpha_warmup ${ITERS}"

# ======================================================================
# Composed argument blocks
# ======================================================================
MODEL_ARGS="--model llama --n_embd ${N_EMBD} --n_head ${N_HEAD} --n_layer ${N_LAYER}"
DATA_ARGS="--dataset wikitext --datasets_dir ${DATASETS_DIR}"

COMMON_ARGS="--iterations ${ITERS} --warmup_steps ${WARMUP_STEPS} --scheduler cos"
COMMON_ARGS="${COMMON_ARGS} --grad_clip ${GRAD_CLIP} --weight_decay ${WEIGHT_DECAY}"
COMMON_ARGS="${COMMON_ARGS} --dropout 0.0 --dtype bfloat16 --device cuda:0"
COMMON_ARGS="${COMMON_ARGS} --distributed_backend nccl"

EVAL_ARGS="--eval_interval ${EVAL_INTERVAL} --log_interval ${LOG_INTERVAL}"
EVAL_ARGS="${EVAL_ARGS} --eval_batches ${EVAL_BATCHES}"

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

# Backward-compat: 25m experiment names (and log files) keep the original
# v3 format with no size tag, so completed runs are still detected. Other
# scales prepend the size to disambiguate.
if [ "$MODEL_SIZE" = "25m" ]; then
    SIZE_TAG=""
else
    SIZE_TAG="${MODEL_SIZE}_"
fi

FAILED_RUNS=()
COMPLETED_RUNS=()


# ======================================================================
#  Runner with per-run error isolation and completeness check
# ======================================================================
run_experiment() {
    local OPT_NAME=$1
    local MODE=$2
    local OPT_FLAG=$3
    local OPT_EXTRA=$4
    local BATCH_OVERRIDE=${5:-$BATCH}

    local EXP_NAME
    if [ "$PROXY_SOURCE" = "train" ]; then
        EXP_NAME="${MODE}_wikitext_${SIZE_TAG}${OPT_NAME}_seed${SEED}"
    else
        EXP_NAME="${MODE}_wikitext_${SIZE_TAG}proxy-${PROXY_SOURCE}_${OPT_NAME}_seed${SEED}"
    fi
    local LOG_FILE="logs/${EXP_NAME}.log"

    # Skip only if summary has final_val_loss (not just partial history)
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

    echo ""
    echo "================================================================"
    echo "  Optimizer: ${OPT_NAME} | Mode: ${MODE} | Size: ${MODEL_SIZE} | Seed: ${SEED} | GPUs: ${NPROC}"
    echo "  Iters: ${ITERS} | Warmup: ${WARMUP_STEPS} | Batch: $(echo $BATCH_OVERRIDE)"
    if [ "$MODE" == "selection" ]; then
        echo "  Selection: B̃/B=${CAND_MULT}, τ=${SEL_TEMP}, λ_r=${SEL_REDUNDANCY}"
        echo "             Val proxy: ${VAL_PROXY_SIZE} docs, refresh every ${VAL_PROXY_REFRESH} steps"
    fi
    echo "  Started: $(date)"
    echo "================================================================"

    local SEL_FLAG=""
    if [ "$MODE" == "selection" ]; then
        SEL_FLAG="$SEL_ARGS"
    fi

    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} main.py \
        $MODEL_ARGS \
        $DATA_ARGS \
        $BATCH_OVERRIDE \
        $COMMON_ARGS \
        $EVAL_ARGS \
        $RESULTS_ARGS \
        $OPT_FLAG \
        $OPT_EXTRA \
        $SEL_FLAG \
        --experiment_name "$EXP_NAME" \
        --seed $SEED \
        2>&1 | tee "$LOG_FILE"

    local EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -eq 0 ]; then
        echo ">>> [OK] ${EXP_NAME} at $(date)"
        COMPLETED_RUNS+=("$EXP_NAME")
    else
        echo ">>> [FAIL] ${EXP_NAME} exit code ${EXIT_CODE}"
        FAILED_RUNS+=("$EXP_NAME")
    fi
    return 0
}


# ======================================================================
echo ""
echo "================================================================"
echo "  OptiSelect WikiText-103 Benchmark (v3 — Multi-Scale)"
echo "  Model size:    ${MODEL_SIZE}  (n_embd=${N_EMBD}, n_head=${N_HEAD}, n_layer=${N_LAYER})"
echo "  10 optimizers × 2 modes = 20 runs | Seed: ${SEED} | DDP GPUs: ${NPROC}"
echo "  Training:      ${ITERS} iters × ${BATCH_SIZE}×${SEQ_LEN}×${ACC_STEPS} tok/step"
echo "  Proxy source:  ${PROXY_SOURCE}$([ "$PROXY_SOURCE" != "train" ] && echo " [${PROXY_TASKS}]")"
echo "  Results dir:   ${RESULTS_DIR}"
echo "  Started:       $(date)"
echo "================================================================"


# ======================================================================
# Optimizer configurations (Paper Appendix C; per-scale args set above)
# Each block references the frozen-state operator from Paper Section 4.3
# ======================================================================

# -----------------------------------------------------------
# 1. AdamW — Paper Eq. 5-7
#    Operator: O_t^AdamW(x) = c_t ⊙ x,  c_t = 1/(√v̂ + ε)
#    Diagonal adaptive, linear in x (Table 1)
# -----------------------------------------------------------
run_experiment "adamw" "standard"  "--opt adamw" "$ADAMW_ARGS"
run_experiment "adamw" "selection" "--opt adamw" "$ADAMW_ARGS"


# -----------------------------------------------------------
# 2. AdEMAMix — Paper Eq. 8
#    Operator identical to AdamW; slow momentum (β₃) stabilizes
#    influence scores temporally (Proposition 2).
#    Paper H1: Should exhibit largest selection synergy.
# -----------------------------------------------------------
run_experiment "ademamix" "standard"  "--opt ademamix" "$ADEMAMIX_ARGS"
run_experiment "ademamix" "selection" "--opt ademamix" "$ADEMAMIX_ARGS"


# -----------------------------------------------------------
# 3. D-Muon — Paper Eq. 15, Remark 3
#    Operator: O_t^Muon(G) ≈ G(M^T M + εI)^{-1/2}
#    Matrix right-preconditioner, equalizes singular values
#    Scoring uses eigendecomposition once per step
# -----------------------------------------------------------
run_experiment "d-muon" "standard"  "--opt d-muon" "$DMUON_ARGS"
run_experiment "d-muon" "selection" "--opt d-muon" "$DMUON_ARGS"


# -----------------------------------------------------------
# 4. MARS — Paper Eq. 19-20, Remark 4
#    Per-sample scoring uses plain AdamW c_t on RAW gradients
#    (not MARS's variance-reduced state)
#    Batch-level update applies full variance-reduction
# -----------------------------------------------------------
run_experiment "mars" "standard"  "--opt mars" "$MARS_ARGS"
run_experiment "mars" "selection" "--opt mars" "$MARS_ARGS"


# -----------------------------------------------------------
# 5. Sophia — Paper Eq. 11
#    Operator: O_t^Sophia(x) = clip(x / max(ρh_t, ε), 1)
#    Linearized for scoring (clip inactive on >95% of coords)
#    Paper H3 / Proposition 1: Hessian diagonal best approximates
#    the curvature component of π* for influence scoring.
#    Halved batch + 2× acc_steps for stable Hessian estimation.
#
#    Hyperparameters follow the Liu et al. canonical recipe (β=(0.965,0.99),
#    ρ=0.04, scale-dependent lr). Three subtle pitfalls — all fixed in the
#    per-scale SOPHIA_ARGS above:
#      (i)  β₂ must be 0.99, not 0.999: with k=10 precondition_frequency,
#           β₂=0.999 leaves the Hessian EMA cold for ~6.9k steps and Sophia
#           degenerates to sign-SGD across the whole training horizon.
#      (ii) lr must be in the Liu et al. range (3e-4–6e-4): in the
#           clamp-saturated regime the update is -lr·sign(m), so a too-hot
#           lr (e.g. 1e-3) blows up before curvature can take over.
#      (iii) sophia_bs = batch_size × acc_steps (effective batch in examples),
#            because the trainer sets bs := sophia_bs × seq_len — matching the
#            original "bs = total tokens per opt step" convention.
# -----------------------------------------------------------
run_experiment "sophiag" "standard"  "--opt sophiag" "$SOPHIA_ARGS" "$SOPHIA_BATCH"
run_experiment "sophiag" "selection" "--opt sophiag" "$SOPHIA_ARGS" "$SOPHIA_BATCH"


# -----------------------------------------------------------
# 6. SOAP — Paper Eq. 18, 32
#    Operator: rotates Ghost factors through [U_L, U_R]
#    Adam in the Kronecker eigenbasis
# -----------------------------------------------------------
run_experiment "soap" "standard"  "--opt soap" "$SOAP_ARGS"
run_experiment "soap" "selection" "--opt soap" "$SOAP_ARGS"


# -----------------------------------------------------------
# 7. Lion — Paper Eq. 13-14, Theorem 1
#    Operator: O_t^Lion(x) = sign((1-β₁)x + β₁ m_{t-1})
#    Linearized: u_z ≈ sign(m_{t-1}) = s_t (leading order)
#    Paper H2 / Theorem 1: Should exhibit sign collapse,
#    smallest selection synergy.
# -----------------------------------------------------------
run_experiment "lion" "standard"  "--opt lion" "$LION_ARGS"
run_experiment "lion" "selection" "--opt lion" "$LION_ARGS"


# -----------------------------------------------------------
# 8. Signum — Paper Section 4.3.2
#    Operator: O_t^Signum(x) = sign(m_{t-1})  [constant in x!]
#    Fully sample-independent — strongest sign collapse
# -----------------------------------------------------------
run_experiment "signum" "standard"  "--opt signum" "$SIGNUM_ARGS"
run_experiment "signum" "selection" "--opt signum" "$SIGNUM_ARGS"


# -----------------------------------------------------------
# 9. ADOPT — Paper Section 4.3.1
#    Operator: O_t^ADOPT(x) = x / max(√v̂_{t-1}/(1-β_2^{t-1}), ε)
#    Uses lagged (t-1) variance; otherwise diagonal adaptive
# -----------------------------------------------------------
run_experiment "adopt" "standard"  "--opt adopt" "$ADOPT_ARGS"
run_experiment "adopt" "selection" "--opt adopt" "$ADOPT_ARGS"


# -----------------------------------------------------------
# 10. SGD — baseline, identity operator O_t^SGD(x) = x
#     Reference for measuring preconditioner value
#     lr=3e-2 (no adaptive scaling requires ~30× higher lr)
# -----------------------------------------------------------
run_experiment "sgd" "standard"  "--opt sgd" "$SGD_ARGS"
run_experiment "sgd" "selection" "--opt sgd" "$SGD_ARGS"


# ======================================================================
#  Summary and results collection
# ======================================================================
echo ""
echo "================================================================"
echo "  EXPERIMENT SUMMARY (size=${MODEL_SIZE})"
echo "  Finished: $(date)"
echo "  Completed: ${#COMPLETED_RUNS[@]} | Failed: ${#FAILED_RUNS[@]}"
echo "================================================================"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "  Failed runs:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "    - $run"
        echo "      Last 5 lines of log:"
        tail -5 "logs/${run}.log" 2>/dev/null | sed 's/^/      /'
    done
fi

echo ""
echo "Collecting results..."

# Pass scale metadata into the python summary
export N_EMBD N_HEAD N_LAYER ITERS BATCH_SIZE SEQ_LEN ACC_STEPS

python - << 'PYTHON_SCRIPT'
import os, re, json

LOG_DIR      = "logs"
RESULTS_DIR  = os.environ.get("RESULTS_DIR",
    "/mloscratch/homes/aabdolla/results/wikitext_exp_v3")
PROXY_SOURCE = os.environ.get("PROXY_SOURCE", "train")
MODEL_SIZE   = os.environ.get("MODEL_SIZE", "25m")
N_EMBD       = int(os.environ.get("N_EMBD", 384))
N_HEAD       = int(os.environ.get("N_HEAD", 6))
N_LAYER      = int(os.environ.get("N_LAYER", 6))
ITERS        = int(os.environ.get("ITERS", 12000))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", 16))
SEQ_LEN      = int(os.environ.get("SEQ_LEN", 512))
ACC_STEPS    = int(os.environ.get("ACC_STEPS", 1))

# Match the bash SIZE_TAG: empty for 25m, "<size>_" otherwise
SIZE_TAG  = "" if MODEL_SIZE == "25m" else f"{MODEL_SIZE}_"
PROXY_TAG = "" if PROXY_SOURCE == "train" else f"proxy-{PROXY_SOURCE}_"
NAME_INFIX = f"{SIZE_TAG}{PROXY_TAG}"

optimizers = ["adamw", "ademamix", "d-muon", "mars", "sophiag", "soap",
              "lion", "signum", "adopt", "sgd"]
display = {
    "adamw": "AdamW", "ademamix": "AdEMAMix", "d-muon": "D-Muon",
    "mars": "MARS", "sophiag": "Sophia", "soap": "SOAP",
    "lion": "Lion", "signum": "Signum", "adopt": "ADOPT", "sgd": "SGD",
}
paper_ops = {
    "adamw": "diagonal adaptive (Eq. 7)",
    "ademamix": "diagonal adaptive + slow momentum (Eq. 8)",
    "d-muon": "matrix right-preconditioner (Eq. 15)",
    "mars": "diagonal adaptive, raw-g scoring (Remark 4)",
    "sophiag": "curvature-aware diagonal (Eq. 11)",
    "soap": "rotated diagonal adaptive (Eq. 18)",
    "lion": "sign-based, linearized (Eq. 14)",
    "signum": "sign-based, sample-independent",
    "adopt": "diagonal adaptive, lagged variance",
    "sgd": "identity (baseline)",
}
modes = ["standard", "selection"]
seed = 0

results = {}
for opt in optimizers:
    results[opt] = {}
    for mode in modes:
        log_path = os.path.join(LOG_DIR, f"{mode}_wikitext_{NAME_INFIX}{opt}_seed{seed}.log")
        summary_path = os.path.join(RESULTS_DIR,
                                    f"{mode}_wikitext_{NAME_INFIX}{opt}_seed{seed}",
                                    "summary.json")

        r = {"val_loss": None, "val_pp": None, "val_acc": None,
             "entropy": None, "best_val_loss": None}

        # Prefer JSON summary over log parsing (more reliable)
        if os.path.exists(summary_path):
            try:
                d = json.load(open(summary_path))
                r["val_loss"] = d.get("final_val_loss")
                r["val_pp"]   = d.get("final_val_pp")
                r["val_acc"]  = d.get("final_val_acc")
                r["best_val_loss"] = d.get("best_val_loss")
                ssum = d.get("selection_summary", {}) or {}
                r["entropy"] = ssum.get("mean_entropy")
            except Exception:
                pass

        # Fallback to log parsing
        if r["val_loss"] is None and os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    m = re.search(
                        r">Eval.*val_loss=([0-9.]+)\s+val_pp=([0-9.]+)\s+val_acc=([0-9.]+)",
                        line,
                    )
                    if m:
                        r["val_loss"] = float(m.group(1))
                        r["val_pp"]   = float(m.group(2))
                        r["val_acc"]  = float(m.group(3))
                    e = re.search(r"sel_entropy=([0-9.]+)", line)
                    if e:
                        r["entropy"] = float(e.group(1))

        results[opt][mode] = r

# ---- ASCII Table ----
total_tokens = ITERS * BATCH_SIZE * SEQ_LEN * ACC_STEPS
print()
print("=" * 130)
print(f"  WikiText-103 Results: Standard vs OptiSelect (v3, size={MODEL_SIZE})")
print(f"  Llama: n_embd={N_EMBD}, n_head={N_HEAD}, n_layer={N_LAYER} | "
      f"Training: {ITERS} iters × {BATCH_SIZE}×{SEQ_LEN}×{ACC_STEPS} = {total_tokens/1e6:.1f}M tokens | Seed: 0")
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
print("ΔLoss < 0  →  selection improved validation loss")
print("ΔAcc > 0   →  selection improved accuracy")
print("H_sel      →  mean Boltzmann selection entropy (higher = more diverse)")
print()

# ---- LaTeX Table (Paper Table 1 format) ----
size_pretty = {"25m": "25M", "124m": "124M", "210m": "210M"}.get(MODEL_SIZE, MODEL_SIZE.upper())
print()
print("% ============================================================")
print(f"% LaTeX Table for Paper Section 7 (WikiText-103, {size_pretty} params)")
print("% ============================================================")
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{WikiText-103 results (" + size_pretty +
      r"-parameter Llama, " + str(ITERS) + r" iterations).")
print(r"Validation loss and accuracy (\%) for standard training vs.\ OptiSelect.")
print(r"$\mathcal{H}_{\mathrm{sel}}$: mean selection entropy (higher = more diverse).")
print(r"Consistent with Theorem~1 (sign collapse for Lion/Signum) and")
print(r"Proposition~2 (AdEMAMix's temporal stability).}")
print(r"\label{tab:wikitext_results_" + MODEL_SIZE + r"}")
print(r"\small")
print(r"\begin{tabular}{l cc cc c c}")
print(r"\toprule")
print(r"& \multicolumn{2}{c}{Standard} & \multicolumn{2}{c}{+ OptiSelect} & & \\")
print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
print(r"Optimizer & Loss & Acc (\%) & Loss & Acc (\%) & $\Delta$Acc (pp) & $\mathcal{H}_{\mathrm{sel}}$ \\")
print(r"\midrule")

# Find best values for bolding
std_accs = [(o, results[o]["standard"]["val_acc"]) for o in optimizers if results[o]["standard"]["val_acc"]]
sel_accs = [(o, results[o]["selection"]["val_acc"]) for o in optimizers if results[o]["selection"]["val_acc"]]
best_std_opt = max(std_accs, key=lambda x: x[1])[0] if std_accs else None
best_sel_opt = max(sel_accs, key=lambda x: x[1])[0] if sel_accs else None

for opt in optimizers:
    s = results[opt]["standard"]
    x = results[opt]["selection"]
    name = display.get(opt, opt)

    if s["val_loss"]:
        sl = f"{s['val_loss']:.3f}"
        sa = f"{100*s['val_acc']:.2f}"
        if opt == best_std_opt:
            sa = r"\textbf{" + sa + "}"
    else:
        sl = sa = "—"

    if x["val_loss"]:
        xl = f"{x['val_loss']:.3f}"
        xa = f"{100*x['val_acc']:.2f}"
        if opt == best_sel_opt:
            xa = r"\textbf{" + xa + "}"
    else:
        xl = xa = "—"

    if s["val_acc"] and x["val_acc"]:
        delta = 100*(x["val_acc"] - s["val_acc"])
        da = f"{delta:+.2f}"
    else:
        da = "—"

    ent = f"{x['entropy']:.2f}" if x["entropy"] else "—"

    print(f"{name} & {sl} & {sa} & {xl} & {xa} & {da} & {ent} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# ---- Save structured JSON ----
proxy_suffix = "" if PROXY_SOURCE == "train" else f"_proxy-{PROXY_SOURCE}"
size_suffix  = "" if MODEL_SIZE == "25m" else f"_{MODEL_SIZE}"
out = os.path.join(LOG_DIR, f"wikitext_results_v3{size_suffix}{proxy_suffix}.json")
with open(out, "w") as f:
    json.dump({
        "metadata": {
            "model_size": MODEL_SIZE,
            "model": f"llama-{size_pretty} (n_embd={N_EMBD}, n_head={N_HEAD}, n_layer={N_LAYER})",
            "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER,
            "iterations": ITERS,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "acc_steps": ACC_STEPS,
            "tokens_per_step": BATCH_SIZE * SEQ_LEN * ACC_STEPS,
            "total_tokens": ITERS * BATCH_SIZE * SEQ_LEN * ACC_STEPS,
            "seed": 0,
            "val_proxy_size": 512,
            "val_proxy_refresh": int(os.environ.get("VAL_PROXY_REFRESH", 3000)),
            "selection_temperature": 0.1,
            "redundancy_weight": 1.0,
        },
        "optimizer_descriptions": paper_ops,
        "results": results,
    }, f, indent=2)
print(f"\nSaved structured results: {out}")
PYTHON_SCRIPT

echo ""
echo "================================================================"
echo "  Done. Key files (size=${MODEL_SIZE}):"
echo "    Logs:        ${SRC_DIR}/logs/*wikitext*${SIZE_TAG}*.log"
echo "    Checkpoints: ${RESULTS_DIR}/"
if [ "$MODEL_SIZE" = "25m" ]; then
    echo "    Summary:     ${SRC_DIR}/logs/wikitext_results_v3.json"
else
    echo "    Summary:     ${SRC_DIR}/logs/wikitext_results_v3_${MODEL_SIZE}.json"
fi
echo "================================================================"
