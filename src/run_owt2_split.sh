#!/bin/bash
# =============================================================================
# OptiSelect Experiment (OpenWebText2): Publication-Ready Benchmark
# =============================================================================
#
# OpenWebText2 (Skylion007/openwebtext): Reddit-linked web pages filtered
# for quality — the same data distribution used to train GPT-2.
# GPT-2 BPE tokenizer (vocab=50257). ~8B tokens.
#
# This is a strong complement to SlimPajama in the paper:
#   - SlimPajama: multi-domain (CC, C4, GitHub, Wiki, Books, ArXiv, SE)
#   - OpenWebText2: single-domain, high-quality web text
#   - Together they test whether selection benefits are distribution-dependent
#
# ---- Scales ----
#
#   SMALL (~25M params, ~5 hours total for all 20 runs):
#     Architecture: n_embd=384, n_head=6, n_layer=6
#     Batch: 32 × 512 × 1 = 16,384 tokens/step
#     Standard:  12,000 iters → 196M tokens
#     Selection: 12,000 iters × 2× candidates → 393M scored, 196M trained
#
#   FULL (124M params, ~3 days total, ~15h with 5-way parallel split):
#     Architecture: n_embd=768, n_head=12, n_layer=12
#     Batch: 64 × 512 × 4 = 131,072 tokens/step
#     Standard:  38,000 iters → 5.0B tokens (~0.6 epochs)
#     Selection: 19,000 iters × 2× candidates → same compute budget
#
# Usage:
#   bash run_owt2_split.sh [SPLIT_ID] [SCALE] [SEED] [NPROC]
#
# Each optimizer run uses DDP across NPROC GPUs (default 4). The nccl
# backend auto-divides batch_size*acc_steps across ranks, so effective
# batch is preserved — runs are just ~NPROC× faster.
#
# Launch as parallel training jobs from WSL (4 GPUs per run):
#   for i in 1 2 3 4 5; do
#     python csub.py -n owt2-$i -g 4 -t 3d --train \
#       --command "cd /mloscratch/homes/aabdolla/llm-optimizer-benchmark/src && \
#         source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate && \
#         export PYTHONPATH=/mloscratch/homes/aabdolla/GhostSuite:/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src:\$PYTHONPATH && \
#         bash run_owt2_split.sh \$i full 0 4"
#   done
# =============================================================================

SPLIT=${1:-0}       # 0=all sequential, 1-5=parallel splits
SCALE=${2:-small}
SEED=${3:-0}
NPROC=${4:-4}       # GPUs per run (DDP world size)

# ---- Validation-proxy source for OptiSelect ----
# PROXY_SOURCE=train (default, Paper Appendix C) or downstream (mixture
# of HellaSwag/ARC-E/ARC-C/PIQA/SciQ). Downstream requires GPT-2 BPE.
PROXY_SOURCE=${PROXY_SOURCE:-train}
PROXY_TASKS=${PROXY_TASKS:-hellaswag,arc_easy,arc_challenge,piqa,sciq}

# ---- Paths ----
SRC_DIR="/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src"
DATASETS_DIR="/mloscratch/homes/aabdolla/datasets"
if [ "$PROXY_SOURCE" = "train" ]; then
    RESULTS_DIR="/mloscratch/homes/aabdolla/results/owt2_exp_${SCALE}"
else
    RESULTS_DIR="/mloscratch/homes/aabdolla/results/owt2_exp_${SCALE}_proxy_${PROXY_SOURCE}"
fi

cd "$SRC_DIR"
source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate
export PYTHONPATH="/mloscratch/homes/aabdolla/GhostSuite:${SRC_DIR}:$PYTHONPATH"
export HF_HOME=/mloscratch/homes/aabdolla/.hf_cache
export HF_DATASETS_CACHE=/mloscratch/homes/aabdolla/.hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROXY_SOURCE RESULTS_DIR

mkdir -p "$RESULTS_DIR" logs

# ---- Verify data ----
if [ ! -f "${DATASETS_DIR}/openwebtext2/train.bin" ]; then
    echo "[FATAL] OpenWebText2 not found at ${DATASETS_DIR}/openwebtext2/train.bin"
    exit 1
fi

python -c "
import numpy as np, os
p = '${DATASETS_DIR}/openwebtext2/train.bin'
data = np.memmap(p, dtype=np.uint16, mode='r')
print(f'[OK] OpenWebText2 train: {len(data):,} tokens ({os.path.getsize(p)/1e9:.1f} GB)')
p2 = '${DATASETS_DIR}/openwebtext2/val.bin'
data2 = np.memmap(p2, dtype=np.uint16, mode='r')
print(f'[OK] OpenWebText2 val:   {len(data2):,} tokens')
"

# ---- Scale-dependent parameters ----
if [ "$SCALE" == "full" ]; then
    MODEL_ARGS="--model llama --n_embd 768 --n_head 12 --n_layer 12"
    BATCH="--batch_size 64 --sequence_length 512 --acc_steps 4"
    SOPHIA_BATCH="--batch_size 32 --sequence_length 512 --acc_steps 8"
    STD_ITERS=38000     # 38K × 131K = 5.0B tokens
    SEL_ITERS=19000     # 19K × 131K × 2 = 5.0B scored
    STD_WARMUP=760      # 2%
    SEL_WARMUP=380
    EVAL_INTERVAL=1000
    LOG_INTERVAL=100
    SOPHIA_BS=32
    DESCRIPTION="124M params, 5.0B tokens"
else
    MODEL_ARGS="--model llama --n_embd 384 --n_head 6 --n_layer 6"
    BATCH="--batch_size 32 --sequence_length 512 --acc_steps 1"
    SOPHIA_BATCH="--batch_size 16 --sequence_length 512 --acc_steps 2"
    STD_ITERS=12000     # 12K × 16K = 196M tokens
    SEL_ITERS=12000
    STD_WARMUP=240      # 2%
    SEL_WARMUP=240
    EVAL_INTERVAL=400
    LOG_INTERVAL=100
    SOPHIA_BS=16
    DESCRIPTION="25M params, 196M tokens"
fi

# ---- Common args ----
DATA_ARGS="--dataset openwebtext2 --datasets_dir ${DATASETS_DIR}"
COMMON="--scheduler cos --grad_clip 1.0 --weight_decay 0.1 --dropout 0.0 --dtype bfloat16 --device cuda:0 --distributed_backend nccl"
EVAL_ARGS="--eval_interval ${EVAL_INTERVAL} --log_interval ${LOG_INTERVAL}"
RESULTS_ARGS="--results_base_folder ${RESULTS_DIR}"
SEL_ARGS="--selection --candidate_multiplier 2 --selection_temperature 0.1 --selection_sketch_dim 1024 --val_proxy_refresh 5000"
SEL_ARGS="${SEL_ARGS} --val_proxy_source ${PROXY_SOURCE} --val_proxy_tasks ${PROXY_TASKS}"
# Chunk the candidate forward/backward: the full-scale 128-candidate CE
# softmax would materialize ~13 GB fp32 logits. Processing in
# candidate_multiplier chunks of batch_size halves it. For full scale we
# set this explicitly to batch_size / 2 for extra headroom.
if [ "$SCALE" == "full" ]; then
    SEL_ARGS="${SEL_ARGS} --candidate_chunk_size 32"
fi

FAILED_RUNS=()
COMPLETED_RUNS=()


# ======================================================================
#  Runner function
# ======================================================================
run_experiment() {
    local OPT_NAME=$1
    local MODE=$2
    local OPT_FLAG=$3
    local OPT_EXTRA=$4
    local BATCH_OVR=${5:-$BATCH}

    local EXP_NAME
    if [ "$PROXY_SOURCE" = "train" ]; then
        EXP_NAME="${MODE}_owt2_${SCALE}_${OPT_NAME}_seed${SEED}"
    else
        EXP_NAME="${MODE}_owt2_${SCALE}_proxy-${PROXY_SOURCE}_${OPT_NAME}_seed${SEED}"
    fi
    local LOG_FILE="logs/${EXP_NAME}.log"

    local ITERS WARMUP
    if [ "$MODE" == "standard" ]; then
        ITERS=$STD_ITERS; WARMUP=$STD_WARMUP
    else
        ITERS=$SEL_ITERS; WARMUP=$SEL_WARMUP
    fi

    if [ -f "${RESULTS_DIR}/${EXP_NAME}/summary.json" ]; then
        echo "[SKIP] ${EXP_NAME}"
        COMPLETED_RUNS+=("$EXP_NAME")
        return 0
    fi

    rm -f "$LOG_FILE"

    echo ""
    echo "============================================================"
    echo "  ${OPT_NAME} | ${MODE} | owt2-${SCALE} | seed=${SEED} | GPUs=${NPROC}"
    echo "  Iters: ${ITERS} | Warmup: ${WARMUP}"
    [ "$MODE" == "selection" ] && echo "  Selection: τ=0.1, 2× candidates"
    echo "  Started: $(date)"
    echo "============================================================"

    local SEL_FLAG=""
    [ "$MODE" == "selection" ] && SEL_FLAG="$SEL_ARGS"

    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} main.py \
        $MODEL_ARGS \
        $DATA_ARGS \
        $BATCH_OVR \
        --iterations $ITERS \
        --warmup_steps $WARMUP \
        $COMMON \
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
        COMPLETED_RUNS+=("$EXP_NAME")
        echo ">>> [OK] ${EXP_NAME} at $(date)"
    else
        FAILED_RUNS+=("$EXP_NAME")
        echo ">>> [FAIL] ${EXP_NAME} exit code ${EXIT_CODE}"
    fi
    return 0
}


# ======================================================================
#  Optimizer definitions (Appendix B)
# ======================================================================
run_adamw()    { run_experiment "adamw"    "$1" "--opt adamw"    "--lr 1e-3 --beta1 0.9 --beta2 0.999"; }
run_ademamix() { run_experiment "ademamix" "$1" "--opt ademamix" "--lr 1e-3 --beta1 0.9 --beta2 0.999 --adema_beta3 0.9999 --adema_alpha 0.8"; }
run_dmuon()    { run_experiment "d-muon"   "$1" "--opt d-muon"   "--lr 1e-3 --beta1 0.9 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5"; }
run_mars()     { run_experiment "mars"     "$1" "--opt mars"     "--lr 1e-3 --mars_lr 3e-3 --beta1 0.9 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99"; }
run_sophiag()  { run_experiment "sophiag"  "$1" "--opt sophiag"  "--lr 1e-3 --beta1 0.9 --beta2 0.999 --sophia_rho 0.04 --precondition_frequency 10 --sophia_bs ${SOPHIA_BS}" "$SOPHIA_BATCH"; }
run_soap()     { run_experiment "soap"     "$1" "--opt soap"     "--lr 1e-3 --beta1 0.9 --beta2 0.95 --precondition_frequency 10"; }
run_lion()     { run_experiment "lion"     "$1" "--opt lion"     "--lr 3e-4 --beta1 0.9 --beta2 0.99"; }
run_signum()   { run_experiment "signum"   "$1" "--opt signum"   "--lr 3e-4 --momentum 0.9"; }
run_adopt()    { run_experiment "adopt"    "$1" "--opt adopt"    "--lr 1e-3 --beta1 0.9 --beta2 0.999"; }
run_sgd()      { run_experiment "sgd"      "$1" "--opt sgd"      "--lr 3e-2 --momentum 0.9"; }


# ======================================================================
#  Dispatch
# ======================================================================
echo ""
echo "================================================================"
echo "  OptiSelect OpenWebText2 | Split: ${SPLIT} | Scale: ${SCALE}"
echo "  ${DESCRIPTION} | Seed: ${SEED} | DDP GPUs: ${NPROC}"
echo "  Proxy source: ${PROXY_SOURCE}$([ "$PROXY_SOURCE" != "train" ] && echo " [${PROXY_TASKS}]")"
echo "  Results dir:  ${RESULTS_DIR}"
echo "  Started: $(date)"
echo "================================================================"
echo ""

#  Selection runs before standard so OOMs / selection-specific failures
#  surface early instead of after the standard run has already consumed
#  the wall-clock budget.
case $SPLIT in
    0)  # All sequential
        for opt in adamw ademamix dmuon mars sophiag soap lion signum adopt sgd; do
            run_${opt} "selection"
            run_${opt} "standard"
        done
        ;;
    1) run_adamw "selection"; run_adamw "standard"; run_ademamix "selection"; run_ademamix "standard" ;;
    2) run_dmuon "selection"; run_dmuon "standard"; run_mars "selection"; run_mars "standard" ;;
    3) run_sophiag "selection"; run_sophiag "standard"; run_soap "selection"; run_soap "standard" ;;
    4) run_lion "selection"; run_lion "standard"; run_signum "selection"; run_signum "standard" ;;
    5) run_adopt "selection"; run_adopt "standard"; run_sgd "selection"; run_sgd "standard" ;;
    *)  echo "Unknown split: $SPLIT (use 0=all, 1-5=parallel)"; exit 1 ;;
esac


# ======================================================================
#  Results Collection
# ======================================================================
echo ""
echo "================================================================"
echo "  Split ${SPLIT} | Completed: ${#COMPLETED_RUNS[@]} | Failed: ${#FAILED_RUNS[@]}"
echo "  Finished: $(date)"
echo "================================================================"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "  Failed:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "    - $run"
        tail -3 "logs/${run}.log" 2>/dev/null | sed 's/^/      /'
    done
fi

# Only collect full table when all splits are done (split 0) or at end of any split
python - << 'PYTHON_SCRIPT'
import os, re, json

LOG_DIR = "logs"
PROXY_SOURCE = os.environ.get("PROXY_SOURCE", "train")
NAME_INFIX = "" if PROXY_SOURCE == "train" else f"proxy-{PROXY_SOURCE}_"
optimizers = ["adamw", "ademamix", "d-muon", "mars", "sophiag", "soap",
              "lion", "signum", "adopt", "sgd"]
display = {"adamw": "AdamW", "ademamix": "AdEMAMix", "d-muon": "D-Muon",
           "mars": "MARS", "sophiag": "Sophia", "soap": "SOAP",
           "lion": "Lion", "signum": "Signum", "adopt": "ADOPT", "sgd": "SGD"}
modes = ["standard", "selection"]

# Auto-detect scale and seed from logs
scale_tag = seed = None
for f in os.listdir(LOG_DIR):
    if "owt2" in f:
        if "small" in f: scale_tag = "small"
        elif "full" in f: scale_tag = "full"
        m = re.search(r"seed(\d+)", f)
        if m: seed = int(m.group(1))
        break
if not scale_tag: scale_tag = "small"
if seed is None: seed = 0

results = {}
n_found = 0
for opt in optimizers:
    results[opt] = {}
    for mode in modes:
        log_path = os.path.join(LOG_DIR, f"{mode}_owt2_{scale_tag}_{NAME_INFIX}{opt}_seed{seed}.log")
        r = {"val_loss": None, "val_pp": None, "val_acc": None, "entropy": None}
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    m = re.search(r">Eval.*val_loss=([0-9.]+)\s+val_pp=([0-9.]+)\s+val_acc=([0-9.]+)", line)
                    if m:
                        r["val_loss"] = float(m.group(1))
                        r["val_pp"]   = float(m.group(2))
                        r["val_acc"]  = float(m.group(3))
                        n_found += 1
                    e = re.search(r"sel_entropy=([0-9.]+)", line)
                    if e:
                        r["entropy"] = float(e.group(1))
        results[opt][mode] = r

if n_found == 0:
    print("No results found yet (other splits may still be running)")
    exit(0)

# ---- ASCII Table ----
desc = "25M params" if scale_tag == "small" else "124M params"
print()
print("=" * 130)
print(f"  OpenWebText2 Results: Standard vs OptiSelect ({desc}, seed {seed})")
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
print("ΔLoss < 0 → selection improved | ΔAcc > 0 → selection improved")

# ---- LaTeX Table ----
print()
print("% ====== LaTeX Table for Paper ======")
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{OpenWebText2 results: validation loss and accuracy (\%) for standard")
print(r"training vs.\ OptiSelect. " + f"Model: {desc}" + r" Llama.}")
print(r"\label{tab:owt2_results}")
print(r"\small")
print(r"\begin{tabular}{l cc cc c c}")
print(r"\toprule")
print(r"& \multicolumn{2}{c}{Standard} & \multicolumn{2}{c}{+ OptiSelect} & & \\")
print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
print(r"Optimizer & Loss & Acc (\%) & Loss & Acc (\%) & $\Delta$Acc & $\mathcal{H}_{\text{sel}}$ \\")
print(r"\midrule")

best_std = max((results[o]["standard"]["val_acc"] or 0) for o in optimizers)
best_sel = max((results[o]["selection"]["val_acc"] or 0) for o in optimizers)

for opt in optimizers:
    s = results[opt]["standard"]
    x = results[opt]["selection"]
    name = display.get(opt, opt)
    if s["val_loss"]:
        sl = f"{s['val_loss']:.3f}"
        sa = f"{100*s['val_acc']:.2f}"
        if s["val_acc"] and abs(s["val_acc"] - best_std) < 1e-6:
            sa = r"\textbf{" + sa + "}"
    else: sl = sa = "—"
    if x["val_loss"]:
        xl = f"{x['val_loss']:.3f}"
        xa = f"{100*x['val_acc']:.2f}"
        if x["val_acc"] and abs(x["val_acc"] - best_sel) < 1e-6:
            xa = r"\textbf{" + xa + "}"
    else: xl = xa = "—"
    da = f"{100*(x['val_acc']-s['val_acc']):+.2f}" if s["val_acc"] and x["val_acc"] else "—"
    ent = f"{x['entropy']:.2f}" if x["entropy"] else "—"
    print(f"{name} & {sl} & {sa} & {xl} & {xa} & {da} & {ent} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

_psuffix = "" if PROXY_SOURCE == "train" else f"_proxy-{PROXY_SOURCE}"
out = os.path.join(LOG_DIR, f"owt2_{scale_tag}_results_seed{seed}{_psuffix}.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
PYTHON_SCRIPT