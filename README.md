# OptiSelect

**Online, optimizer-aware in-batch data selection for LLM pretraining.**

OptiSelect is a data-selection method that, at every training step, scores a
larger *candidate* batch and picks the most useful subset to actually train on.
Unlike selection methods that ignore the optimizer, OptiSelect's score depends
on the **optimizer-specific update geometry** (AdamW, AdEMAMix, Sophia, Lion,
Muon, SOAP, GaLore, C-AdamW, …), so the selected batch is tuned to the update
the optimizer will actually take.

The pipeline per step is:

1. Draw a candidate batch $\tilde{B}$ of size $kB$ ($k$ = `candidate_multiplier`).
2. Forward/backward once on the candidates, capturing per-sample gradient
   factors $(a(z), b(z))$ with the **Ghost** trick (no full per-sample
   gradients materialized).
3. Compute a validation-proxy gradient $g_V$ (averaged over a fixed proxy set,
   refreshed every `val_proxy_refresh` steps).
4. Score each candidate $z$ using the optimizer's frozen-state operator
   $\mathcal{O}_t$:
   $$U(z,t) \approx \eta\,\langle \mathcal{O}_t(g_z), g_V\rangle - \eta^2\,\lambda_r\,\langle \mathcal{O}_t(g_z), G_t\rangle$$
5. Greedy Boltzmann selection over scores with redundancy penalty.
6. Standard optimizer step on the selected $B$ samples only.

---

## Acknowledgements

This repository stands on two open-source projects and would not exist
without them:

- **[llm-optimizer-benchmark](https://github.com/epfml/llm-optimizer-benchmark)**
  (EPFL MLO). We fork their training / evaluation framework and all their
  optimizer implementations (AdamW, AdEMAMix, Sophia, Lion, Muon, SOAP,
  Prodigy, ADOPT, Signum, SF-AdamW, MARS, D-Muon, …). The `src/` tree —
  except `src/selection/` and the `run_*.sh` scripts — is substantially
  unchanged from upstream, with only the additions needed to wire OptiSelect
  into the training loop. See [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md) for
  details.
- **[GhostSuite](https://github.com/Jiachen-T-Wang/GhostSuite)** (Jiachen T.
  Wang). Vendored at `GhostSuite/`. We use its ghost-gradient engines to
  capture per-sample gradient factors without materializing full per-sample
  gradients. The folder retains its upstream `README.md` and `LICENSE`.

OptiSelect-specific contributions live in:
- `src/selection/` — scoring, engine, and training loop.
- Additions to `src/config/base.py` and `src/main.py` that expose the
  `--selection` flag set.
- `src/run_*.sh` experiment scripts.
- `src/collect_results.py`.

---

## Repository layout

```
.
├── README.md                   # this file
├── LICENSE                     # Apache 2.0 (inherited from upstream)
├── requirements.txt            # benchmark deps (see GhostSuite/requirements.txt for ghost deps)
├── assets/                     # figures from upstream
├── scripts/                    # upstream benchmark launch scripts (124m/210m/720m/moe-520m)
├── GhostSuite/                 # vendored copy of GhostSuite (unmodified)
└── src/
    ├── main.py                 # upstream training entry point + OptiSelect hook
    ├── collect_results.py      # helper to aggregate selection runs
    ├── run_shakespeare_exp.sh  # selection experiments on Shakespeare-char
    ├── run_wikitext_exp.sh     # selection experiments on WikiText
    ├── run_owt2_split.sh       # selection experiments on OpenWebText2
    ├── run_slimpajama_split.sh # selection experiments on SlimPajama-6B
    ├── config/                 # upstream config (extended with --selection_* args)
    ├── data/                   # upstream dataloaders (shakespeare, owt2, slimpajama, wikitext, fineweb)
    ├── distributed/            # upstream distributed utils
    ├── logger/                 # upstream logger
    ├── models/                 # upstream model zoo (GPT-2, Llama, MoE)
    ├── optim/                  # upstream optimizer zoo
    └── selection/              # ★ OptiSelect
        ├── influence_scoring.py   # per-optimizer operators O_t
        ├── optiselect_engine.py   # Ghost-factor capture + scoring + sampling
        ├── train_with_selection.py# training loop with selection
        └── downstream_proxy.py    # task-mixture validation proxy (HellaSwag/ARC/PIQA/SciQ)
```

---

## Quickstart

### 1. Environment

```bash
conda create -n optiselect python=3.10 -y
conda activate optiselect
pip install -r requirements.txt
pip install -r GhostSuite/requirements.txt
```

### 2. Data

Tokenized datasets are **not** checked in. The dataloaders under
`src/data/` will download and tokenize on first use. Point the dataset
cache somewhere with enough disk:

```bash
export HF_DATASETS_CACHE=/path/to/datasets
```

Supported datasets out of the box: `shakespeare-char`, `wikitext`,
`openwebtext2`, `slimpajama6B`, `fineweb-100BT` — see `src/data/utils.py`.

### 3. Baseline (no selection)

```bash
python src/main.py --config_format base --model llama \
    --dataset slimpajama6B --opt adamw
```

### 4. Training with OptiSelect

Enable selection with `--selection` and tune the selection-specific flags:

```bash
python src/main.py --config_format base --model llama \
    --dataset slimpajama6B --opt adamw \
    --selection \
    --candidate_multiplier 2 \
    --selection_temperature 0.1 \
    --selection_redundancy_weight 1.0 \
    --val_proxy_size 4096 \
    --val_proxy_refresh 5000
```

OptiSelect flags (added to `src/config/base.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--selection` | off | Enable OptiSelect |
| `--candidate_multiplier` | `2` | Candidate batch size = $kB$ |
| `--selection_temperature` | `0.1` | Boltzmann temperature $\tau$ |
| `--selection_redundancy_weight` | `1.0` | Redundancy weight $\lambda_r$ |
| `--selection_sketch_dim` | `1024` | CountSketch dim (exact scoring at 124M scale) |
| `--val_proxy_size` | `4096` | Fixed validation proxy size |
| `--val_proxy_refresh` | `5000` | Proxy resample interval (steps) |
| `--val_proxy_source` | `train` | Proxy distribution: `train` (default, paper behaviour) or `downstream` (task mixture — see below) |
| `--val_proxy_tasks` | `hellaswag,arc_easy,arc_challenge,piqa,sciq` | Tasks used when `--val_proxy_source=downstream` |
| `--selection_geometry_override` | `None` | Force scoring geometry (`sgd`/`adamw`/`sophia`/`lion`/`muon`) |

### 4.1 Validation-proxy source (`train` vs `downstream`)

The score $U(z,t)$ measures how much a candidate's update reduces loss on a
validation proxy set. By default the proxy is drawn from the **training
dataset's** val split (paper behaviour). You can instead draw it from a
mixture of held-out multiple-choice benchmarks — *HellaSwag, ARC-Easy,
ARC-Challenge, PIQA, SciQ* — turning $U(z,t)$ into a **task-aware**
alignment signal: "which training sample most improves next-token prediction
on the correct completions of downstream tasks".

Switch modes with `--val_proxy_source`:

```bash
# Paper-default proxy (same distribution as training)
python src/main.py ... --selection --val_proxy_source train

# Downstream task-mixture proxy
python src/main.py ... --selection --val_proxy_source downstream

# Restrict to a subset of tasks
python src/main.py ... --selection --val_proxy_source downstream \
    --val_proxy_tasks hellaswag,arc_easy,arc_challenge
```

Implementation: [`src/selection/downstream_proxy.py`](src/selection/downstream_proxy.py)
reuses the GPT-2 BPE tokenizers in
[`src/data/benchmarks.py`](src/data/benchmarks.py). Each task's val split is
tokenized once into a cached `.bin` file (~5–10 min on the very first run,
rank 0 builds while other DDP ranks wait on a barrier); subsequent runs
reuse the cache.

**Caveats**

- **Requires GPT-2 BPE (vocab ≥ 50257).** The downstream `.bin` files use
  GPT-2 tokenization, so the model's vocabulary must cover it. Character-
  level runs (e.g., Shakespeare-Char, vocab=95) are rejected with a clear
  `ValueError` at proxy-build time.
- **PIQA and SciQ may fail to load** depending on your `datasets` version.
  As of `datasets ≥ 3.0`, script-based HF datasets (including `piqa` and
  `sciq`) are rejected with
  `RuntimeError: Dataset scripts are no longer supported`. The proxy handles
  this gracefully: broken tasks are skipped with a warning, the doc budget
  is rebalanced over the surviving tasks, and the run continues. If all
  tasks fail, a clear error is raised. To suppress the warnings, pin
  `--val_proxy_tasks hellaswag,arc_easy,arc_challenge` explicitly.

### 5. End-to-end experiment scripts

The scripts under `src/run_*.sh` drive full selection-vs-baseline sweeps with
seed loops and logging. Open one and adjust paths / wandb entity / GPU count
before running:

```bash
bash src/run_shakespeare_exp.sh     # quick sanity check
bash src/run_wikitext_exp.sh
bash src/run_owt2_split.sh
bash src/run_slimpajama_split.sh
```

The WikiText and Shakespeare scripts also expose the downstream-proxy toggle
via environment variables (passed through to `--val_proxy_source` /
`--val_proxy_tasks`):

```bash
# WikiText with downstream-task proxy (results in wikitext_exp_v3_proxy_downstream/)
PROXY_SOURCE=downstream bash src/run_wikitext_exp.sh 4

# Restrict to reliable tasks (avoids PIQA/SciQ HF-script deprecation)
PROXY_SOURCE=downstream PROXY_TASKS=hellaswag,arc_easy,arc_challenge \
    bash src/run_wikitext_exp.sh 4
```

When `PROXY_SOURCE != train`, the scripts automatically suffix `RESULTS_DIR`
and experiment names with `_proxy_<source>` so downstream-proxy runs don't
collide with existing train-proxy results.

Results are written under `src/logs/` and `src/exps/` (both gitignored).

---

## DCLM-Edu demo benchmark

The DCLM-Edu benchmark entry points are:

- [`src/demo.ipynb`](src/demo.ipynb) — interactive, single-process notebook for
  developing and inspecting runs.
- [`src/demo_config.yaml`](src/demo_config.yaml) — canonical benchmark
  configuration: dataset shards, model size, token budgets, optimizer
  hyperparameters, OptiSelect settings, and evaluation settings.
- [`src/demo_DCLM.sh`](src/demo_DCLM.sh) — non-interactive launcher around
  [`src/demo_dclm_ddp.py`](src/demo_dclm_ddp.py), intended for reproducible
  local, DDP, and scheduler-managed runs.

Although these files are named "demo", they are configured for rigorous
optimizer and OptiSelect comparisons: fixed update-token budgets, fixed
validation budgets, explicit optimizer-specific learning rates, and result
JSONs with token accounting.

### Running `demo.ipynb`

Use the notebook when you want an inspectable single-GPU workflow, quick
configuration iteration, plots during training, or final
`lm-evaluation-harness` evaluation from the trained notebook model.

1. Start Jupyter from the repository or source directory.
2. Run the dependency/setup cells once in a fresh environment.
3. Run the config and derived-settings cells. The notebook loads
   `src/demo_config.yaml` into `CFG`.
4. Optionally override `CFG` fields in the sweep cell before launching.
5. Run the sweep cell. Each run writes a per-run `summary.json`; the sweep
   cell also writes one combined summary JSON under `CFG['paths']['results_dir']`.

Example single-run downstream-proxy sweep:

```python
ALL_RESULTS = {}
ALL_HISTORIES = {}
LAST_MODEL = None

CFG['experiment']['run_keys'] = ['optiselect_signsgd']

CFG['selection']['proxy_source'] = 'downstream'
CFG['selection']['proxy_tasks'] = [
    'hellaswag',
    'arc_easy',
    'arc_challenge',
    'openbookqa',
]
CFG['selection']['candidate_multiplier'] = 2
CFG['selection']['use_countsketch'] = True
CFG['selection']['sketch_dim'] = 16384
CFG['selection']['countsketch_token_block'] = 128
CFG['selection']['val_proxy_size'] = 8192
CFG['selection']['proxy_batch_size'] = 32
CFG['selection']['candidate_chunk_size'] = 16

CFG['paths']['results_dir'] = '/mloscratch/homes/aabdolla/results/demo_signsgd_downstream_proxy_cm2'
Path(CFG['paths']['results_dir']).expanduser().mkdir(parents=True, exist_ok=True)

validate_selection_proxy(CFG)

for run_key in CFG['experiment']['run_keys']:
    result, LAST_MODEL = run_experiment(run_key, CFG, live_histories=ALL_HISTORIES)
    ALL_RESULTS[run_key] = result
    ALL_HISTORIES[run_key] = result['history']
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
```

Valid run keys are defined in `demo_config.yaml` under `runs:`. Standard and
OptiSelect variants use the same optimizer hyperparameter block:

```text
adamw, optiselect_adamw
d-muon, optiselect_d-muon
sgd, optiselect_sgd
signsgd, optiselect_signsgd
lion, optiselect_lion
signum, optiselect_signum
ademamix, optiselect_ademamix
sophia, optiselect_sophia
```

For sign-based momentum experiments, use `signum`. The `signsgd` run is the
zero-momentum signSGD baseline; Nesterov momentum requires `momentum > 0` and
therefore belongs in the `signum` configuration.

### Running `demo_DCLM.sh`

Use `demo_DCLM.sh` for unattended runs. It reads `demo_config.yaml`, applies
environment-variable overrides, validates the requested run keys, and launches
`demo_dclm_ddp.py` with `torchrun`.

Single-GPU run:

```bash
cd /mloscratch/homes/aabdolla/llm-optimizer-benchmark/src
source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate
DEMO_NPROC=1 \
DEMO_RUN_KEYS=optiselect_signsgd \
bash demo_DCLM.sh
```

Multi-GPU run on one node:

```bash
cd /mloscratch/homes/aabdolla/llm-optimizer-benchmark/src
source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate
DEMO_NPROC=4 \
DEMO_RUN_KEYS=optiselect_adamw,optiselect_lion \
DEMO_RESULTS_DIR=/mloscratch/homes/aabdolla/results/demo_dclm_adamw_lion_ddp4_seed42 \
bash demo_DCLM.sh
```

The launcher keeps the benchmark's batch semantics global: `DEMO_NPROC`
controls `torchrun --nproc_per_node`, while `training.device_batch_size` and
`training.total_batch_tokens` remain global settings from the YAML unless
explicitly overridden.

Useful launcher variables:

| Variable | Purpose |
|---|---|
| `DEMO_NPROC` | Number of local GPU processes. |
| `DEMO_CONFIG` | Path to `demo_config.yaml`. |
| `DEMO_RUN_KEYS` | Comma-separated run keys from the YAML. |
| `DEMO_SEED` | Training/data seed. |
| `DEMO_STANDARD_UPDATE_TOKENS` | Standard-run update-token budget. |
| `DEMO_OPTISELECT_UPDATE_TOKENS` | OptiSelect selected/update-token budget. |
| `DEMO_PROXY_SOURCE` | `dclm_edu_heldout` or `downstream`. |
| `DEMO_PROXY_TASKS` | Comma-separated downstream proxy tasks. |
| `DEMO_CANDIDATE_MULTIPLIER` | Candidate batch multiplier for OptiSelect. |
| `DEMO_USE_COUNTSKETCH` | `1` for CountSketch scoring, `0` for exact Ghost factors. |
| `DEMO_SKETCH_DIM` | CountSketch feature dimension. |
| `DEMO_VAL_PROXY_SIZE` | Number of proxy examples/documents. |
| `DEMO_PROXY_BATCH_SIZE` | Proxy factor batch size. |
| `DEMO_CANDIDATE_CHUNK_SIZE` | Candidate-factor chunk size. |
| `DEMO_RESULTS_DIR` | Output directory for per-run summaries and sweep summary. |
| `DEMO_SUMMARY_NAME` | Combined sweep summary filename. |
| `DEMO_SMOKE_STEPS` | Short-run override for launcher validation. |
| `DEMO_DRY_RUN` | Set to `1` to print the generated `torchrun` command. |

Recommended smoke test before a long allocation:

```bash
DEMO_NPROC=1 \
DEMO_RUN_KEYS=optiselect_signsgd \
DEMO_SMOKE_STEPS=5 \
DEMO_EVAL_INTERVAL=5 \
DEMO_EVAL_TOKENS=1048576 \
DEMO_FINAL_EVAL_TOKENS=1048576 \
DEMO_DRY_RUN=1 \
bash demo_DCLM.sh
```

Remove `DEMO_DRY_RUN=1` to actually launch the smoke test.

### Run:ai / `csub.py` example

For remote H100 jobs managed through Run:ai, pass the launcher variables inside
the submitted command. This example runs the 1B-token downstream-proxy
OptiSelect signSGD benchmark on two H100 GPUs:

```bash
python csub.py -n demo-dclm-signsgd-equal1b-downproxy-ddp2 -g 2 -t 1d \
  --train --large-shm --node-type h100 \
  --command "cd /mloscratch/homes/aabdolla/llm-optimizer-benchmark/src && \
    source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate && \
    DEMO_NPROC=2 \
    DEMO_CONFIG=/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src/demo_config.yaml \
    DEMO_RUN_KEYS=optiselect_signsgd \
    DEMO_SEED=42 \
    DEMO_STANDARD_UPDATE_TOKENS=1000000000 \
    DEMO_OPTISELECT_UPDATE_TOKENS=1000000000 \
    DEMO_PROXY_SOURCE=downstream \
    DEMO_PROXY_TASKS=hellaswag,arc_easy,arc_challenge,openbookqa \
    DEMO_CANDIDATE_MULTIPLIER=2 \
    DEMO_USE_COUNTSKETCH=1 \
    DEMO_SKETCH_DIM=16384 \
    DEMO_COUNTSKETCH_ROW_BLOCK=32 \
    DEMO_COUNTSKETCH_TOKEN_BLOCK=128 \
    DEMO_VAL_PROXY_SIZE=8192 \
    DEMO_PROXY_BATCH_SIZE=32 \
    DEMO_CANDIDATE_CHUNK_SIZE=16 \
    DEMO_RESULTS_DIR=/mloscratch/homes/aabdolla/results/demo_dclm_signsgd_equal1B_downstreamproxy_hs_arc_obqa_ddp2_h100_seed42_new \
    DEMO_SUMMARY_NAME=demo_dclm_ddp_summary.json \
    bash demo_DCLM.sh"
```

For Signum with Nesterov momentum, use the `signum` optimizer rather than
`signsgd`. The `signsgd` benchmark is the zero-momentum sign baseline, while
Nesterov requires `momentum > 0`. This command creates a derived YAML config
inside the results tree, enables Nesterov for Signum, and then launches the
OptiSelect Signum run:

```bash
python csub.py -n demo-dclm-signum-nesterov-equal1b-downproxy-ddp2 -g 2 -t 1d \
  --train --large-shm --node-type h100 \
  --command "cd /mloscratch/homes/aabdolla/llm-optimizer-benchmark/src && \
    source /mloscratch/homes/aabdolla/optiselect/.venv/bin/activate && \
    python -c 'from pathlib import Path; import yaml; src=Path(\"/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src/demo_config.yaml\"); dst=Path(\"/mloscratch/homes/aabdolla/results/configs/demo_config_signum_nesterov_seed42.yaml\"); dst.parent.mkdir(parents=True, exist_ok=True); cfg=yaml.safe_load(src.read_text()); cfg[\"optimizers\"][\"signum\"][\"momentum\"]=0.9; cfg[\"optimizers\"][\"signum\"][\"dampening\"]=0.0; cfg[\"optimizers\"][\"signum\"][\"nesterov\"]=True; dst.write_text(yaml.safe_dump(cfg, sort_keys=False))' && \
    DEMO_NPROC=2 \
    DEMO_CONFIG=/mloscratch/homes/aabdolla/results/configs/demo_config_signum_nesterov_seed42.yaml \
    DEMO_RUN_KEYS=optiselect_signum \
    DEMO_SEED=42 \
    DEMO_STANDARD_UPDATE_TOKENS=1000000000 \
    DEMO_OPTISELECT_UPDATE_TOKENS=1000000000 \
    DEMO_PROXY_SOURCE=downstream \
    DEMO_PROXY_TASKS=hellaswag,arc_easy,arc_challenge,openbookqa \
    DEMO_CANDIDATE_MULTIPLIER=2 \
    DEMO_USE_COUNTSKETCH=1 \
    DEMO_SKETCH_DIM=16384 \
    DEMO_COUNTSKETCH_ROW_BLOCK=32 \
    DEMO_COUNTSKETCH_TOKEN_BLOCK=128 \
    DEMO_VAL_PROXY_SIZE=8192 \
    DEMO_PROXY_BATCH_SIZE=32 \
    DEMO_CANDIDATE_CHUNK_SIZE=16 \
    DEMO_RESULTS_DIR=/mloscratch/homes/aabdolla/results/demo_dclm_signum_nesterov_equal1B_downstreamproxy_hs_arc_obqa_ddp2_h100_seed42_new \
    DEMO_SUMMARY_NAME=demo_dclm_ddp_summary.json \
    bash demo_DCLM.sh"
```

The inline `python -c` command is a reproducibility convenience. It avoids
editing the tracked `src/demo_config.yaml` in a scheduler job and writes the
exact derived config used by the run:

```python
from pathlib import Path
import yaml

src = Path('/mloscratch/homes/aabdolla/llm-optimizer-benchmark/src/demo_config.yaml')
dst = Path('/mloscratch/homes/aabdolla/results/configs/demo_config_signum_nesterov_seed42.yaml')
dst.parent.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(src.read_text())
cfg['optimizers']['signum']['momentum'] = 0.9
cfg['optimizers']['signum']['dampening'] = 0.0
cfg['optimizers']['signum']['nesterov'] = True
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
```

The launcher then points `DEMO_CONFIG` at this derived YAML and uses
`DEMO_RUN_KEYS=optiselect_signum`. Keep the generated config path, job name,
and `DEMO_RESULTS_DIR` aligned with the optimizer and seed so later result
aggregation remains unambiguous.

Change `DEMO_RUN_KEYS` and `DEMO_RESULTS_DIR` together for each new optimizer
or ablation so outputs do not collide. For example, use
`DEMO_RUN_KEYS=optiselect_lion` with a results directory containing `lion` for
a Lion run, or `DEMO_RUN_KEYS=signum,optiselect_signum` to run both Signum
variants.

---

## Supported optimizers for selection

The scoring operator $\mathcal{O}_t$ is implemented for the following
optimizer families in [`src/selection/influence_scoring.py`](src/selection/influence_scoring.py):

| Family | Function | Notes |
|---|---|---|
| AdamW / AdEMAMix | `compute_adam_family_scores` | Eq. (7), (8) |
| Sophia | `compute_sophia_scores` | Eq. (11), with clip |
| Lion | `compute_lion_scores` | Eq. (13) |
| Muon | `compute_muon_scores` | Eq. (15), right-preconditioner |
| SOAP | `compute_soap_scores` | Eq. (18), rotated operator |
| GaLore | `compute_galore_scores` | Eq. (22), projected |
| C-AdamW | `compute_cadamw_scores` | Eq. (24), masked |

Equation numbers refer to the OptiSelect paper (in preparation).

---

## License

Apache 2.0 — see [LICENSE](LICENSE). Inherited from
[epfml/llm-optimizer-benchmark](https://github.com/epfml/llm-optimizer-benchmark).
The vendored `GhostSuite/` directory carries its own Apache 2.0 license; see
[GhostSuite/LICENSE](GhostSuite/LICENSE).

## Citation

If you use this code, please cite OptiSelect **and** the two projects it
builds on:

```bibtex
@misc{optiselect2026,
  title  = {OptiSelect: Optimizer-Aware Online Data Selection for LLM Pretraining},
  author = {Abdollahpour, Alireza and co-authors},
  year   = {2026},
  note   = {In preparation}
}

@article{semenov2025benchmarking,
  title   = {Benchmarking Optimizers for Large Language Model Pretraining},
  author  = {Semenov, Andrei and Pagliardini, Matteo and Jaggi, Martin},
  journal = {arXiv:2509.01440},
  year    = {2025}
}

@misc{wang2024ghostsuite,
  title  = {GhostSuite: Efficient Per-Sample Gradient Computations},
  author = {Wang, Jiachen T.},
  year   = {2024},
  url    = {https://github.com/Jiachen-T-Wang/GhostSuite}
}
```
