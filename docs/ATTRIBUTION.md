# Attribution

OptiSelect is built on top of two open-source projects. This document makes
explicit what code is upstream, what is modified, and what is new.

## Upstream: `epfml/llm-optimizer-benchmark`

Source: https://github.com/epfml/llm-optimizer-benchmark
License: Apache 2.0
Reference: Semenov, Pagliardini, Jaggi — *Benchmarking Optimizers for Large
Language Model Pretraining*, arXiv:2509.01440, 2025.

**Vendored unchanged** (except for `__pycache__` / compiled files):
- `LICENSE`
- `assets/`
- `scripts/` (124m, 210m, 720m, moe-520m launch scripts)
- `src/data/` — dataloaders for Shakespeare-char, WikiText, OpenWebText2,
  SlimPajama-6B, FineWeb-100BT
- `src/distributed/`
- `src/logger/`
- `src/models/` — GPT-2, Llama, MoE
- `src/optim/` — AdamW, AdEMAMix, ADOPT, D-Muon, Lion, MARS, Muon, Prodigy,
  SF-AdamW, Signum, SOAP, Sophia
- `src/config/` — **extended**, see below

**Modified**:
- `src/config/base.py` — added `--selection*` flags (the `OptiSelect:
  online in-batch data selection` block)
- `src/main.py` — added the branch that dispatches to
  `train_with_selection` when `--selection` is set
- `src/optim/utils.py` — minor logging additions

## Vendored: `Jiachen-T-Wang/GhostSuite`

Source: https://github.com/Jiachen-T-Wang/GhostSuite
License: Apache 2.0

The entire `GhostSuite/` directory is a vendored copy, unmodified, minus its
`.git`. It retains its own `LICENSE` and `README.md`. We use its
ghost-gradient engines from `ghostEngines/` to capture per-sample gradient
factors without materializing full per-sample gradients.

## New in OptiSelect

- `src/selection/` — the full selection pipeline
  - `influence_scoring.py` — per-optimizer frozen-state operators $\mathcal{O}_t$
  - `optiselect_engine.py` — Ghost-factor capture + scoring + Boltzmann sampling
  - `train_with_selection.py` — training loop with selection
- `src/collect_results.py` — aggregation helper for selection sweeps
- `src/run_shakespeare_exp.sh`, `src/run_wikitext_exp.sh`,
  `src/run_owt2_split.sh`, `src/run_slimpajama_split.sh` — experiment drivers
- `README.md` — this repo's README (the upstream one is preserved in the
  upstream repo's git history; a copy is not redundantly kept here)
- `.gitignore` — extended to exclude datasets / results / logs / checkpoints
- `docs/ATTRIBUTION.md` — this file
