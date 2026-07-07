# CLAUDE.md — DeepRAG

## Research Problem

Traditional RAG uses discrete, non-differentiable retrieval (BM25, FAISS) — the retriever can't learn from the generator's feedback. **DeepRAG** makes retrieval end-to-end trainable: a T5-based Differentiable Search Index (DSI) maps queries directly to document identifiers, and a Transformer agent learns *when* to retrieve using reinforcement learning.

## Architecture

```
Query
  → DSI (T5-based) — learns query-to-docID mapping (replaces traditional index)
  → RL Agent (Transformer generator) — decides when to retrieve vs. generate
  → Joint optimization: supervised (DSI) + RL reward signal (accuracy vs. efficiency)
```

Two-component system: **DSI** (retriever) + **RL policy** (when to retrieve). Trained jointly — changes to one affect the other.

## Package Layout

```
deprag/           (main package — note: not src/)
  models/         DSI model, Transformer generator
  rl/             RL policy, reward functions, PPO/GRPO trainer
  trainers/       Training loops (supervised + RL)
  data/           Dataset loaders, tokenization
  eval/           Evaluation harness
  configs/        Hydra configs (defaults.yaml + nested overrides)
  cli.py          CLI entrypoint
```

## Setup & Commands

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"   # viz extras for experiment plots
pre-commit install

make setup       # full setup (venv + install + pre-commit)
make lint        # ruff + mypy
make test        # pytest tests/ — includes unit, models, rl, smoke
make train       # launch training (requires GPU)
make evaluate    # run eval on trained checkpoint
```

## Configuration

Uses **Hydra** — override configs from CLI:

```bash
deeprag train model=t5-base train.batch_size=32
deeprag train +experiment=dpr_nq              # load experiment preset
deeprag evaluate model.checkpoint=outputs/run_42/best.ckpt
```

Config files: `deprag/configs/defaults.yaml` → overrides in `model/`, `train/`, `data/`

## Training Notes

- **GPU required** for training. CPU-only runs work for unit tests and smoke tests only.
- The RL phase is sensitive to reward shaping — see `deprag/rl/` for reward function definitions.
- Smoke tests in `tests/smoke/` verify the pipeline runs end-to-end on tiny synthetic data without a GPU.
- TRL (`trl`) handles PPO; `accelerate` handles distributed training setup.

## Evaluation

Benchmarks: Natural Questions (NQ), TriviaQA, WebQ

Metrics: Exact Match, F1, retrieval efficiency (avg retrievals per query)

## Key Dependencies

- `transformers` + `accelerate` — T5, distributed training
- `trl` — RL training (PPO/GRPO)
- `hydra-core` — config management
- `torch` — core framework (GPU strongly recommended)
- `datasets` (HuggingFace) — NQ, TriviaQA data

## Research Notes

The critical tradeoff is retrieval accuracy vs. efficiency — the RL agent learns to skip retrieval when it has enough context. When tuning rewards, watch both the final EM score and the average number of retrievals per question; improving one often hurts the other.
