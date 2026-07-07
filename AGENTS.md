# DeepRAG Agent Notes

DeepRAG is a research portfolio repo for end-to-end differentiable retrieval
augmented generation. It explores a Differentiable Search Index (DSI) plus a
learned retrieval policy trained with supervised and reinforcement learning
signals.

Read local `CLAUDE.md` if present; in this workspace it may be untracked.

## Architecture

- DSI model maps queries directly to document identifiers.
- Agent/generator learns when to retrieve and how much to retrieve.
- Training uses Hugging Face Transformers, Accelerate, TRL, Hydra configs, and
  PyTorch.
- Configs live under `deprag/configs`.
- Main CLI surface is `cli.py` with commands for data prep, DSI training,
  agent training, joint training, and evaluation.

## Commands

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,viz]"
pytest -k "smoke or unit"
python cli.py train-dsi --config-name=train/dsi_pretrain data=synthetic model=dsi_small training.max_steps=10
```

Use small synthetic-data smoke runs before launching expensive training.

## Editing Rules

- Keep data/model/training/evaluation boundaries clean; this repo is research,
  so reproducibility matters more than quick scripts.
- Do not commit checkpoints, datasets, generated reports, wandb artifacts, or
  local notebooks with outputs.
- Prefer Hydra config overrides over hard-coded training constants.
- Add tests for data collators and training-loop assumptions before changing
  expected batch keys.
