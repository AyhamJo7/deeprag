# Usage Guide

This guide provides detailed instructions on how to use the `deeprag` CLI.

## Configuration

All training and evaluation runs are configured via YAML files located in `deprag/configs`. The main configuration file is `defaults.yaml`, which sets the default data, model, and training configurations.

You can override any setting from the command line using Hydra's syntax. For example, to change the learning rate:

```bash
python cli.py train-dsi training.learning_rate=1.0e-4
```

## Commands

### `prepare-data`

This command prepares the datasets and document stores.

**Usage:**
```bash
python cli.py prepare-data --config-name=data/<dataset_name>
```

-   `dataset_name`: Can be `hotpotqa` or `synthetic`.

This script will download data if necessary and create a `docstore.jsonl` file in the path specified in the data config.

### `train-dsi`

Pre-trains the Differentiable Search Index.

**Usage:**
```bash
python cli.py train-dsi [hydra overrides]
```

**Example:**
```bash
python cli.py train-dsi data=hotpotqa model=dsi_small training.batch_size=64
```

### `train-agent`

Trains the agent using PPO.

**Usage:**
```bash
python cli.py train-agent [hydra overrides]
```

**Example:**
```bash
python cli.py train-agent data=hotpotqa model=agent_small train=agent_ppo
```

### `evaluate-model`

Evaluates a trained model.

**Usage:**
```bash
python cli.py evaluate-model [hydra overrides]
```

This command will load a checkpoint, run inference on the test set, and compute metrics like Exact Match, F1, and average retrievals per query.
