# DeepRAG: End-to-End Differentiable RAG

DeepRAG is a PyTorch-based framework for building and training Retrieval-Augmented Generation (RAG) models where the retrieval step is fully differentiable. It replaces traditional, discrete retrievers (like BM25 or FAISS) with a **Differentiable Search Index (DSI)** and learns a retrieval policy as part of the generation process using reinforcement learning.

## Key Features

- **Differentiable Search Index (DSI)**: A T5-style model that maps queries directly to document identifiers, allowing for end-to-end training.
- **Learned Retrieval Policy**: An agent (Transformer generator) learns *when* to retrieve information using RL, balancing accuracy and efficiency.
- **End-to-End Training**: Jointly optimizes the DSI and the generator through a combination of supervised and reinforcement learning signals.
- **Modular & Extensible**: Built with clean separation between models, data, training loops, and evaluation.
- **Powered by Best Practices**: Uses Hugging Face Transformers, Accelerate, TRL, Hydra for configuration, and Typer for a clean CLI.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AyhamJo7/deeprag.git
    cd deeprag
    ```

2.  **Create a Python virtual environment and install dependencies:**
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e ".[dev,viz]"
    ```

3.  **Install pre-commit hooks for development:**
    ```bash
    pre-commit install
    ```

## Quickstart

1.  **Prepare synthetic data for smoke testing:**
    ```bash
    deeprag
    ```

2.  **Run the smoke tests to verify the installation:**
    ```bash
    pytest -k "smoke or unit"
    ```

3.  **Run a minimal DSI pre-training job:**
    ```bash
    python cli.py train-dsi --config-name=train/dsi_pretrain data=synthetic model=dsi_small training.max_steps=10
    ```

## Usage

The main entrypoint is `cli.py`, which provides several commands:

- `prepare-data`: Download and preprocess datasets.
- `train-dsi`: Pre-train the DSI model.
- `train-agent`: Train the agent with PPO.
- `train-joint`: Jointly fine-tune the DSI and agent.
- `evaluate-model`: Evaluate a trained model on a test set.

All configurations are managed through YAML files in `deprag/configs` and can be overridden from the command line via Hydra.

**Example: Training the DSI on HotpotQA**
```bash
# First, download and prepare the data
./scripts/download_hotpotqa.sh
python cli.py prepare-data --config-name=data/hotpotqa

# Launch the training job
python cli.py train-dsi --config-name=train/dsi_pretrain data=hotpotqa model=dsi_small
```

## Architecture

For a detailed explanation of the architecture, see `docs/architecture.md`.

## Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` for details on how to get started.

## License

This project is licensed under the Apache-2.0 License. See `LICENSE` for more information.
