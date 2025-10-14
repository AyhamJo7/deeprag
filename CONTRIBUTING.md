# Contributing to DeepRAG

First off, thank you for considering contributing to DeepRAG! Any contribution, from fixing a typo to implementing a new feature, is greatly appreciated.

## How to Contribute

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally: `git clone https://github.com/AyhamJo7/deeprag.git`
3.  **Create a new branch** for your changes: `git checkout -b feat/my-new-feature`.
4.  **Install dependencies** in editable mode: `pip install -e ".[dev,viz]"`.
5.  **Set up pre-commit hooks**: `pre-commit install`.
6.  **Make your changes**. Ensure you add or update tests as appropriate.
7.  **Run tests and linting**: `pytest` and `pre-commit run --all-files`.
8.  **Commit your changes** with a descriptive commit message.
9.  **Push your branch** to your fork: `git push origin feat/my-new-feature`.
10. **Open a pull request** to the `main` branch of the original repository.

## Code Style

This project uses `black` for code formatting, `ruff` for linting, and `mypy` for type checking. The pre-commit hooks will enforce this style automatically.

## Reporting Bugs

If you find a bug, please open an issue on the GitHub repository. Include a clear description of the bug, a minimal reproducible example, and any relevant logs or error messages.
