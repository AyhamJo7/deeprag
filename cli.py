import hydra
import typer
from hydra import compose, initialize

from deprag.configs.config import register_configs
from deprag.data.prepare import prepare_data as prepare_data_script
from deprag.eval.evaluate import evaluate as evaluate_func
from deprag.trainers.train_agent import train_agent as train_agent_func
from deprag.trainers.train_dsi import train_dsi as train_dsi_func
from deprag.trainers.train_joint import train_joint as train_joint_func

# Register structured configs with Hydra
register_configs()

app = typer.Typer(pretty_exceptions_show_locals=False)


def run_hydra_job(task_function, overrides: list[str]):
    """
    Initializes Hydra and runs a task function with a composed config.
    Parses a --config-name from the overrides list.
    """
    # Default config group to pull from `deprag/configs`
    # This will be the starting point before applying overrides.
    config_name = "defaults"

    # Separate the special `--config-name` from other overrides
    other_overrides = []
    for o in overrides:
        if o.startswith("--config-name"):
            config_name = o.split("=")[1]
        else:
            other_overrides.append(o)

    with initialize(config_path="deprag/configs", version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=other_overrides)
        task_function(cfg)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train_dsi(ctx: typer.Context):
    """Pre-train the Differentiable Search Index (DSI)."""
    run_hydra_job(train_dsi_func, ctx.args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train_agent(ctx: typer.Context):
    """Train the agent using Reinforcement Learning (PPO)."""
    run_hydra_job(train_agent_func, ctx.args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train_joint(ctx: typer.Context):
    """Jointly fine-tune the DSI and the agent."""
    run_hydra_job(train_joint_func, ctx.args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def evaluate_model(ctx: typer.Context):
    """Evaluate a trained DeepRAG model."""
    run_hydra_job(evaluate_func, ctx.args)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def prepare_data(ctx: typer.Context):
    """Prepare datasets and document stores."""
    run_hydra_job(prepare_data_script, ctx.args)


if __name__ == "__main__":
    app()