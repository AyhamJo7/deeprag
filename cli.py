import hydra
import typer

from deprag.configs.config import register_configs
from deprag.trainers.train_agent import train_agent as train_agent_func
from deprag.trainers.train_dsi import train_dsi as train_dsi_func
from deprag.trainers.train_joint import train_joint as train_joint_func
from deprag.eval.evaluate import evaluate as evaluate_func
from deprag.scripts.prepare_data import prepare_data as prepare_data_script

# Register structured configs with Hydra
register_configs()

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train_dsi(ctx: typer.Context):
    """Pre-train the Differentiable Search Index (DSI)."""
    run_hydra("train/dsi_pretrain", train_dsi_func, ctx.args)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train_agent(ctx: typer.Context):
    """Train the agent using Reinforcement Learning (PPO)."""
    run_hydra("train/agent_ppo", train_agent_func, ctx.args)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train_joint(ctx: typer.Context):
    """Jointly fine-tune the DSI and the agent."""
    run_hydra("train/joint_finetune", train_joint_func, ctx.args)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def evaluate_model(ctx: typer.Context):
    """Evaluate a trained DeepRAG model."""
    run_hydra("eval", evaluate_func, ctx.args)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def prepare_data(ctx: typer.Context):
    """Prepare datasets and document stores."""
    run_hydra("data/hotpotqa", prepare_data_script, ctx.args)


def run_hydra(config_name: str, task_function, overrides: list[str]):
    """Utility to run a Hydra job."""
    hydra.main(
        config_path="deprag/configs", config_name=config_name, version_base="1.3"
    )(lambda cfg: task_function(cfg))(overrides)


if __name__ == "__main__":
    app()
