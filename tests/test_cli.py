from typer.testing import CliRunner

from deprag.cli import app

runner = CliRunner()


def test_train_dsi_help():
    """Test that the train-dsi command shows help text."""
    result = runner.invoke(app, ["train-dsi", "--help"])
    assert result.exit_code == 0
    assert "Pre-train the Differentiable Search Index" in result.stdout


def test_train_agent_help():
    """Test that the train-agent command shows help text."""
    result = runner.invoke(app, ["train-agent", "--help"])
    assert result.exit_code == 0
    assert "Train the agent using Reinforcement Learning" in result.stdout


def test_prepare_data_help():
    """Test that the prepare-data command shows help text."""
    result = runner.invoke(app, ["prepare-data", "--help"])
    assert result.exit_code == 0
    assert "Prepare datasets and document stores" in result.stdout
