import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from deprag.configs.config import register_configs


@pytest.fixture(scope="session", autouse=True)
def pre_test_setup():
    """Register all configs before tests run."""
    register_configs()


@pytest.fixture
def hydra_dsi_cfg():
    """Fixture for DSI model config."""
    GlobalHydra.instance().clear()
    with initialize(config_path="../deprag/configs", version_base="1.3"):
        cfg = compose(config_name="defaults", overrides=["data=synthetic", "model=dsi_small"])
        yield cfg


@pytest.fixture
def hydra_agent_cfg():
    """Fixture for Agent model config."""
    GlobalHydra.instance().clear()
    with initialize(config_path="../deprag/configs", version_base="1.3"):
        cfg = compose(config_name="defaults", overrides=["data=synthetic", "model=agent_small"])
        yield cfg