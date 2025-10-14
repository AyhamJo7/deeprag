import pytest
from hydra import compose, initialize

from deprag.configs.config import register_configs


@pytest.fixture(scope="session")
def hydra_cfg():
    """Fixture to initialize Hydra and get the default config."""
    register_configs()
    with initialize(config_path="../deprag/configs", version_base="1.3"):
        cfg = compose(
            config_name="defaults", overrides=["data=synthetic", "model=dsi_small"]
        )
        yield cfg
