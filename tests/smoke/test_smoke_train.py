import pytest
from hydra import compose, initialize

from deprag.configs.config import register_configs
from deprag.trainers.train_dsi import train_dsi


@pytest.mark.smoke
def test_smoke_train_dsi():
    """Run a single step of DSI training to catch integration errors."""
    register_configs()
    with initialize(config_path="../../deprag/configs", version_base="1.3"):
        cfg = compose(
            config_name="defaults",
            overrides=[
                "data=synthetic",
                "model=dsi_small",
                "train=dsi_pretrain",
                "training.max_steps=1",
                "training.batch_size=1",
                "device=cpu",
            ],
        )
        # The test passes if this runs without raising an exception
        train_dsi(cfg)
