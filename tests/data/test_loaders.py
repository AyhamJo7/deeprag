from deprag.data.loaders import get_dataset


def test_get_synthetic_dataset(hydra_dsi_cfg):
    """Test loading the synthetic dataset."""
    dataset = get_dataset(hydra_dsi_cfg.data)
    assert dataset is not None
    assert len(dataset) == 1
    assert "question" in dataset.features
