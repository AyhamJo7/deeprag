from deprag.models.dsi import DSI


def test_dsi_creation(hydra_cfg):
    """Test that the DSI model can be created."""
    dsi = DSI(hydra_cfg.model)
    assert dsi is not None
    assert dsi.model.config.model_type == "t5"


def test_dsi_retrieval(hydra_cfg):
    """Test the retrieval method of the DSI (inference)."""
    dsi = DSI(hydra_cfg.model)
    queries = ["who is the lead singer of the beatles?"]
    retrieved_docs = dsi.retrieve(queries)
    assert len(retrieved_docs) == 1
    assert len(retrieved_docs[0]) == hydra_cfg.model.top_k
    # The base t5-small model will generate gibberish, so we just check the structure
    assert isinstance(retrieved_docs[0][0], str)
