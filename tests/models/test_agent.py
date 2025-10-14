import torch

from deprag.models.agent import DeepRAGAgent


def test_agent_creation(hydra_cfg):
    """Test that the DeepRAGAgent can be created."""
    agent = DeepRAGAgent(hydra_cfg.model)
    assert agent is not None
    assert agent.value_head is not None


def test_agent_forward_pass(hydra_cfg):
    """Test the forward pass of the agent."""
    agent = DeepRAGAgent(hydra_cfg.model)
    input_ids = torch.randint(0, 1000, (2, 10))
    logits, values = agent.forward(input_ids)
    assert logits.shape == (2, 10, agent.model.config.vocab_size)
    assert values.shape == (2, 10)
