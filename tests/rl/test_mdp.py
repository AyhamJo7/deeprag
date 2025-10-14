import torch

from deprag.rl.mdp import DeepRAGMDP, RLEnvironmentState


def test_mdp_step():
    """Test the state transition logic of the MDP."""
    mdp = DeepRAGMDP(retrieval_penalty=0.1)
    initial_state = RLEnvironmentState(
        query="test query",
        history=torch.tensor([101]),
        retrieved_docs=[],
        step=0,
    )

    # Test a non-retrieval action
    next_state, reward = mdp.step(initial_state, action=500, is_retrieval=False)
    assert reward == 0.0
    assert next_state.step == 1
    assert len(next_state.history) == 2

    # Test a retrieval action
    next_state, reward = mdp.step(initial_state, action=50265, is_retrieval=True)
    assert reward == -0.1
    assert next_state.step == 1
