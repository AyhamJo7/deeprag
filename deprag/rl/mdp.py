from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class RLEnvironmentState:
    """Represents the state of the MDP.

    Attributes:
        query: The initial user query.
        history: The sequence of generated tokens so far.
        retrieved_docs: A list of documents retrieved.
        step: The current generation step.
    """

    query: str
    history: torch.Tensor
    retrieved_docs: List[str]
    step: int


class DeepRAGMDP:
    """Defines the MDP for the DeepRAG agent.

    This class manages the state transitions and rewards.
    """

    def __init__(self, retrieval_penalty: float):
        self.retrieval_penalty = retrieval_penalty

    def step(
        self, state: RLEnvironmentState, action: int, is_retrieval: bool
    ) -> Tuple[RLEnvironmentState, float]:
        """Performs a state transition.

        Args:
            state: The current state.
            action: The token ID of the action taken.
            is_retrieval: Whether the action was a retrieval.

        Returns:
            A tuple of (next_state, reward).
        """
        # Update history
        new_history = torch.cat([state.history, torch.tensor([action])])

        # Determine reward
        reward = 0.0
        if is_retrieval:
            reward -= self.retrieval_penalty

        # Create next state
        next_state = RLEnvironmentState(
            query=state.query,
            history=new_history,
            retrieved_docs=state.retrieved_docs,  # This would be updated after a real retrieval
            step=state.step + 1,
        )

        return next_state, reward
