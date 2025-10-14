from dataclasses import dataclass

import torch


@dataclass
class PPOBuffer:
    """Buffer to store PPO trajectories.

    Attributes:
        query_tensors: The input queries.
        response_tensors: The generated responses.
        logprobs: Log probabilities of the actions taken.
        values: Value estimates for the states.
        rewards: Rewards received.
        advantages: Advantages computed from rewards and values.
        returns: Returns computed from rewards.
    """

    query_tensors: torch.Tensor
    response_tensors: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor

    def __post_init__(self):
        self.advantages: torch.Tensor = torch.zeros_like(self.rewards)
        self.returns: torch.Tensor = torch.zeros_like(self.rewards)

    def compute_advantages_and_returns(self, gamma: float, lam: float) -> None:
        """Computes advantages and returns using GAE."""
        last_gae_lam = 0
        for t in reversed(range(self.rewards.size(1))):
            next_values = self.values[:, t + 1] if t < self.rewards.size(1) - 1 else 0
            delta = self.rewards[:, t] + gamma * next_values - self.values[:, t]
            last_gae_lam = delta + gamma * lam * last_gae_lam
            self.advantages[:, t] = last_gae_lam
        self.returns = self.advantages + self.values
