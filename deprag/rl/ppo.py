from typing import TYPE_CHECKING

from trl import PPOConfig, PPOTrainer

from ..configs.config import TrainConfig
from ..models.agent import DeepRAGAgent

if TYPE_CHECKING:
    from .buffers import PPOBuffer


class DeepRAGPPOTrainer(PPOTrainer):
    """A PPO trainer for the DeepRAG agent.

    This class wraps the TRL PPOTrainer to handle the custom
    DeepRAG agent and environment.
    """

    def __init__(self, config: TrainConfig, agent: DeepRAGAgent, tokenizer, **kwargs):
        ppo_config = PPOConfig(
            steps=config.ppo.p_steps,
            epochs=config.ppo.p_epochs,
            ppo_batch_size=config.ppo.p_batch_size,
            init_kl_coef=config.ppo.init_kl_coef,
            target=config.ppo.target_kl,
            adap_kl_ctrl=config.ppo.adap_kl_ctrl,
            gamma=config.ppo.gamma,
            lam=config.ppo.lam,
            cliprange=config.ppo.clip_range,
            cliprange_value=config.ppo.clip_range_vf,
            vf_coef=config.ppo.vf_coef,
            whiten_advantages=config.ppo.whiten_advantages,
            learning_rate=config.learning_rate,
        )
        super().__init__(
            config=ppo_config, model=agent.model, tokenizer=tokenizer, **kwargs
        )

    def train_step(self, buffer: "PPOBuffer") -> dict:
        """Performs a PPO training step.

        Args:
            buffer: The PPO buffer containing trajectories.

        Returns:
            A dictionary of training statistics.
        """
        # The `step` method in TRL's PPOTrainer does the work.
        stats = self.step(
            buffer.query_tensors,
            buffer.response_tensors,
            buffer.rewards,
        )
        return stats
