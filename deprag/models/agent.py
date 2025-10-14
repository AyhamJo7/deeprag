from typing import Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from ..configs.config import ModelConfig
from .policy_heads import ValueHead


class DeepRAGAgent(nn.Module):
    """The DeepRAG Agent model.

    This model consists of a pretrained language model (the actor)
    and a value head for reinforcement learning.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Load the pretrained model
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)

        # Apply LoRA if configured
        if config.use_lora:
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        # The value head
        self.value_head = ValueHead(self.model.config)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get logits and value estimates."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits
        last_hidden_state = outputs.hidden_states[-1]
        values = self.value_head(last_hidden_state).squeeze(-1)

        return logits, values

    def generate(
        self, input_ids: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences and compute their log probabilities."""
        # This is a simplified generate function for PPO.
        # It generates sequences and then re-computes the forward pass
        # to get logits for the generated tokens.

        generated_sequences = self.model.generate(input_ids, **kwargs)

        # Get logits and values for the full sequence
        attention_mask = (generated_sequences != kwargs.get("pad_token_id")).long()
        logits, values = self.forward(generated_sequences, attention_mask)

        return generated_sequences, logits, values
