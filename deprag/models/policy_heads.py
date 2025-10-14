import torch
import torch.nn as nn
from transformers import PretrainedConfig


class ValueHead(nn.Module):
    """A head for predicting the value of a state.

    This is attached to the backbone language model.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
            if hasattr(config, "hidden_dropout_prob")
            else 0.1
        )
        self.summary = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the value.

        Args:
            hidden_states: The last hidden state from the language model.

        Returns:
            A tensor of shape (batch_size, sequence_length, 1) representing the value.
        """
        output = self.dropout(hidden_states)
        return self.summary(output)
