from typing import List

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from ..configs.config import ModelConfig


class DSI:
    """Differentiable Search Index (DSI) model.

    This model learns to map natural language queries to document identifiers.
    It is based on a T5 encoder-decoder architecture.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.model_name_or_path
        )
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name_or_path)

    def train_step(self, batch: dict) -> torch.Tensor:
        """Performs a single training step and returns the loss."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    @torch.no_grad()
    def retrieve(self, queries: List[str]) -> List[List[str]]:
        """Given a list of queries, retrieve the top-k doc IDs for each."""
        inputs = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.model.device)

        # Generate doc ID sequences
        generated_ids = self.model.generate(
            **inputs,
            max_length=self.config.max_answer_length,
            num_beams=self.config.top_k,
            num_return_sequences=self.config.top_k,
        )

        # Decode the generated IDs
        decoded_preds = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Group predictions by query
        retrieved_docs = []
        for i in range(len(queries)):
            start = i * self.config.top_k
            end = start + self.config.top_k
            retrieved_docs.append(decoded_preds[start:end])

        return retrieved_docs
