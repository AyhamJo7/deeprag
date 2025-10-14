from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizer


@dataclass
class DSICollator:
    """Collator for DSI pre-training.

    Takes a list of examples, each with a 'query' and a 'doc_id'.
    Tokenizes them and prepares batches for encoder-decoder models.
    """

    tokenizer: PreTrainedTokenizer
    max_source_length: int
    max_target_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        queries = [f["query"] for f in features]
        doc_ids = [f["doc_id"] for f in features]

        model_inputs = self.tokenizer(
            queries,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            doc_ids,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # For T5, the decoder input_ids should be shifted right.
        # The model handles this internally if we pass 'labels'.
        # We replace padding token id in the labels with -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        return model_inputs
