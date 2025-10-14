from typing import List

from transformers import PreTrainedTokenizer


def add_special_tokens(tokenizer: PreTrainedTokenizer, special_tokens: List[str]) -> None:
    """Adds special tokens to the tokenizer and resizes model embeddings.

    Args:
        tokenizer: The tokenizer to modify.
        special_tokens: A list of special tokens to add (e.g., ['<RET>']).
    """
    # Check which tokens are new
    new_tokens = []
    for token in special_tokens:
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)

    if new_tokens:
        tokenizer.add_tokens(new_tokens)
