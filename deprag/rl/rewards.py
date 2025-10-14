from typing import List

import torch
from rouge_score import rouge_scorer

from ..eval.metrics import compute_exact_match


def compute_qa_reward(predictions: List[str], references: List[str]) -> torch.Tensor:
    """Computes a reward based on Exact Match for QA tasks.

    Args:
        predictions: The generated answers.
        references: The ground truth answers.

    Returns:
        A tensor of rewards, one for each example.
    """
    rewards = []
    for pred, ref in zip(predictions, references):
        # Simple binary reward
        rewards.append(1.0 if compute_exact_match(pred, ref) else 0.0)
    return torch.tensor(rewards)


def compute_rouge_reward(
    predictions: List[str], references: List[str], rouge_type: str = "rougeL"
) -> torch.Tensor:
    """Computes a reward based on ROUGE scores.

    Args:
        predictions: The generated summaries or answers.
        references: The ground truth texts.
        rouge_type: The ROUGE metric to use.

    Returns:
        A tensor of rewards, one for each example.
    """
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    rewards = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        rewards.append(score[rouge_type].fmeasure)
    return torch.tensor(rewards)
