import re
import string
from collections import Counter
from typing import List


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(predictions: List[str], references: List[str]) -> float:
    """Computes F1 score between predictions and references."""
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_answer(pred).split()
        ref_tokens = normalize_answer(ref).split()
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            f1_scores.append(0.0)
            continue

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Computes Exact Match score between predictions and references."""
    em_scores = []
    for pred, ref in zip(predictions, references):
        em_scores.append(normalize_answer(pred) == normalize_answer(ref))

    return sum(em_scores) / len(em_scores)
