from typing import Iterator

from datasets import load_dataset, Dataset

from ..configs.config import DataConfig


def load_hotpotqa_data(config: DataConfig) -> Iterator[dict]:
    """Loads the HotpotQA dataset and yields examples.

    Each example contains a 'query' and a 'doc_id' for the positive context.
    """
    dataset = load_dataset(config.path, config.subset)

    for item in dataset[config.train_split]:
        # For DSI, we need query -> doc_id pairs.
        # We can create these from the supporting facts.
        question = item["question"]
        for title, _ in item["supporting_facts"]["title"]:
            # In a real scenario, you would map titles to unique doc_ids.
            # For this example, we'll use a simplified version.
            yield {"query": question, "doc_id": f"doc-{title.replace(' ', '_')}"}


def get_dataset(config: DataConfig) -> Dataset:
    """Generic dataset loader based on the config."""
    if config.dataset_name == "hotpot_qa":
        # This is a placeholder for a more complex loading function.
        # In a real implementation, you'd handle splits and preprocessing here.
        return load_dataset(config.path, config.subset, split=config.train_split)
    elif config.dataset_name == "synthetic":
        return load_dataset("json", data_files=config.path, split="train")
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
