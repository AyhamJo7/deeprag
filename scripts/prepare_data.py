import json

import hydra
from datasets import load_dataset
from tqdm import tqdm

from deprag.configs.config import DeepRAGConfig
from deprag.utils.io import write_jsonl
from deprag.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_data(config: DeepRAGConfig) -> None:
    """Loads a dataset, extracts a document store, and saves it.

    A document store is a JSONL file where each line is a dictionary
    with 'doc_id' and 'text' keys.
    """
    logger.info(f"Preparing data for: {config.data.dataset_name}")
    logger.info(f"Output document store: {config.data.doc_store_path}")

    if config.data.dataset_name == "synthetic":
        # For the synthetic dataset, the docs are part of the fixture file.
        # We extract them here.
        with open(config.data.path, "r") as f:
            data = json.load(f)

        documents = []
        for item in data["validation"]:
            for title, sentences in zip(
                item["context"]["title"], item["context"]["sentences"]
            ):
                doc_id = f"doc-{title.replace(' ', '_')}"
                text = " ".join(sentences)
                documents.append({"doc_id": doc_id, "text": text})

    elif config.data.dataset_name == "hotpot_qa":
        dataset = load_dataset(config.data.path, config.data.subset)
        documents = []
        seen_titles = set()

        for split in [config.data.train_split, config.data.val_split]:
            for item in tqdm(dataset[split], desc=f"Processing {split} split"):
                for title, sentences in zip(
                    item["context"]["title"], item["context"]["sentences"]
                ):
                    if title not in seen_titles:
                        doc_id = f"doc-{title.replace(' ', '_')}"
                        text = " ".join(sentences)
                        documents.append({"doc_id": doc_id, "text": text})
                        seen_titles.add(title)
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset_name}")

    # Write the document store to a file
    write_jsonl(documents, config.data.doc_store_path)
    logger.info(f"Successfully created document store with {len(documents)} documents.")


@hydra.main(config_path="../deprag/configs", config_name="defaults", version_base="1.3")
def main(cfg: DeepRAGConfig) -> None:
    prepare_data(cfg)


if __name__ == "__main__":
    main()
