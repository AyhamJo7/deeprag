from typing import Dict, List, Optional

from ..utils.io import read_jsonl


class DocumentStore:
    """A simple in-memory document store.

    Args:
        path (str): Path to the JSONL file containing the documents.
                    Each line should be a dictionary with at least
                    'doc_id' and 'text' keys.
    """

    def __init__(self, path: str):
        self.path = path
        self._data: List[Dict] = []
        self._lookup: Dict[str, str] = {}
        self._build_store()

    def _build_store(self) -> None:
        """Loads data and builds the lookup table."""
        self._data = read_jsonl(self.path)
        self._lookup = {item["doc_id"]: item["text"] for item in self._data}

    def get_document(self, doc_id: str) -> Optional[str]:
        """Retrieve a document by its ID."""
        return self._lookup.get(doc_id)

    def get_documents(self, doc_ids: List[str]) -> List[Optional[str]]:
        """Retrieve multiple documents by their IDs."""
        return [self.get_document(doc_id) for doc_id in doc_ids]

    def __len__(self) -> int:
        return len(self._data)

    @property
    def doc_ids(self) -> List[str]:
        """Return a list of all document IDs."""
        return list(self._lookup.keys())
