from typing import List

from rank_bm25 import BM25Okapi
from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever

from ..data.docstore import DocumentStore


class BM25Retriever:
    """A simple BM25 retriever."""

    def __init__(self, doc_store: DocumentStore):
        self.doc_store = doc_store
        self.doc_ids = doc_store.doc_ids
        corpus = [doc_store.get_document(doc_id) for doc_id in self.doc_ids]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve the top-k document IDs for a query."""
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = doc_scores.argsort()[-k:][::-1]
        return [self.doc_ids[i] for i in top_k_indices]


class BaselineRAG:
    """A baseline RAG model using Hugging Face's implementation."""

    def __init__(self, model_name: str = "facebook/rag-token-nq"):
        self.tokenizer = RagTokenizer.from_pretrained(model_name)
        self.retriever = RagRetriever.from_pretrained(
            model_name, index_name="exact", use_dummy_dataset=True
        )
        self.model = RagTokenForGeneration.from_pretrained(
            model_name, retriever=self.retriever
        )

    def generate(self, queries: List[str]) -> List[str]:
        """Generate answers for a list of queries."""
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True)
        generated_ids = self.model.generate(input_ids=inputs["input_ids"])
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
