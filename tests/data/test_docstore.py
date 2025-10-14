from deprag.data.docstore import DocumentStore


def test_docstore_loading():
    """Test that the document store loads data correctly."""
    doc_store = DocumentStore("tests/fixtures/sample_docs.jsonl")
    assert len(doc_store) == 2
    assert len(doc_store.doc_ids) == 2


def test_get_document():
    """Test retrieving a single document."""
    doc_store = DocumentStore("tests/fixtures/sample_docs.jsonl")
    doc_text = doc_store.get_document("doc-The_Weakerthans")
    assert doc_text is not None
    assert "Winnipeg" in doc_text
    assert doc_store.get_document("non_existent_id") is None
