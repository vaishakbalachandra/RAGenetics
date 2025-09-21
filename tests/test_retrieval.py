from ragenetics.retrieval.vectorstore import LocalBM25Store


def test_empty_store():
    """
    Ensure similarity_search returns an empty list when no docs are present.
    """
    s = LocalBM25Store()
    out = s.similarity_search("x")
    assert out == []
