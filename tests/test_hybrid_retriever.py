from unittest.mock import patch

from src.hybrid_retriever import HybridRetriever


def test_hybrid_search_returns_results():
    retriever = HybridRetriever()
    semantic = [
        {
            "document": "alpha",
            "metadata": {"chunk_id": "a", "file_name": "a.pdf", "page_number": 1},
            "distance": 0.1,
            "semantic_score": 0.9,
            "embedding": [0.1, 0.2],
        }
    ]
    bm25 = [
        {
            "document": "beta",
            "metadata": {"chunk_id": "b", "file_name": "b.pdf", "page_number": 2},
            "bm25_score": 1.0,
            "bm25_normalized": 1.0,
        }
    ]
    with patch.object(retriever, "semantic_search", return_value=semantic), patch.object(
        retriever, "bm25_search", return_value=bm25
    ), patch.object(retriever, "mmr_search", side_effect=lambda *args, **kwargs: kwargs["candidates"][: kwargs["k"]]):
        results = retriever.hybrid_search("query")
        assert len(results) == 2
        assert any(item["metadata"]["chunk_id"] == "a" for item in results)
        assert any(item["metadata"]["chunk_id"] == "b" for item in results)


def test_semantic_search_handles_errors(monkeypatch):
    retriever = HybridRetriever()

    def fail_query(*args, **kwargs):
        raise RuntimeError("failure")

    monkeypatch.setattr(retriever.collection, "query", fail_query)
    results = retriever.semantic_search("query")
    assert results == []


def test_bm25_search_uses_full_corpus():
    retriever = HybridRetriever()
    corpus = [
        {
            "id": "1",
            "document": "machine learning overview",
            "metadata": {"chunk_id": "a", "file_name": "a.pdf", "page_number": 1},
            "embedding": [],
        },
        {
            "id": "2",
            "document": "database indexing strategies",
            "metadata": {"chunk_id": "b", "file_name": "b.pdf", "page_number": 2},
            "embedding": [],
        },
    ]
    with patch("src.hybrid_retriever.get_all_chunks", return_value=corpus):
        results = retriever.bm25_search("machine learning", k=1)
    assert len(results) == 1
    assert "machine learning" in results[0]["document"]
