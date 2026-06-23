from unittest.mock import patch

from src.embeddings import embed_texts


def test_embed_texts_batches_large_inputs():
    texts = [f"document chunk {index}" for index in range(150)]
    with patch("src.embeddings.build_embedding_client") as mock_client:
        instance = mock_client.return_value
        instance.embed_documents.side_effect = lambda batch: [[0.1, 0.2] for _ in batch]
        vectors = embed_texts(texts)
    assert len(vectors) == 150
    assert instance.embed_documents.call_count == 3
