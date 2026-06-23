import os
from unittest.mock import patch

import chromadb
from chromadb import PersistentClient

from src.config import EMBEDDING_DIMENSION
from src.vector_store import create_chroma_client, get_collection, persist_chunks, clear_collection


def _mock_embeddings(texts):
    return [[0.1] * EMBEDDING_DIMENSION for _ in texts]


def test_create_chroma_client_and_collection(tmp_path):
    os.environ["CHROMA_DB_DIR"] = str(tmp_path / "chromadb")
    client = create_chroma_client()
    assert client is not None
    collection = get_collection("test_collection")
    assert collection.name == "test_collection"


def test_persist_chunks_and_clear_collection(tmp_path):
    os.environ["CHROMA_DB_DIR"] = str(tmp_path / "chromadb")
    chunks = [
        {
            "text": "hello world",
            "metadata": {"file_name": "a.pdf", "page_number": 1, "chunk_id": "a.pdf_p1_c1"},
        }
    ]
    with patch("src.vector_store.embed_texts", side_effect=_mock_embeddings):
        collection = persist_chunks(chunks, collection_name="test_collection")
    assert collection.count() >= 1
    clear_collection("test_collection")
