from src.chunking import create_chunks


def test_create_chunks_preserves_metadata():
    documents = [
        {
            "file_name": "a.pdf",
            "page_number": 1,
            "text": "This is a test document. " * 100,
            "metadata": {"file_name": "a.pdf", "page_number": 1},
        }
    ]
    chunks = create_chunks(documents, chunk_size=50, chunk_overlap=10)
    assert chunks
    assert all(chunk["metadata"]["file_name"] == "a.pdf" for chunk in chunks)
    assert all(chunk["metadata"]["page_number"] == 1 for chunk in chunks)
    assert all("chunk_id" in chunk["metadata"] for chunk in chunks)
