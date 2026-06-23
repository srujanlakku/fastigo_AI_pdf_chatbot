from typing import Dict, List


def _split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space > 0:
                end = start + last_space
                chunk = text[start:end]
        chunk_text = chunk.strip()
        if chunk_text:
            chunks.append(chunk_text)
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def create_chunks(
    documents: List[Dict[str, object]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    file_hash: str = "",
) -> List[Dict[str, object]]:
    chunks: List[Dict[str, object]] = []
    for doc in documents:
        page_text = doc.get("text", "")
        if not page_text:
            continue

        pieces = _split_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_index, piece in enumerate(pieces, start=1):
            chunks.append(
                {
                    "text": piece,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_id": f"{doc.get('file_name', 'unknown')}_p{doc.get('page_number', '0')}_c{chunk_index}",
                        "page_number": doc.get("page_number", 0),
                        "file_hash": file_hash or doc.get("file_hash", ""),
                    },
                }
            )
    return chunks
