import os
from typing import Dict, List, Optional

from src.cloud_bootstrap import apply_cloud_runtime_patches

apply_cloud_runtime_patches()

import chromadb
from chromadb.api.models.Collection import Collection

from .config import CHROMA_DB_DIR, EMBEDDING_DIMENSION
from .embeddings import embed_texts
from .logger import logger

_chroma_client: Optional[chromadb.api.ClientAPI] = None


def _as_list(value: object) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _as_vector(value: object) -> List[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return list(value.tolist())
    if isinstance(value, list):
        return value
    return list(value)


def create_chroma_client() -> chromadb.api.ClientAPI:
    global _chroma_client
    try:
        if _chroma_client is None:
            chroma_dir = os.getenv("CHROMA_DB_DIR", CHROMA_DB_DIR)
            os.makedirs(chroma_dir, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=chroma_dir)
        return _chroma_client
    except Exception as exc:
        logger.error("Failed to initialize ChromaDB client: %s", exc)
        raise RuntimeError("Unable to connect to the vector database.") from exc


def get_collection(name: str = "fastigo_documents") -> Collection:
    client = create_chroma_client()
    try:
        return client.get_or_create_collection(name=name)
    except Exception:
        logger.info("Creating new collection: %s", name)
        return client.create_collection(name=name)


def get_all_chunks(collection_name: str = "fastigo_documents") -> List[Dict[str, object]]:
    collection = get_collection(collection_name)
    try:
        if collection.count() == 0:
            return []
        results = collection.get(include=["documents", "metadatas", "embeddings"])
        documents = _as_list(results.get("documents"))
        metadatas = _as_list(results.get("metadatas"))
        ids = _as_list(results.get("ids"))
        embeddings = _as_list(results.get("embeddings"))
        records: List[Dict[str, object]] = []
        for idx, document in enumerate(documents):
            records.append(
                {
                    "id": ids[idx] if idx < len(ids) else f"chunk_{idx}",
                    "document": document,
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "embedding": _as_vector(embeddings[idx]) if idx < len(embeddings) else [],
                }
            )
        return records
    except Exception as exc:
        logger.error("Failed to load chunks from ChromaDB: %s", exc)
        raise RuntimeError("Unable to read indexed document chunks.") from exc


def persist_chunks(
    chunks: List[Dict[str, object]],
    collection_name: str = "fastigo_documents",
) -> Collection:
    if not chunks:
        raise ValueError("No chunks provided for persistence.")

    collection = get_collection(collection_name)
    ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    try:
        existing = set(collection.get(ids=ids, include=[]).get("ids", []) or [])
        new_indices = [idx for idx, chunk_id in enumerate(ids) if chunk_id not in existing]
        if not new_indices:
            logger.info("All chunks already indexed; skipping embedding upsert.")
            return collection

        new_ids = [ids[idx] for idx in new_indices]
        new_documents = [documents[idx] for idx in new_indices]
        new_metadatas = [metadatas[idx] for idx in new_indices]
        embeddings = embed_texts(new_documents)
        if len(embeddings) != len(new_documents):
            raise ValueError("Embedding count does not match document count.")
        if embeddings and len(embeddings[0]) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {len(embeddings[0])}."
            )
        collection.upsert(
            ids=new_ids,
            documents=new_documents,
            metadatas=new_metadatas,
            embeddings=embeddings,
        )
    except Exception as exc:
        logger.error("Failed to upsert chunks to ChromaDB: %s", exc)
        raise RuntimeError("Unable to store document chunks in the vector database.") from exc
    return collection


def clear_collection(collection_name: str = "fastigo_documents") -> None:
    global _chroma_client
    client = create_chroma_client()
    try:
        client.delete_collection(name=collection_name)
    except Exception as exc:
        logger.warning("Failed to delete collection: %s", exc)
        collection = get_collection(collection_name)
        try:
            all_ids = collection.get(include=[]).get("ids", []) or []
            if all_ids:
                collection.delete(ids=all_ids)
        except Exception as inner_exc:
            logger.warning("Failed to clear collection records: %s", inner_exc)
    _chroma_client = None


def load_documents(collection_name: str = "fastigo_documents") -> Optional[Collection]:
    return get_collection(collection_name)
