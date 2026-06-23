from typing import List

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from .config import EMBEDDING_DIMENSION, EMBEDDING_MODEL, GOOGLE_API_KEY, validate_api_settings
from .logger import logger

_EMBED_BATCH_SIZE = 64


def build_embedding_client() -> GoogleGenerativeAIEmbeddings:
    try:
        validate_api_settings()
        client = GoogleGenerativeAIEmbeddings(
            api_key=GOOGLE_API_KEY,
            model=EMBEDDING_MODEL,
            output_dimensionality=EMBEDDING_DIMENSION,
        )
        return client
    except Exception as exc:
        logger.error("Failed to instantiate embedding client: %s", exc)
        raise


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    client = build_embedding_client()
    vectors: List[List[float]] = []
    try:
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            vectors.extend(client.embed_documents(batch))
        return vectors
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        raise RuntimeError(
            "Unable to generate embeddings. Verify GOOGLE_API_KEY and EMBEDDING_MODEL."
        ) from exc


def embed_query(query: str) -> List[float]:
    client = build_embedding_client()
    try:
        return client.embed_query(query)
    except Exception as exc:
        logger.error("Query embedding failed: %s", exc)
        raise RuntimeError(
            "Unable to embed the query. Verify GOOGLE_API_KEY and EMBEDDING_MODEL."
        ) from exc
