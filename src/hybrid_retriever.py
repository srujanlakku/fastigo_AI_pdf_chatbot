import math
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from .embeddings import embed_query
from .logger import logger
from .vector_store import get_all_chunks, get_collection


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(min_score, max_score):
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class HybridRetriever:
    def __init__(self, collection_name: str = "fastigo_documents"):
        self.collection = get_collection(collection_name)
        self.collection_name = collection_name

    def _load_corpus(self) -> Tuple[List[str], List[Dict[str, object]], List[str], List[List[float]]]:
        records = get_all_chunks(self.collection_name)
        if not records:
            return [], [], [], []

        documents = [record["document"] for record in records]
        metadatas = [record["metadata"] for record in records]
        ids = [record["id"] for record in records]
        embeddings = [record.get("embedding") or [] for record in records]
        return documents, metadatas, ids, embeddings

    def semantic_search(self, query: str, k: int = 5, fetch_k: int = 20) -> List[Dict[str, object]]:
        try:
            query_embedding = embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
            batch_documents = results.get("documents", [])
            batch_metadatas = results.get("metadatas", [])
            batch_distances = results.get("distances", [])
            batch_embeddings = results.get("embeddings", [])
            if not batch_documents or not batch_documents[0]:
                return []

            documents = batch_documents[0]
            metadatas = batch_metadatas[0]
            distances = batch_distances[0]
            embeddings = batch_embeddings[0] if batch_embeddings else []

            hits: List[Dict[str, object]] = []
            for idx, metadata in enumerate(metadatas):
                distance = distances[idx] if idx < len(distances) else None
                if distance is None:
                    continue
                similarity = 1.0 / (1.0 + float(distance))
                hits.append(
                    {
                        "document": documents[idx],
                        "metadata": metadata,
                        "distance": distance,
                        "semantic_score": similarity,
                        "embedding": embeddings[idx] if idx < len(embeddings) else [],
                    }
                )
            return sorted(hits, key=lambda item: item["semantic_score"], reverse=True)[:k]
        except Exception as exc:
            logger.warning("Semantic retrieval failed: %s", exc)
            return []

    def bm25_search(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        try:
            documents, metadatas, _, _ = self._load_corpus()
            if not documents:
                return []

            tokenized_corpus = [_tokenize(doc) for doc in documents]
            if not any(tokenized_corpus):
                return []

            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(_tokenize(query))
            normalized = _normalize_scores(list(scores))

            ranked = sorted(
                [
                    {
                        "document": documents[i],
                        "metadata": metadatas[i],
                        "bm25_score": float(scores[i]),
                        "bm25_normalized": normalized[i],
                    }
                    for i in range(len(documents))
                ],
                key=lambda item: item["bm25_score"],
                reverse=True,
            )
            return ranked[:k]
        except Exception as exc:
            logger.warning("BM25 retrieval failed: %s", exc)
            return []

    def _fuse_results(
        self,
        semantic_results: List[Dict[str, object]],
        bm25_results: List[Dict[str, object]],
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> List[Dict[str, object]]:
        combined: Dict[str, Dict[str, object]] = {}

        for rank, item in enumerate(semantic_results):
            chunk_id = item["metadata"]["chunk_id"]
            rrf = 1.0 / (60 + rank + 1)
            combined[chunk_id] = {
                **item,
                "fusion_score": semantic_weight * (item.get("semantic_score", 0.0) + rrf),
            }

        for rank, item in enumerate(bm25_results):
            chunk_id = item["metadata"]["chunk_id"]
            rrf = 1.0 / (60 + rank + 1)
            if chunk_id in combined:
                combined[chunk_id]["bm25_score"] = item.get("bm25_score", 0.0)
                combined[chunk_id]["bm25_normalized"] = item.get("bm25_normalized", 0.0)
                combined[chunk_id]["fusion_score"] = float(combined[chunk_id]["fusion_score"]) + bm25_weight * (
                    item.get("bm25_normalized", 0.0) + rrf
                )
            else:
                combined[chunk_id] = {
                    **item,
                    "fusion_score": bm25_weight * (item.get("bm25_normalized", 0.0) + rrf),
                }

        return sorted(combined.values(), key=lambda item: item.get("fusion_score", 0.0), reverse=True)

    def hybrid_search(self, query: str, k: int = 5, fetch_k: int = 20) -> List[Dict[str, object]]:
        semantic_results = self.semantic_search(query, k=fetch_k, fetch_k=fetch_k)
        bm25_results = self.bm25_search(query, k=fetch_k)
        if not semantic_results and not bm25_results:
            return []
        if not semantic_results:
            return bm25_results[:k]
        if not bm25_results:
            return semantic_results[:k]

        fused = self._fuse_results(semantic_results, bm25_results)[:fetch_k]
        embedding_map = {
            item["metadata"]["chunk_id"]: item.get("embedding", [])
            for item in semantic_results
            if item.get("embedding")
        }
        if len(embedding_map) < len(fused):
            for record in get_all_chunks(self.collection_name):
                chunk_id = record["metadata"].get("chunk_id")
                if chunk_id and chunk_id not in embedding_map:
                    embedding_map[chunk_id] = record.get("embedding") or []
        for item in fused:
            chunk_id = item["metadata"]["chunk_id"]
            item["embedding"] = embedding_map.get(chunk_id, item.get("embedding", []))
        return self.mmr_search(query, candidates=fused, k=k)

    def mmr_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        candidates: Optional[List[Dict[str, object]]] = None,
        lambda_mult: float = 0.7,
    ) -> List[Dict[str, object]]:
        try:
            if candidates is None:
                candidates = self._fuse_results(
                    self.semantic_search(query, k=fetch_k, fetch_k=fetch_k),
                    self.bm25_search(query, k=fetch_k),
                )
            if not candidates:
                return []

            query_embedding = embed_query(query)
            selected: List[Dict[str, object]] = []
            remaining = list(candidates)

            while remaining and len(selected) < k:
                best_idx = 0
                best_score = float("-inf")
                for idx, candidate in enumerate(remaining):
                    candidate_embedding = candidate.get("embedding") or []
                    relevance = candidate.get("semantic_score")
                    if relevance is None:
                        relevance = candidate.get("bm25_normalized", 0.0)
                    if relevance is None or relevance == 0.0:
                        relevance = _cosine_similarity(query_embedding, candidate_embedding)
                    redundancy = 0.0
                    if selected:
                        selected_embeddings = [item.get("embedding") or [] for item in selected]
                        redundancy = max(
                            _cosine_similarity(candidate_embedding, chosen)
                            for chosen in selected_embeddings
                            if chosen
                        ) or 0.0
                    mmr_score = lambda_mult * float(relevance) - (1.0 - lambda_mult) * redundancy
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                selected.append(remaining.pop(best_idx))

            return selected
        except Exception as exc:
            logger.warning("MMR retrieval failed, falling back to fused ranking: %s", exc)
            if candidates:
                return candidates[:k]
            return self.semantic_search(query, k=k, fetch_k=fetch_k)
