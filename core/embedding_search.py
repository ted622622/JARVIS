"""Embedding Search — Gemini Embedding semantic search + BM25 hybrid.

Uses google-genai SDK for async embedding, numpy for cosine similarity.
Falls back to pure BM25 when GEMINI_API_KEY is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from google import genai

    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class EmbeddingIndex:
    """Gemini Embedding semantic search engine.

    Usage:
        idx = EmbeddingIndex("./memory", "./data/embedding_index.json", api_key)
        await idx.build_index()
        results = await idx.search("拉麵推薦")
    """

    MODEL = "gemini-embedding-001"
    DIMENSION = 768

    def __init__(
        self,
        memory_dir: str,
        cache_path: str,
        api_key: str,
    ):
        self.memory_dir = Path(memory_dir)
        self.cache_path = Path(cache_path)
        self.api_key = api_key
        self._client: Any = None
        self._chunks: list[str] = []
        self._sources: list[str] = []
        self._embeddings: np.ndarray | None = None  # (N, dim)
        self._hashes: list[str] = []

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def build_index(self) -> int:
        """Scan memory/*.md, embed new/changed chunks, cache results.

        Returns:
            Number of chunks indexed.
        """
        if not _HAS_GENAI or not _HAS_NUMPY:
            logger.warning("google-genai or numpy not installed, embedding search disabled")
            return 0

        # 1. Scan markdown files
        chunks: list[str] = []
        sources: list[str] = []
        for md_file in sorted(self.memory_dir.rglob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue
            paragraphs = re.split(r"\n\n+", content)
            for para in paragraphs:
                para = para.strip()
                if para and len(para) > 5:
                    chunks.append(para)
                    sources.append(str(md_file))

        if not chunks:
            logger.debug("No memory chunks for embedding")
            return 0

        # 2. Compute hashes
        hashes = [hashlib.sha256(c.encode("utf-8")).hexdigest()[:16] for c in chunks]

        # 3. Load cache
        cache = self._load_cache()
        cached_map: dict[str, list[float]] = {}
        if cache.get("model") == self.MODEL:
            for entry in cache.get("chunks", []):
                cached_map[entry["hash"]] = entry["embedding"]

        # 4. Find new/changed chunks
        new_indices = [i for i, h in enumerate(hashes) if h not in cached_map]
        if new_indices:
            new_texts = [chunks[i] for i in new_indices]
            logger.info(f"Embedding {len(new_indices)} new chunks (total {len(chunks)})")
            new_embeddings = await self._embed_texts(new_texts, "RETRIEVAL_DOCUMENT")
            for i, idx in enumerate(new_indices):
                cached_map[hashes[idx]] = new_embeddings[i]

        # 5. Build final arrays
        all_embeddings = []
        for h in hashes:
            all_embeddings.append(cached_map[h])

        self._chunks = chunks
        self._sources = sources
        self._hashes = hashes
        self._embeddings = np.array(all_embeddings, dtype=np.float32)

        # 6. Save cache
        self._save_cache({
            "model": self.MODEL,
            "dimension": self.DIMENSION,
            "chunks": [
                {
                    "text": chunks[i][:500],
                    "source": sources[i],
                    "hash": hashes[i],
                    "embedding": all_embeddings[i],
                }
                for i in range(len(chunks))
            ],
        })

        logger.info(
            f"Embedding index built: {len(chunks)} chunks, "
            f"{len(new_indices)} newly embedded"
        )
        return len(chunks)

    async def _embed_texts(
        self, texts: list[str], task_type: str,
    ) -> list[list[float]]:
        """Embed texts via Gemini API (async, batched)."""
        client = self._get_client()
        result = await client.aio.models.embed_content(
            model=self.MODEL,
            contents=texts,
            config={
                "task_type": task_type,
                "output_dimensionality": self.DIMENSION,
            },
        )
        return [list(e.values) for e in result.embeddings]

    async def search(self, query: str, top_k: int = 6) -> list[dict]:
        """Search memory chunks by semantic similarity.

        Returns:
            List of dicts with text, source, score (same format as BM25).
        """
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        # Embed query
        query_embs = await self._embed_texts([query], "RETRIEVAL_QUERY")
        query_vec = np.array(query_embs[0], dtype=np.float32)

        # Compute similarities
        scores = []
        for i in range(len(self._chunks)):
            sim = _cosine_similarity(query_vec, self._embeddings[i])
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                results.append({
                    "text": self._chunks[idx][:700],
                    "source": self._sources[idx],
                    "score": score,
                })
        return results

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self, data: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )


class HybridSearch:
    """BM25 + Embedding hybrid search.

    Normalizes scores from both engines, merges with configurable weights,
    and boosts items found by both.
    """

    def __init__(
        self,
        bm25: Any,
        embedding: EmbeddingIndex | None,
        bm25_weight: float = 0.3,
        embed_weight: float = 0.7,
    ):
        self.bm25 = bm25
        self.embedding = embedding
        self.bm25_weight = bm25_weight
        self.embed_weight = embed_weight

    async def build_index(self) -> int:
        """Build both BM25 and embedding indices."""
        bm25_count = self.bm25.build_index()
        embed_count = 0
        if self.embedding:
            embed_count = await self.embedding.build_index()
        return max(bm25_count, embed_count)

    async def search(self, query: str, top_k: int = 6) -> list[dict]:
        """Hybrid search: BM25 + Embedding, merged and deduplicated."""
        # BM25 (sync)
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # Embedding (async) — if available
        embed_results = []
        if self.embedding:
            try:
                embed_results = await self.embedding.search(query, top_k=top_k * 2)
            except Exception as e:
                logger.warning(f"Embedding search failed, using BM25 only: {e}")

        if not embed_results:
            return bm25_results[:top_k]

        # Normalize scores (min-max per engine)
        bm25_normed = self._normalize(bm25_results)
        embed_normed = self._normalize(embed_results)

        # Merge: deduplicate by text
        merged: dict[str, dict] = {}
        bm25_texts = set()
        for r in bm25_normed:
            key = r["text"][:200]
            bm25_texts.add(key)
            merged[key] = {
                "text": r["text"],
                "source": r["source"],
                "bm25_score": r["score"],
                "embed_score": 0.0,
            }
        embed_texts = set()
        for r in embed_normed:
            key = r["text"][:200]
            embed_texts.add(key)
            if key in merged:
                merged[key]["embed_score"] = r["score"]
            else:
                merged[key] = {
                    "text": r["text"],
                    "source": r["source"],
                    "bm25_score": 0.0,
                    "embed_score": r["score"],
                }

        # Combined score + boost for items found by both
        results = []
        both_keys = bm25_texts & embed_texts
        for key, entry in merged.items():
            combined = (
                self.bm25_weight * entry["bm25_score"]
                + self.embed_weight * entry["embed_score"]
            )
            if key in both_keys:
                combined += 0.1
            results.append({
                "text": entry["text"],
                "source": entry["source"],
                "score": combined,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_sync(self, query: str, top_k: int = 6) -> list[dict]:
        """Pure BM25 fallback for non-async contexts."""
        return self.bm25.search(query, top_k=top_k)

    @staticmethod
    def _normalize(results: list[dict]) -> list[dict]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return []
        scores = [r["score"] for r in results]
        min_s = min(scores)
        max_s = max(scores)
        spread = max_s - min_s
        if spread == 0:
            return [
                {**r, "score": 1.0 if r["score"] > 0 else 0.0}
                for r in results
            ]
        return [
            {**r, "score": (r["score"] - min_s) / spread}
            for r in results
        ]
