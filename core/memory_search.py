"""Memory Search — BM25 keyword search over markdown memory files.

Scans memory/ directory for .md files, tokenizes Chinese text,
and provides BM25-based search.

Phase 2: add embedding-based semantic search.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


def _tokenize_chinese(text: str) -> list[str]:
    """Simple character + word tokenizer for Chinese text.

    Splits on whitespace and punctuation, keeps Chinese character bigrams
    for better matching, plus individual words/chars.
    """
    # Remove markdown formatting
    text = re.sub(r"[#*>\-`\[\]()]", " ", text)
    # Split into tokens
    tokens = []
    for word in text.split():
        tokens.append(word.lower())
        # Add character bigrams for Chinese text
        if any("\u4e00" <= c <= "\u9fff" for c in word):
            for i in range(len(word) - 1):
                if "\u4e00" <= word[i] <= "\u9fff":
                    tokens.append(word[i : i + 2])
    return [t for t in tokens if t.strip()]


class MemorySearch:
    """BM25-based search over markdown memory files.

    Usage:
        search = MemorySearch("./memory")
        search.build_index()
        results = search.search("拉麵")
    """

    def __init__(self, memory_dir: str = "./memory"):
        self.memory_dir = Path(memory_dir)
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.bm25: BM25Okapi | None = None

    def build_index(self) -> int:
        """Scan memory/ for .md files, build BM25 index.

        Returns:
            Number of chunks indexed.
        """
        if not _HAS_BM25:
            logger.warning("rank_bm25 not installed, search disabled")
            return 0

        self.chunks = []
        self.sources = []

        for md_file in sorted(self.memory_dir.rglob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            # Split by paragraphs (double newline)
            paragraphs = re.split(r"\n\n+", content)
            for para in paragraphs:
                para = para.strip()
                if para and len(para) > 5:  # skip tiny fragments
                    self.chunks.append(para)
                    self.sources.append(str(md_file))

        if not self.chunks:
            logger.debug("No memory chunks to index")
            return 0

        tokenized = [_tokenize_chinese(chunk) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        logger.info(f"Memory search index built: {len(self.chunks)} chunks from {len(set(self.sources))} files")
        return len(self.chunks)

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        """Search memory chunks, return top_k results.

        Args:
            query: search query
            top_k: max results to return

        Returns:
            List of dicts with text, source, score
        """
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = _tokenize_chinese(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": self.chunks[idx][:700],
                    "source": self.sources[idx],
                    "score": float(scores[idx]),
                })

        return results
