"""Tests for EmbeddingIndex, HybridSearch, and cosine similarity."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.embedding_search import EmbeddingIndex, HybridSearch, _cosine_similarity


# ── cosine_similarity ──────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        import numpy as np
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert abs(_cosine_similarity(a, b) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        import numpy as np
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        import numpy as np
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_handled(self):
        import numpy as np
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        # Should not raise, returns ~0 due to 1e-9 epsilon
        result = _cosine_similarity(a, b)
        assert abs(result) < 1e-3

    def test_known_similarity(self):
        import numpy as np
        a = np.array([1.0, 1.0])
        b = np.array([1.0, 0.0])
        # cos(45°) ≈ 0.707
        result = _cosine_similarity(a, b)
        assert abs(result - 0.7071) < 0.01


# ── EmbeddingIndex ──────────────────────────────────────────────


def _make_memory_dir(tmp_path: Path) -> Path:
    """Create a temp memory directory with test markdown files."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "test.md").write_text(
        "# Test Memory\n\nTed喜歡吃拉麵\n\n今天天氣很好\n\n明天要開會",
        encoding="utf-8",
    )
    return mem_dir


class TestEmbeddingIndexInit:
    def test_attributes(self, tmp_path):
        idx = EmbeddingIndex(
            memory_dir=str(tmp_path),
            cache_path=str(tmp_path / "cache.json"),
            api_key="test-key",
        )
        assert idx.api_key == "test-key"
        assert idx.memory_dir == tmp_path


class TestEmbeddingIndexBuild:
    @pytest.mark.asyncio
    async def test_build_index_embeds_chunks(self, tmp_path):
        """build_index should call embed API for new chunks."""
        mem_dir = _make_memory_dir(tmp_path)
        cache_path = tmp_path / "cache.json"

        idx = EmbeddingIndex(
            memory_dir=str(mem_dir),
            cache_path=str(cache_path),
            api_key="test-key",
        )

        # Mock the genai client
        import numpy as np
        fake_embeddings = [
            MagicMock(values=list(np.random.rand(768).astype(float)))
            for _ in range(3)  # 3 chunks expected
        ]
        mock_result = MagicMock()
        mock_result.embeddings = fake_embeddings

        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)
        idx._client = mock_client

        with patch("core.embedding_search._HAS_GENAI", True), \
             patch("core.embedding_search._HAS_NUMPY", True):
            count = await idx.build_index()
        assert count == 3
        assert len(idx._chunks) == 3
        assert idx._embeddings is not None
        assert idx._embeddings.shape == (3, 768)
        assert cache_path.exists()

    @pytest.mark.asyncio
    async def test_build_index_cache_hit(self, tmp_path):
        """Second build should not re-embed cached chunks."""
        import numpy as np
        mem_dir = _make_memory_dir(tmp_path)
        cache_path = tmp_path / "cache.json"

        idx = EmbeddingIndex(
            memory_dir=str(mem_dir),
            cache_path=str(cache_path),
            api_key="test-key",
        )

        # First build
        fake_embeddings = [
            MagicMock(values=list(np.random.rand(768).astype(float)))
            for _ in range(3)
        ]
        mock_result = MagicMock()
        mock_result.embeddings = fake_embeddings

        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)
        idx._client = mock_client

        with patch("core.embedding_search._HAS_GENAI", True), \
             patch("core.embedding_search._HAS_NUMPY", True):
            await idx.build_index()
        assert mock_client.aio.models.embed_content.call_count == 1

        # Second build — same content, should use cache
        idx2 = EmbeddingIndex(
            memory_dir=str(mem_dir),
            cache_path=str(cache_path),
            api_key="test-key",
        )
        mock_client2 = MagicMock()
        mock_client2.aio.models.embed_content = AsyncMock()
        idx2._client = mock_client2

        with patch("core.embedding_search._HAS_GENAI", True), \
             patch("core.embedding_search._HAS_NUMPY", True):
            count = await idx2.build_index()
        assert count == 3
        # Should NOT have called embed API since all chunks are cached
        mock_client2.aio.models.embed_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_index_cache_miss_new_chunk(self, tmp_path):
        """Adding a new chunk should trigger API call only for the new one."""
        import numpy as np
        mem_dir = _make_memory_dir(tmp_path)
        cache_path = tmp_path / "cache.json"

        idx = EmbeddingIndex(
            memory_dir=str(mem_dir),
            cache_path=str(cache_path),
            api_key="test-key",
        )

        # First build
        fake_embeddings = [
            MagicMock(values=list(np.random.rand(768).astype(float)))
            for _ in range(3)
        ]
        mock_result = MagicMock()
        mock_result.embeddings = fake_embeddings
        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)
        idx._client = mock_client
        with patch("core.embedding_search._HAS_GENAI", True), \
             patch("core.embedding_search._HAS_NUMPY", True):
            await idx.build_index()

        # Add a new chunk
        (mem_dir / "new.md").write_text("新的一段記憶內容需要超過五個字", encoding="utf-8")

        # Second build
        idx2 = EmbeddingIndex(
            memory_dir=str(mem_dir),
            cache_path=str(cache_path),
            api_key="test-key",
        )
        new_embedding = [MagicMock(values=list(np.random.rand(768).astype(float)))]
        mock_result2 = MagicMock()
        mock_result2.embeddings = new_embedding
        mock_client2 = MagicMock()
        mock_client2.aio.models.embed_content = AsyncMock(return_value=mock_result2)
        idx2._client = mock_client2

        with patch("core.embedding_search._HAS_GENAI", True), \
             patch("core.embedding_search._HAS_NUMPY", True):
            count = await idx2.build_index()
        assert count == 4
        # Only the 1 new chunk should be embedded
        call_args = mock_client2.aio.models.embed_content.call_args
        texts = call_args[1]["contents"]
        assert len(texts) == 1

    @pytest.mark.asyncio
    async def test_build_index_no_genai(self, tmp_path):
        """Gracefully returns 0 when google-genai is not installed."""
        mem_dir = _make_memory_dir(tmp_path)
        idx = EmbeddingIndex(
            memory_dir=str(mem_dir),
            cache_path=str(tmp_path / "cache.json"),
            api_key="test-key",
        )

        with patch("core.embedding_search._HAS_GENAI", False):
            count = await idx.build_index()
        assert count == 0

    @pytest.mark.asyncio
    async def test_build_index_no_numpy(self, tmp_path):
        """Gracefully returns 0 when numpy is not installed."""
        mem_dir = _make_memory_dir(tmp_path)
        idx = EmbeddingIndex(
            memory_dir=str(mem_dir),
            cache_path=str(tmp_path / "cache.json"),
            api_key="test-key",
        )

        with patch("core.embedding_search._HAS_NUMPY", False):
            count = await idx.build_index()
        assert count == 0

    @pytest.mark.asyncio
    async def test_build_index_empty_dir(self, tmp_path):
        """Empty memory dir returns 0."""
        mem_dir = tmp_path / "empty_mem"
        mem_dir.mkdir()
        idx = EmbeddingIndex(
            memory_dir=str(mem_dir),
            cache_path=str(tmp_path / "cache.json"),
            api_key="test-key",
        )
        mock_client = MagicMock()
        idx._client = mock_client
        with patch("core.embedding_search._HAS_GENAI", True), \
             patch("core.embedding_search._HAS_NUMPY", True):
            count = await idx.build_index()
        assert count == 0


class TestEmbeddingIndexSearch:
    @pytest.mark.asyncio
    async def test_search_returns_ranked_results(self, tmp_path):
        """Search should return results sorted by similarity."""
        import numpy as np
        mem_dir = _make_memory_dir(tmp_path)
        idx = EmbeddingIndex(
            memory_dir=str(mem_dir),
            cache_path=str(tmp_path / "cache.json"),
            api_key="test-key",
        )

        # Build index with mock
        fake_embeddings = [
            MagicMock(values=list(np.array([1.0, 0.0, 0.0] + [0.0] * 765, dtype=float))),
            MagicMock(values=list(np.array([0.0, 1.0, 0.0] + [0.0] * 765, dtype=float))),
            MagicMock(values=list(np.array([0.9, 0.1, 0.0] + [0.0] * 765, dtype=float))),
        ]
        mock_result = MagicMock()
        mock_result.embeddings = fake_embeddings
        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)
        idx._client = mock_client
        with patch("core.embedding_search._HAS_GENAI", True), \
             patch("core.embedding_search._HAS_NUMPY", True):
            await idx.build_index()

        # Search with query embedding close to chunk 0
        query_emb = [MagicMock(values=list(np.array([1.0, 0.0, 0.0] + [0.0] * 765, dtype=float)))]
        query_result = MagicMock()
        query_result.embeddings = query_emb
        mock_client.aio.models.embed_content = AsyncMock(return_value=query_result)

        results = await idx.search("拉麵", top_k=3)
        assert len(results) >= 1
        # First result should be most similar (chunk 0 or chunk 2)
        assert results[0]["score"] > results[-1]["score"]

    @pytest.mark.asyncio
    async def test_search_empty_index(self, tmp_path):
        """Search on empty index returns empty list."""
        idx = EmbeddingIndex(
            memory_dir=str(tmp_path),
            cache_path=str(tmp_path / "cache.json"),
            api_key="test-key",
        )
        results = await idx.search("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_api_error(self, tmp_path):
        """If search embed API fails, it should raise."""
        import numpy as np
        mem_dir = _make_memory_dir(tmp_path)
        idx = EmbeddingIndex(
            memory_dir=str(mem_dir),
            cache_path=str(tmp_path / "cache.json"),
            api_key="test-key",
        )

        # Build successfully
        fake_embeddings = [
            MagicMock(values=list(np.random.rand(768).astype(float)))
            for _ in range(3)
        ]
        mock_result = MagicMock()
        mock_result.embeddings = fake_embeddings
        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)
        idx._client = mock_client
        with patch("core.embedding_search._HAS_GENAI", True), \
             patch("core.embedding_search._HAS_NUMPY", True):
            await idx.build_index()

        # Now make search fail
        mock_client.aio.models.embed_content = AsyncMock(side_effect=Exception("API error"))
        with pytest.raises(Exception, match="API error"):
            await idx.search("test")


# ── HybridSearch ────────────────────────────────────────────────


class TestHybridSearch:
    def _make_bm25(self):
        mock = MagicMock()
        mock.build_index.return_value = 5
        mock.search.return_value = [
            {"text": "Ted喜歡吃拉麵", "source": "a.md", "score": 3.0},
            {"text": "今天天氣很好", "source": "a.md", "score": 1.5},
        ]
        return mock

    def _make_embedding(self):
        mock = AsyncMock()
        mock.build_index.return_value = 5
        mock.search.return_value = [
            {"text": "Ted喜歡吃拉麵", "source": "a.md", "score": 0.9},
            {"text": "明天要開會", "source": "a.md", "score": 0.7},
        ]
        return mock

    @pytest.mark.asyncio
    async def test_merge_bm25_and_embedding(self):
        """Results from both engines should be merged."""
        hs = HybridSearch(bm25=self._make_bm25(), embedding=self._make_embedding())
        results = await hs.search("拉麵", top_k=6)
        texts = [r["text"] for r in results]
        assert "Ted喜歡吃拉麵" in texts
        assert "明天要開會" in texts
        assert "今天天氣很好" in texts

    @pytest.mark.asyncio
    async def test_dedup_same_text(self):
        """Same text from both engines should appear only once."""
        hs = HybridSearch(bm25=self._make_bm25(), embedding=self._make_embedding())
        results = await hs.search("拉麵", top_k=6)
        texts = [r["text"] for r in results]
        assert texts.count("Ted喜歡吃拉麵") == 1

    @pytest.mark.asyncio
    async def test_boost_both_engines(self):
        """Items found by both engines get a +0.1 bonus."""
        hs = HybridSearch(bm25=self._make_bm25(), embedding=self._make_embedding())
        results = await hs.search("拉麵", top_k=6)
        # "Ted喜歡吃拉麵" found by both → should be first
        assert results[0]["text"] == "Ted喜歡吃拉麵"

    @pytest.mark.asyncio
    async def test_embedding_unavailable_pure_bm25(self):
        """When embedding is None, return pure BM25 results."""
        bm25 = self._make_bm25()
        hs = HybridSearch(bm25=bm25, embedding=None)
        results = await hs.search("拉麵", top_k=3)
        assert len(results) == 2
        assert results[0]["text"] == "Ted喜歡吃拉麵"

    @pytest.mark.asyncio
    async def test_embedding_error_fallback_bm25(self):
        """When embedding search raises, fall back to BM25."""
        bm25 = self._make_bm25()
        embedding = AsyncMock()
        embedding.search.side_effect = Exception("network error")
        hs = HybridSearch(bm25=bm25, embedding=embedding)
        results = await hs.search("拉麵", top_k=3)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_build_index_calls_both(self):
        """build_index should call both BM25 and embedding."""
        bm25 = self._make_bm25()
        embedding = self._make_embedding()
        hs = HybridSearch(bm25=bm25, embedding=embedding)
        count = await hs.build_index()
        bm25.build_index.assert_called_once()
        embedding.build_index.assert_called_once()
        assert count == 5

    def test_search_sync_uses_bm25(self):
        """search_sync should use BM25 directly."""
        bm25 = self._make_bm25()
        hs = HybridSearch(bm25=bm25, embedding=None)
        results = hs.search_sync("拉麵", top_k=3)
        assert len(results) == 2
        bm25.search.assert_called_once_with("拉麵", top_k=3)

    @pytest.mark.asyncio
    async def test_normalize_correctness(self):
        """Normalized scores should be in [0, 1]."""
        results = [
            {"text": "a", "source": "x", "score": 5.0},
            {"text": "b", "source": "x", "score": 1.0},
            {"text": "c", "source": "x", "score": 3.0},
        ]
        normed = HybridSearch._normalize(results)
        scores = [r["score"] for r in normed]
        assert max(scores) == 1.0
        assert min(scores) == 0.0
        # Middle value should be (3-1)/(5-1) = 0.5
        assert abs(scores[2] - 0.5) < 1e-6

    @pytest.mark.asyncio
    async def test_normalize_all_same_score(self):
        """When all scores are equal, normalize to 1.0."""
        results = [
            {"text": "a", "source": "x", "score": 3.0},
            {"text": "b", "source": "x", "score": 3.0},
        ]
        normed = HybridSearch._normalize(results)
        assert all(r["score"] == 1.0 for r in normed)

    @pytest.mark.asyncio
    async def test_normalize_empty(self):
        assert HybridSearch._normalize([]) == []


# ── Patch T: Temporal Decay ──────────────────────────────────────


class TestTemporalDecay:
    def test_extract_date_daily(self):
        """Should parse YYYY-MM-DD from daily memory path."""
        dt = HybridSearch._extract_date_from_source(
            "C:/ted/JARVIS/memory/daily/2026-01-15.md"
        )
        assert dt == datetime(2026, 1, 15)

    def test_extract_date_no_date(self):
        """Non-dated files return None."""
        dt = HybridSearch._extract_date_from_source("memory/MEMORY.md")
        assert dt is None

    def test_extract_date_soul_growth(self):
        """SOUL_GROWTH has no date."""
        dt = HybridSearch._extract_date_from_source(
            "memory/clawra/SOUL_GROWTH.md"
        )
        assert dt is None

    def test_decay_recent_memory_no_change(self):
        """Today's memory should have decay factor ~1.0."""
        today = datetime.now().strftime("%Y-%m-%d")
        hs = HybridSearch(bm25=MagicMock(), embedding=None)
        results = [{"text": "test", "source": f"daily/{today}.md", "score": 1.0}]
        decayed = hs._apply_temporal_decay(results)
        assert decayed[0]["score"] > 0.99

    def test_decay_old_memory_reduced(self):
        """30-day-old memory should be ~74% of original score."""
        old_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        hs = HybridSearch(bm25=MagicMock(), embedding=None)
        results = [{"text": "test", "source": f"daily/{old_date}.md", "score": 1.0}]
        decayed = hs._apply_temporal_decay(results)
        expected = math.exp(-0.01 * 30)  # ~0.74
        assert abs(decayed[0]["score"] - expected) < 0.01

    def test_decay_very_old_memory(self):
        """180-day-old memory should be ~16% of original score."""
        old_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        hs = HybridSearch(bm25=MagicMock(), embedding=None)
        results = [{"text": "test", "source": f"daily/{old_date}.md", "score": 1.0}]
        decayed = hs._apply_temporal_decay(results)
        expected = math.exp(-0.01 * 180)  # ~0.165
        assert abs(decayed[0]["score"] - expected) < 0.02

    def test_no_decay_for_undated_files(self):
        """Files without date should not be decayed."""
        hs = HybridSearch(bm25=MagicMock(), embedding=None)
        results = [{"text": "important", "source": "MEMORY.md", "score": 0.8}]
        decayed = hs._apply_temporal_decay(results)
        assert decayed[0]["score"] == 0.8

    def test_decay_reorders_results(self):
        """Old high-score should drop below recent lower-score."""
        old_date = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        hs = HybridSearch(bm25=MagicMock(), embedding=None)
        results = [
            {"text": "old high", "source": f"daily/{old_date}.md", "score": 1.0},
            {"text": "new low", "source": f"daily/{today}.md", "score": 0.5},
        ]
        decayed = hs._apply_temporal_decay(results)
        # New low (0.5 * ~1.0 = 0.5) > old high (1.0 * ~0.135 = 0.135)
        assert decayed[0]["text"] == "new low"


# ── Patch T: MMR Re-ranking ──────────────────────────────────────


class TestMMR:
    def test_text_similarity_identical(self):
        assert HybridSearch._text_similarity("hello world", "hello world") == 1.0

    def test_text_similarity_different(self):
        sim = HybridSearch._text_similarity("apple", "banana")
        assert sim < 0.5

    def test_text_similarity_similar(self):
        sim = HybridSearch._text_similarity(
            "今天去了漢江邊散步，風景很好",
            "今天去了漢江邊散步，風景非常好",
        )
        assert sim > 0.7

    def test_mmr_removes_duplicates(self):
        """Near-identical texts should be deduplicated."""
        hs = HybridSearch(bm25=MagicMock(), embedding=None)
        results = [
            {"text": "今天去漢江邊散步風景很好心情也好", "source": "a.md", "score": 0.9},
            {"text": "今天去漢江邊散步風景很好心情真好", "source": "b.md", "score": 0.8},
            {"text": "明天要開會準備PPT", "source": "c.md", "score": 0.7},
        ]
        selected = hs._apply_mmr(results, top_k=3)
        assert len(selected) == 2  # second is dup of first
        texts = [r["text"] for r in selected]
        assert "明天要開會準備PPT" in texts

    def test_mmr_keeps_distinct_items(self):
        """Distinct items should all be kept."""
        hs = HybridSearch(bm25=MagicMock(), embedding=None)
        results = [
            {"text": "Ted喜歡吃拉麵", "source": "a.md", "score": 0.9},
            {"text": "明天要開會", "source": "b.md", "score": 0.8},
            {"text": "首爾今天下雪了", "source": "c.md", "score": 0.7},
        ]
        selected = hs._apply_mmr(results, top_k=3)
        assert len(selected) == 3

    def test_mmr_respects_top_k(self):
        """Should stop at top_k items."""
        hs = HybridSearch(bm25=MagicMock(), embedding=None)
        # Use truly distinct texts to avoid similarity-based dedup
        distinct_texts = [
            "Ted喜歡吃拉麵尤其是味噌口味",
            "明天下午三點要去看醫生",
            "首爾今天下了一場大雪氣溫零下",
            "最近在學習彈吉他進度不錯",
            "週末想去爬山陽明山好久沒去了",
            "新買的咖啡機很好用每天早上都喝",
            "公司下季度的專案計畫要開始了",
            "昨天看了一部很好看的電影推薦",
            "健身房的月費下個月要漲價了",
            "家裡的貓最近食慾變差要去看獸醫",
        ]
        results = [
            {"text": distinct_texts[i], "source": "a.md", "score": 1.0 - i * 0.1}
            for i in range(10)
        ]
        selected = hs._apply_mmr(results, top_k=3)
        assert len(selected) == 3

    def test_mmr_empty(self):
        hs = HybridSearch(bm25=MagicMock(), embedding=None)
        assert hs._apply_mmr([], top_k=6) == []

    def test_mmr_single_item(self):
        hs = HybridSearch(bm25=MagicMock(), embedding=None)
        results = [{"text": "only one", "source": "a.md", "score": 1.0}]
        assert hs._apply_mmr(results, top_k=6) == results

    @pytest.mark.asyncio
    async def test_full_pipeline_with_decay_and_mmr(self):
        """Integration test: search() applies both decay and MMR."""
        old_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")

        bm25 = MagicMock()
        bm25.search.return_value = [
            {"text": "老的拉麵記錄味道很好推薦這家", "source": f"daily/{old_date}.md", "score": 5.0},
            {"text": "老的拉麵記錄味道很好特別推薦", "source": f"daily/{old_date}.md", "score": 4.5},
            {"text": "今天天氣很好適合出門走走散步", "source": f"daily/{today}.md", "score": 2.0},
        ]

        embedding = AsyncMock()
        embedding.search.return_value = [
            {"text": "老的拉麵記錄味道很好推薦這家", "source": f"daily/{old_date}.md", "score": 0.9},
            {"text": "新的拉麵店開了要去試看看的", "source": f"daily/{today}.md", "score": 0.8},
        ]

        hs = HybridSearch(bm25=bm25, embedding=embedding)
        results = await hs.search("拉麵", top_k=3)

        # Should have deduped the near-identical old records
        # and the today record should rank well thanks to no decay
        assert len(results) >= 1
        # All scores should be non-negative (min-max normalization maps lowest to 0)
        assert all(r["score"] >= 0 for r in results)
