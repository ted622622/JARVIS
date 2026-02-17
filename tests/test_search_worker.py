"""Tests for SearchWorker — consolidated DuckDuckGo search.

Run: pytest tests/test_search_worker.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workers.search_worker import SearchWorker


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def worker():
    return SearchWorker()


# ── Basic properties ─────────────────────────────────────────────


class TestBasic:
    def test_name(self, worker):
        assert worker.name == "search"

    @pytest.mark.asyncio
    async def test_close(self, worker):
        await worker.close()  # Should not raise


# ── Execute interface ────────────────────────────────────────────


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_delegates_to_search(self, worker):
        with patch.object(worker, "search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = {"result": "test", "source": "search"}
            result = await worker.execute("台北天氣")
            mock_search.assert_awaited_once_with("台北天氣", max_results=5)

    @pytest.mark.asyncio
    async def test_execute_passes_max_results(self, worker):
        with patch.object(worker, "search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = {"result": "test", "source": "search"}
            await worker.execute("test", max_results=3)
            mock_search.assert_awaited_once_with("test", max_results=3)


# ── Search method ─────────────────────────────────────────────────


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_dict(self, worker):
        with patch.object(worker, "_ddg_instant", new_callable=AsyncMock, return_value=None), \
             patch.object(worker, "_ddg_search", new_callable=AsyncMock, return_value=[]):
            result = await worker.search("test query")
        assert isinstance(result, dict)
        assert "result" in result
        assert result["source"] == "search"
        assert result["worker"] == "search"

    @pytest.mark.asyncio
    async def test_search_includes_instant_answer(self, worker):
        instant = {"abstract": "Test answer", "url": "https://example.com", "source_name": "Wikipedia"}
        with patch.object(worker, "_ddg_instant", new_callable=AsyncMock, return_value=instant), \
             patch.object(worker, "_ddg_search", new_callable=AsyncMock, return_value=[]):
            result = await worker.search("test")
        assert "Test answer" in result["result"]
        assert result["instant"] == instant

    @pytest.mark.asyncio
    async def test_search_includes_html_results(self, worker):
        results = [
            {"title": "Result 1", "snippet": "Snippet 1", "url": "https://example.com/1"},
            {"title": "Result 2", "snippet": "Snippet 2", "url": "https://example.com/2"},
        ]
        with patch.object(worker, "_ddg_instant", new_callable=AsyncMock, return_value=None), \
             patch.object(worker, "_ddg_search", new_callable=AsyncMock, return_value=results):
            result = await worker.search("test")
        assert "Result 1" in result["result"]
        assert "Result 2" in result["result"]
        assert result["results"] == results

    @pytest.mark.asyncio
    async def test_search_no_results(self, worker):
        with patch.object(worker, "_ddg_instant", new_callable=AsyncMock, return_value=None), \
             patch.object(worker, "_ddg_search", new_callable=AsyncMock, return_value=[]):
            result = await worker.search("xyzabc_no_result")
        assert "未找到" in result["result"]


# ── DuckDuckGo Instant Answer ────────────────────────────────────


class TestDDGInstant:
    @pytest.mark.asyncio
    async def test_instant_returns_none_on_empty(self, worker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"AbstractText": "", "Answer": ""}

        with patch("workers.search_worker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await worker._ddg_instant("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_instant_returns_abstract(self, worker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "AbstractText": "Python is a programming language",
            "AbstractURL": "https://en.wikipedia.org/wiki/Python",
            "AbstractSource": "Wikipedia",
            "Answer": "",
        }

        with patch("workers.search_worker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await worker._ddg_instant("python")
        assert result is not None
        assert "Python" in result["abstract"]
        assert result["url"] == "https://en.wikipedia.org/wiki/Python"

    @pytest.mark.asyncio
    async def test_instant_handles_error(self, worker):
        with patch("workers.search_worker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("network error")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await worker._ddg_instant("test")
        assert result is None


# ── HTML parsing ──────────────────────────────────────────────────


class TestParseHTML:
    def test_parse_basic_results(self, worker):
        html = '''
        <a class="result__a" href="https://example.com/1">Title One</a>
        <a class="result__snippet" href="#">Snippet one text</a>
        <a class="result__a" href="https://example.com/2">Title Two</a>
        <a class="result__snippet" href="#">Snippet two text</a>
        '''
        results = worker._parse_ddg_html(html, max_results=5)
        assert len(results) == 2
        assert results[0]["title"] == "Title One"
        assert results[0]["url"] == "https://example.com/1"
        assert "Snippet one" in results[0]["snippet"]

    def test_parse_respects_max_results(self, worker):
        html = '''
        <a class="result__a" href="https://a.com">A</a>
        <a class="result__a" href="https://b.com">B</a>
        <a class="result__a" href="https://c.com">C</a>
        '''
        results = worker._parse_ddg_html(html, max_results=2)
        assert len(results) == 2

    def test_parse_empty_html(self, worker):
        results = worker._parse_ddg_html("", max_results=5)
        assert results == []


# ── Fetch page ────────────────────────────────────────────────────


class TestFetchPage:
    @pytest.mark.asyncio
    async def test_fetch_page_success(self, worker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Hello World</p></body></html>"

        with patch("workers.search_worker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await worker.fetch_page("https://example.com")
        assert "Hello World" in result["result"]
        assert result["source"] == "search"

    @pytest.mark.asyncio
    async def test_fetch_page_http_error(self, worker):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("workers.search_worker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await worker.fetch_page("https://example.com/404")
        assert "error" in result
        assert "404" in result["error"]

    @pytest.mark.asyncio
    async def test_fetch_page_network_error(self, worker):
        with patch("workers.search_worker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await worker.fetch_page("https://example.com")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_fetch_page_truncates(self, worker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "x" * 10000

        with patch("workers.search_worker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await worker.fetch_page("https://example.com", max_chars=100)
        assert len(result["result"]) <= 100


# ── Strip HTML helper ─────────────────────────────────────────────


class TestStripHTML:
    def test_strips_tags(self):
        assert "Hello" in SearchWorker._strip_html("<b>Hello</b>")

    def test_strips_scripts(self):
        html = '<script>alert("xss")</script>Content'
        assert "alert" not in SearchWorker._strip_html(html)

    def test_strips_styles(self):
        html = "<style>body{color:red}</style>Content"
        assert "color" not in SearchWorker._strip_html(html)
        assert "Content" in SearchWorker._strip_html(html)


# ── ReactExecutor chain integration ──────────────────────────────


class TestReactExecutorChain:
    def test_web_search_chain_includes_search(self):
        from core.react_executor import FALLBACK_CHAINS
        assert "search" in FALLBACK_CHAINS["web_search"]
