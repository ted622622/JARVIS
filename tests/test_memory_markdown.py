"""Tests for MarkdownMemory, MemorySearch, and session transcripts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from memory.markdown_memory import MarkdownMemory


class TestMarkdownMemoryInit:
    def test_creates_directories(self, tmp_path):
        mm = MarkdownMemory(str(tmp_path / "mem"))
        assert (tmp_path / "mem" / "daily").exists()
        assert (tmp_path / "mem" / "sessions").exists()
        assert (tmp_path / "mem" / "MEMORY.md").exists()


class TestMarkdownMemoryRemember:
    def test_remember_adds_to_existing_category(self, tmp_path):
        mm = MarkdownMemory(str(tmp_path))
        (tmp_path / "MEMORY.md").write_text(
            "# 長期記憶\n\n## 用戶偏好\n\n- 喜歡咖啡\n",
            encoding="utf-8",
        )
        mm.remember("喜歡拉麵", category="用戶偏好")
        content = mm.read_memory()
        assert "喜歡拉麵" in content
        assert "喜歡咖啡" in content

    def test_remember_creates_new_category(self, tmp_path):
        mm = MarkdownMemory(str(tmp_path))
        mm.remember("決定用方案 A", category="重要決策")
        content = mm.read_memory()
        assert "## 重要決策" in content
        assert "決定用方案 A" in content

    def test_read_memory(self, tmp_path):
        mm = MarkdownMemory(str(tmp_path))
        content = mm.read_memory()
        assert "長期記憶" in content


class TestMarkdownMemoryDaily:
    def test_log_daily_creates_file(self, tmp_path):
        mm = MarkdownMemory(str(tmp_path))
        mm.log_daily("今天幫用戶訂了 UberEats")
        today = datetime.now().strftime("%Y-%m-%d")
        path = tmp_path / "daily" / f"{today}.md"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "UberEats" in content

    def test_log_daily_appends(self, tmp_path):
        mm = MarkdownMemory(str(tmp_path))
        mm.log_daily("第一筆")
        mm.log_daily("第二筆")
        today = datetime.now().strftime("%Y-%m-%d")
        content = (tmp_path / "daily" / f"{today}.md").read_text(encoding="utf-8")
        assert "第一筆" in content
        assert "第二筆" in content

    def test_read_daily(self, tmp_path):
        mm = MarkdownMemory(str(tmp_path))
        assert mm.read_daily() == ""
        mm.log_daily("test")
        assert mm.read_daily() != ""


class TestMarkdownMemorySessions:
    def test_save_session(self, tmp_path):
        mm = MarkdownMemory(str(tmp_path))
        path = mm.save_session(
            "ubereats-order",
            "# 訂餐\n\n**Ted**: 幫我訂雞排\n**Clawra**: 好～",
        )
        assert path.exists()
        assert "ubereats-order" in path.name
        content = path.read_text(encoding="utf-8")
        assert "雞排" in content

    def test_save_session_sanitizes_slug(self, tmp_path):
        mm = MarkdownMemory(str(tmp_path))
        path = mm.save_session("test/bad:chars!", "content")
        assert "/" not in path.name
        assert ":" not in path.name

    def test_list_sessions(self, tmp_path):
        mm = MarkdownMemory(str(tmp_path))
        mm.save_session("first", "a")
        mm.save_session("second", "b")
        sessions = mm.list_sessions()
        assert len(sessions) == 2

    def test_all_markdown_files(self, tmp_path):
        mm = MarkdownMemory(str(tmp_path))
        mm.log_daily("entry")
        mm.save_session("test", "content")
        files = mm.all_markdown_files()
        assert len(files) >= 3  # MEMORY.md + daily + session


class TestMemorySearch:
    def test_build_index(self, tmp_path):
        from core.memory_search import MemorySearch

        (tmp_path / "MEMORY.md").write_text(
            "# 記憶\n\n用戶喜歡吃拉麵\n\n用戶住在板橋",
            encoding="utf-8",
        )
        search = MemorySearch(str(tmp_path))
        count = search.build_index()
        assert count >= 2

    def test_search_finds_match(self, tmp_path):
        from core.memory_search import MemorySearch

        (tmp_path / "MEMORY.md").write_text(
            "# 記憶\n\n用戶喜歡吃拉麵\n\n用戶住在板橋\n\n用戶是量化交易員",
            encoding="utf-8",
        )
        search = MemorySearch(str(tmp_path))
        search.build_index()
        results = search.search("拉麵")
        assert len(results) > 0
        assert any("拉麵" in r["text"] for r in results)

    def test_search_empty_index(self, tmp_path):
        from core.memory_search import MemorySearch

        search = MemorySearch(str(tmp_path))
        results = search.search("test")
        assert results == []

    def test_search_no_match(self, tmp_path):
        from core.memory_search import MemorySearch

        (tmp_path / "test.md").write_text("完全無關的內容", encoding="utf-8")
        search = MemorySearch(str(tmp_path))
        search.build_index()
        results = search.search("xyznonexistent")
        assert results == []
