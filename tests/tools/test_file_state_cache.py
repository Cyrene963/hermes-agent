#!/usr/bin/env python3
"""
Tests for FileStateCache integration in file_tools.

Verifies that:
1. FileStateCache caches content after reads
2. Dedup hits return cached content instead of stubs
3. Cache is invalidated after writes/patches
4. LRU eviction and size limits work correctly

Run with:  python -m pytest tests/tools/test_file_state_cache.py -v
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from tools.file_state_cache import FileStateCache, FileState, file_state_cache


# ---------------------------------------------------------------------------
# Unit tests for FileStateCache itself
# ---------------------------------------------------------------------------

class TestFileStateCacheUnit(unittest.TestCase):
    """Test the FileStateCache class in isolation."""

    def setUp(self):
        self.cache = FileStateCache(max_entries=5, max_size_bytes=1024)

    def test_set_and_get(self):
        """Basic set/get round-trip."""
        self.cache.set("/tmp/test.txt", "hello world", mtime=1.0)
        state = self.cache.get("/tmp/test.txt")
        self.assertIsNotNone(state)
        self.assertEqual(state.content, "hello world")
        self.assertEqual(state.mtime, 1.0)

    def test_path_normalization(self):
        """Paths are normalized for consistent lookups."""
        self.cache.set("/tmp/../tmp/test.txt", "content", mtime=1.0)
        state = self.cache.get("/tmp/test.txt")
        self.assertIsNotNone(state)
        self.assertEqual(state.content, "content")

    def test_miss_returns_none(self):
        """Cache miss returns None."""
        self.assertIsNone(self.cache.get("/nonexistent"))

    def test_lru_eviction_by_count(self):
        """LRU eviction when max_entries exceeded."""
        for i in range(6):
            self.cache.set(f"/tmp/f{i}.txt", f"content{i}", mtime=float(i))
        # Should have evicted the oldest entry
        self.assertEqual(self.cache.size, 5)
        self.assertIsNone(self.cache.get("/tmp/f0.txt"))
        self.assertIsNotNone(self.cache.get("/tmp/f5.txt"))

    def test_lru_eviction_by_size(self):
        """LRU eviction when max_size_bytes exceeded."""
        # Each entry is ~500 bytes, limit is 1024
        self.cache.set("/tmp/a.txt", "x" * 500, mtime=1.0)
        self.cache.set("/tmp/b.txt", "x" * 500, mtime=2.0)
        self.cache.set("/tmp/c.txt", "x" * 500, mtime=3.0)
        # a.txt should be evicted to make room
        self.assertIsNone(self.cache.get("/tmp/a.txt"))
        self.assertIsNotNone(self.cache.get("/tmp/c.txt"))

    def test_overwrite_existing_entry(self):
        """Overwriting an existing path updates the entry."""
        self.cache.set("/tmp/test.txt", "old", mtime=1.0)
        self.cache.set("/tmp/test.txt", "new", mtime=2.0)
        state = self.cache.get("/tmp/test.txt")
        self.assertEqual(state.content, "new")
        self.assertEqual(state.mtime, 2.0)
        self.assertEqual(self.cache.size, 1)

    def test_delete(self):
        """Delete removes an entry."""
        self.cache.set("/tmp/test.txt", "content", mtime=1.0)
        self.assertTrue(self.cache.delete("/tmp/test.txt"))
        self.assertIsNone(self.cache.get("/tmp/test.txt"))
        self.assertFalse(self.cache.delete("/tmp/test.txt"))

    def test_clear(self):
        """Clear removes all entries."""
        self.cache.set("/tmp/a.txt", "a", mtime=1.0)
        self.cache.set("/tmp/b.txt", "b", mtime=2.0)
        self.cache.clear()
        self.assertEqual(self.cache.size, 0)

    def test_stats(self):
        """Stats report accurate information."""
        self.cache.set("/tmp/a.txt", "a", mtime=1.0)
        self.cache.get("/tmp/a.txt")  # hit
        self.cache.get("/tmp/b.txt")  # miss
        stats = self.cache.stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_invalidate_path(self):
        """invalidate_path is an alias for delete."""
        self.cache.set("/tmp/test.txt", "content", mtime=1.0)
        self.cache.invalidate_path("/tmp/test.txt")
        self.assertIsNone(self.cache.get("/tmp/test.txt"))

    def test_snapshot(self):
        """Snapshot returns a copy of the cache."""
        self.cache.set("/tmp/a.txt", "a", mtime=1.0)
        snap = self.cache.snapshot()
        self.assertIn("/tmp/a.txt", snap)
        self.assertEqual(snap["/tmp/a.txt"].content, "a")

    def test_keys(self):
        """Keys returns all cached paths."""
        self.cache.set("/tmp/a.txt", "a", mtime=1.0)
        self.cache.set("/tmp/b.txt", "b", mtime=2.0)
        keys = self.cache.keys()
        self.assertEqual(len(keys), 2)


# ---------------------------------------------------------------------------
# Integration tests with file_tools
# ---------------------------------------------------------------------------

class TestFileStateCacheIntegration(unittest.TestCase):
    """Test FileStateCache integration with read_file_tool."""

    def setUp(self):
        """Clear global caches before each test."""
        from tools.file_tools import _read_tracker
        _read_tracker.clear()
        file_state_cache.clear()

    def test_read_caches_content(self):
        """read_file_tool should cache content in FileStateCache."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello world\n")
            f.flush()
            path = f.name

        try:
            from tools.file_tools import read_file_tool
            # First read — should cache
            result = read_file_tool(path)
            data = json.loads(result)
            self.assertIn("content", data)

            # Check cache
            cached = file_state_cache.get(path)
            self.assertIsNotNone(cached)
            self.assertIn("hello world", cached.content)
        finally:
            os.unlink(path)

    def test_dedup_hit_returns_cached_content(self):
        """Second read of unchanged file returns cached content, not stub."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content\n")
            f.flush()
            path = f.name

        try:
            from tools.file_tools import read_file_tool
            # First read — populates cache
            result1 = read_file_tool(path)
            data1 = json.loads(result1)
            self.assertIn("content", data1)

            # Second read — should return cached content
            result2 = read_file_tool(path)
            data2 = json.loads(result2)
            self.assertEqual(data2.get("status"), "cached")
            self.assertTrue(data2.get("content_returned"))
            self.assertIn("test content", data2.get("content", ""))
        finally:
            os.unlink(path)

    def test_write_invalidates_cache(self):
        """Writing a file should invalidate the FileStateCache entry."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("original\n")
            f.flush()
            path = f.name

        try:
            from tools.file_tools import read_file_tool, write_file_tool

            # Read to populate cache
            read_file_tool(path)
            self.assertIsNotNone(file_state_cache.get(path))

            # Write — should invalidate cache
            write_file_tool(path, "modified\n")
            self.assertIsNone(file_state_cache.get(path))
        finally:
            os.unlink(path)

    def test_cache_stats_in_response(self):
        """Cached responses include cache stats."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("data\n")
            f.flush()
            path = f.name

        try:
            from tools.file_tools import read_file_tool
            # First read
            read_file_tool(path)

            # Second read — should have cache stats
            result = read_file_tool(path)
            data = json.loads(result)
            self.assertIn("_cache_stats", data)
            stats = data["_cache_stats"]
            self.assertIn("hits", stats)
            self.assertIn("hit_rate", stats)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Global singleton test
# ---------------------------------------------------------------------------

class TestGlobalSingleton(unittest.TestCase):
    """Verify the global file_state_cache singleton works."""

    def test_singleton_is_file_state_cache(self):
        self.assertIsInstance(file_state_cache, FileStateCache)

    def test_singleton_persists_across_calls(self):
        file_state_cache.set("/tmp/glob.txt", "global", mtime=1.0)
        state = file_state_cache.get("/tmp/glob.txt")
        self.assertIsNotNone(state)
        self.assertEqual(state.content, "global")
        file_state_cache.delete("/tmp/glob.txt")


if __name__ == "__main__":
    unittest.main()
