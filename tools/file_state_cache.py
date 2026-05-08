"""File state cache with LRU eviction and size limits.

Inspired by Claude Code's FileStateCache — caches actual file content
to avoid redundant disk I/O and support context compression strategies.

Unlike the per-task ``_read_tracker`` in ``file_tools.py`` (which only
stores mtime for dedup stubs), this cache stores the actual content so
it can be re-injected after context compression.

Design:
  - LRU eviction by entry count (default 100 entries)
  - Byte-size cap (default 25 MB) to prevent memory bloat
  - Path normalization for consistent cache hits
  - Thread-safe for concurrent subagent access
  - Integration hooks for compression strategies (snapshot, merge)

Usage:
    from tools.file_state_cache import file_state_cache

    # After a successful read:
    file_state_cache.set(path, content, mtime=current_mtime)

    # Before a disk read:
    cached = file_state_cache.get(path)
    if cached and cached.mtime == current_mtime:
        return cached.content  # skip disk I/O
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional


# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_MAX_ENTRIES = 100
DEFAULT_MAX_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


class FileState:
    """Cached state for a single file."""
    __slots__ = ('content', 'mtime', 'timestamp', 'size_bytes')

    def __init__(
        self,
        content: str,
        mtime: float,
        timestamp: float,
        size_bytes: int,
    ) -> None:
        self.content = content
        self.mtime = mtime
        self.timestamp = timestamp
        self.size_bytes = size_bytes

    def __repr__(self) -> str:
        return (
            f"FileState(mtime={self.mtime}, size={self.size_bytes}, "
            f"content_len={len(self.content)})"
        )


class FileStateCache:
    """LRU file content cache with entry-count and byte-size limits.

    Thread-safe.  All path keys are normalized (resolve, normalize
    separators) before access to ensure consistent hits regardless of
    whether callers pass relative vs absolute paths.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        self._max_entries = max_entries
        self._max_size_bytes = max_size_bytes
        self._cache: OrderedDict[str, FileState] = OrderedDict()
        self._current_size: int = 0
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    # ── Path normalization ────────────────────────────────────────────
    @staticmethod
    def _normalize(key: str) -> str:
        """Normalize path for consistent cache lookups."""
        try:
            return str(Path(key).resolve())
        except (OSError, ValueError):
            return os.path.normpath(key)

    # ── Core operations ───────────────────────────────────────────────
    def get(self, path: str) -> Optional[FileState]:
        """Get cached state for a path, or None on miss."""
        key = self._normalize(path)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return entry
            self._misses += 1
            return None

    def set(self, path: str, content: str, mtime: float) -> None:
        """Cache file content.  Evicts LRU entries if needed."""
        key = self._normalize(path)
        size_bytes = len(content.encode('utf-8', errors='replace'))

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                old = self._cache.pop(key)
                self._current_size -= old.size_bytes

            # Evict LRU entries until we fit
            while (
                self._cache
                and (
                    len(self._cache) >= self._max_entries
                    or self._current_size + size_bytes > self._max_size_bytes
                )
            ):
                _evicted_key, evicted = self._cache.popitem(last=False)
                self._current_size -= evicted.size_bytes

            # Insert new entry
            entry = FileState(
                content=content,
                mtime=mtime,
                timestamp=time.time(),
                size_bytes=size_bytes,
            )
            self._cache[key] = entry
            self._current_size += size_bytes

    def has(self, path: str) -> bool:
        """Check if path is in cache (does not check mtime)."""
        key = self._normalize(path)
        with self._lock:
            return key in self._cache

    def delete(self, path: str) -> bool:
        """Remove a cached entry.  Returns True if it existed."""
        key = self._normalize(path)
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry is not None:
                self._current_size -= entry.size_bytes
                return True
            return False

    def invalidate_path(self, path: str) -> None:
        """Invalidate cache for a path (alias for delete, ignores return)."""
        self.delete(path)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
            self._hits = 0
            self._misses = 0

    # ── Stats ─────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        """Number of cached entries."""
        with self._lock:
            return len(self._cache)

    @property
    def current_size_bytes(self) -> int:
        """Current total size of cached content in bytes."""
        with self._lock:
            return self._current_size

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def max_size_bytes(self) -> int:
        return self._max_size_bytes

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "size_bytes": self._current_size,
                "max_size_bytes": self._max_size_bytes,
                "size_mb": round(self._current_size / (1024 * 1024), 2),
                "max_size_mb": round(self._max_size_bytes / (1024 * 1024), 2),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (
                    round(self._hits / total, 3) if total > 0 else 0.0
                ),
            }

    # ── Snapshot / merge (for compression strategies) ─────────────────
    def snapshot(self) -> Dict[str, FileState]:
        """Return a shallow copy of the cache contents.

        Used by compression strategies to capture file state before
        context collapse.
        """
        with self._lock:
            return dict(self._cache)

    def keys(self) -> list:
        """Return all cached path keys."""
        with self._lock:
            return list(self._cache.keys())

    def merge_from(self, other: 'FileStateCache') -> int:
        """Merge entries from another cache, preferring newer timestamps.

        Returns the number of entries merged.
        """
        merged = 0
        other_snapshot = other.snapshot()
        for key, entry in other_snapshot.items():
            existing = self.get(key.replace(os.sep, '/'))
            if existing is None or entry.timestamp > existing.timestamp:
                self.set(key, entry.content, entry.mtime)
                merged += 1
        return merged


# ── Global singleton ──────────────────────────────────────────────────
file_state_cache = FileStateCache()
