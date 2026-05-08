#!/usr/bin/env python3
"""
Tests for session memory compaction.

Run with:  python -m pytest tests/agent/test_session_memory_compact.py -v
"""

import unittest
from unittest.mock import MagicMock
from agent.session_memory_compact import (
    extract_memory_before_compact,
    reinject_memory_after_compact,
    _find_compaction_summary,
    MEMORY_AUTHORITY_MARKER,
)


class TestExtractMemoryBeforeCompact(unittest.TestCase):
    """Test memory extraction before compression."""

    def test_extracts_from_memory_store(self):
        """Extracts memory and user blocks from memory store."""
        store = MagicMock()
        store.format_for_system_prompt.side_effect = lambda t: {
            "memory": "MEMORY CONTENT",
            "user": "USER CONTENT",
        }.get(t, "")

        snapshot = extract_memory_before_compact(memory_store=store)
        self.assertEqual(snapshot["memory"], "MEMORY CONTENT")
        self.assertEqual(snapshot["user"], "USER CONTENT")

    def test_extracts_from_memory_manager(self):
        """Extracts external memory from memory manager."""
        manager = MagicMock()
        manager.build_system_prompt.return_value = "EXTERNAL MEMORY"

        snapshot = extract_memory_before_compact(memory_manager=manager)
        self.assertEqual(snapshot["external"], "EXTERNAL MEMORY")

    def test_handles_none_gracefully(self):
        """None inputs return empty snapshot."""
        snapshot = extract_memory_before_compact()
        self.assertEqual(snapshot["memory"], "")
        self.assertEqual(snapshot["user"], "")
        self.assertEqual(snapshot["external"], "")

    def test_handles_exception_gracefully(self):
        """Exceptions in memory store are swallowed."""
        store = MagicMock()
        store.format_for_system_prompt.side_effect = RuntimeError("oops")

        snapshot = extract_memory_before_compact(memory_store=store)
        self.assertEqual(snapshot["memory"], "")


class TestFindCompactionSummary(unittest.TestCase):
    """Test finding the compaction summary message."""

    def test_finds_summary_by_prefix(self):
        """Finds message containing SUMMARY_PREFIX."""
        from agent.context_compressor import SUMMARY_PREFIX
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": SUMMARY_PREFIX + " summary text"},
        ]
        idx = _find_compaction_summary(messages)
        self.assertEqual(idx, 1)

    def test_finds_summary_by_legacy_prefix(self):
        """Finds message containing LEGACY_SUMMARY_PREFIX."""
        from agent.context_compressor import LEGACY_SUMMARY_PREFIX
        messages = [
            {"role": "user", "content": LEGACY_SUMMARY_PREFIX + " old summary"},
        ]
        idx = _find_compaction_summary(messages)
        self.assertEqual(idx, 0)

    def test_returns_none_when_no_summary(self):
        """Returns None when no compaction summary found."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        idx = _find_compaction_summary(messages)
        self.assertIsNone(idx)

    def test_handles_list_content(self):
        """Finds summary in list-format content."""
        from agent.context_compressor import SUMMARY_PREFIX
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SUMMARY_PREFIX + " summary"},
                ],
            },
        ]
        idx = _find_compaction_summary(messages)
        self.assertEqual(idx, 0)


class TestReinjectMemoryAfterCompact(unittest.TestCase):
    """Test memory re-injection after compression."""

    def test_reinjects_memory_into_summary(self):
        """Memory is prepended to the compaction summary."""
        from agent.context_compressor import SUMMARY_PREFIX
        messages = [
            {"role": "user", "content": SUMMARY_PREFIX + " summary text"},
        ]
        snapshot = {"memory": "MEMORY RULES", "user": "USER PROFILE", "external": ""}

        result = reinject_memory_after_compact(messages, snapshot)
        content = result[0]["content"]
        self.assertIn("MEMORY RULES", content)
        self.assertIn("USER PROFILE", content)
        self.assertIn(MEMORY_AUTHORITY_MARKER, content)

    def test_noop_when_no_snapshot(self):
        """Empty snapshot does nothing."""
        messages = [{"role": "user", "content": "hello"}]
        result = reinject_memory_after_compact(messages, {})
        self.assertEqual(result[0]["content"], "hello")

    def test_noop_when_no_summary(self):
        """No compaction summary means no re-injection."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        snapshot = {"memory": "RULES", "user": "", "external": ""}
        result = reinject_memory_after_compact(messages, snapshot)
        # Memory should not be injected since there's no compaction summary
        self.assertEqual(result[0]["content"], "hello")

    def test_authority_marker_present(self):
        """Re-injected memory includes the authority marker."""
        from agent.context_compressor import SUMMARY_PREFIX
        messages = [
            {"role": "user", "content": SUMMARY_PREFIX + " summary"},
        ]
        snapshot = {"memory": "RULES", "user": "", "external": ""}

        result = reinject_memory_after_compact(messages, snapshot)
        content = result[0]["content"]
        self.assertIn("ACTIVE MEMORY", content)
        self.assertIn("MANDATORY", content)
        self.assertIn("OVERRIDE", content)


if __name__ == "__main__":
    unittest.main()
