#!/usr/bin/env python3
"""
Tests for post-compact cleanup and time-based micro-compact.

Run with:  python -m pytest tests/agent/test_post_compact_cleanup.py -v
"""

import time
import unittest
from unittest.mock import MagicMock

from agent.post_compact_cleanup import (
    run_post_compact_cleanup,
    should_time_based_compact,
    apply_time_based_compact,
    TimeBasedMCConfig,
)


class TestPostCompactCleanup(unittest.TestCase):
    """Test run_post_compact_cleanup."""

    def test_clears_file_state_cache(self):
        """FileStateCache is cleared after compaction."""
        cache = MagicMock()
        cache.size = 42

        stats = run_post_compact_cleanup(file_state_cache=cache)
        cache.clear.assert_called_once()
        self.assertTrue(stats["file_cache_cleared"])

    def test_clears_read_tracker(self):
        """Read tracker dict is cleared after compaction."""
        tracker = {"key1": "val1", "key2": "val2"}

        stats = run_post_compact_cleanup(read_tracker=tracker)
        self.assertEqual(len(tracker), 0)
        self.assertTrue(stats["read_tracker_cleared"])

    def test_handles_none_gracefully(self):
        """None inputs don't cause errors."""
        stats = run_post_compact_cleanup()
        self.assertFalse(stats["file_cache_cleared"])
        self.assertFalse(stats["read_tracker_cleared"])

    def test_handles_exception_gracefully(self):
        """Exceptions in cache clear are swallowed."""
        cache = MagicMock()
        cache.clear.side_effect = RuntimeError("oops")

        stats = run_post_compact_cleanup(file_state_cache=cache)
        self.assertFalse(stats["file_cache_cleared"])


class TestTimeBasedMCConfig(unittest.TestCase):
    """Test TimeBasedMCConfig defaults."""

    def test_default_config(self):
        config = TimeBasedMCConfig()
        self.assertEqual(config.gap_threshold_seconds, 300)
        self.assertTrue(config.enabled)

    def test_custom_config(self):
        config = TimeBasedMCConfig(gap_threshold_seconds=60, enabled=False)
        self.assertEqual(config.gap_threshold_seconds, 60)
        self.assertFalse(config.enabled)


class TestShouldTimeBasedCompact(unittest.TestCase):
    """Test should_time_based_compact."""

    def test_no_assistant_messages(self):
        """No assistant messages means no trigger."""
        messages = [{"role": "user", "content": "hello"}]
        self.assertFalse(should_time_based_compact(messages))

    def test_recent_messages_no_trigger(self):
        """Recent assistant messages don't trigger."""
        messages = [
            {"role": "assistant", "content": "hi", "timestamp": time.time()},
        ]
        self.assertFalse(should_time_based_compact(messages))

    def test_old_messages_trigger(self):
        """Old assistant messages trigger compaction."""
        messages = [
            {"role": "assistant", "content": "hi", "timestamp": time.time() - 600},
        ]
        config = TimeBasedMCConfig(gap_threshold_seconds=300)
        self.assertTrue(should_time_based_compact(messages, config))

    def test_disabled_config(self):
        """Disabled config never triggers."""
        messages = [
            {"role": "assistant", "content": "hi", "timestamp": time.time() - 9999},
        ]
        config = TimeBasedMCConfig(enabled=False)
        self.assertFalse(should_time_based_compact(messages, config))


class TestApplyTimeBasedCompact(unittest.TestCase):
    """Test apply_time_based_compact."""

    def test_clears_old_tool_results(self):
        """Old tool results before last assistant message are cleared."""
        messages = [
            {"role": "tool", "content": "x" * 500},
            {"role": "assistant", "content": "response", "timestamp": time.time() - 600},
            {"role": "tool", "content": "y" * 500},  # after assistant, should be kept
        ]
        config = TimeBasedMCConfig(gap_threshold_seconds=300)

        result, stats = apply_time_based_compact(messages, config)
        self.assertTrue(stats["triggered"])
        self.assertEqual(stats["cleared_count"], 1)
        self.assertIn("cleared", result[0]["content"])

    def test_keeps_short_results(self):
        """Short tool results are not cleared."""
        messages = [
            {"role": "tool", "content": "short"},
            {"role": "assistant", "content": "response", "timestamp": time.time() - 600},
        ]
        config = TimeBasedMCConfig(gap_threshold_seconds=300)

        result, stats = apply_time_based_compact(messages, config)
        self.assertEqual(stats["cleared_count"], 0)

    def test_disabled_noop(self):
        """Disabled config does nothing."""
        messages = [
            {"role": "tool", "content": "x" * 500},
            {"role": "assistant", "content": "hi", "timestamp": time.time() - 9999},
        ]
        config = TimeBasedMCConfig(enabled=False)

        result, stats = apply_time_based_compact(messages, config)
        self.assertFalse(stats["triggered"])


if __name__ == "__main__":
    unittest.main()
