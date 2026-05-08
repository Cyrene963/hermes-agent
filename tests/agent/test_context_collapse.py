#!/usr/bin/env python3
"""
Tests for context collapse: fold long tool outputs.

Run with:  python -m pytest tests/agent/test_context_collapse.py -v
"""

import unittest
from agent.context_collapse import (
    collapse_tool_content,
    collapse_messages,
    _COLLAPSE_TEMPLATE,
)


class TestCollapseToolContent(unittest.TestCase):
    """Test the collapse_tool_content function."""

    def test_short_content_unchanged(self):
        """Content under max_chars is returned unchanged."""
        content = "short content"
        result, was_collapsed = collapse_tool_content(content, max_chars=100)
        self.assertEqual(result, content)
        self.assertFalse(was_collapsed)

    def test_long_content_collapsed(self):
        """Content over max_chars is collapsed."""
        content = "A" * 5000
        result, was_collapsed = collapse_tool_content(
            content, max_chars=2000, head_chars=500, tail_chars=500,
        )
        self.assertTrue(was_collapsed)
        self.assertIn("A" * 500, result)  # head preserved
        self.assertIn("... [", result)  # collapse marker
        self.assertLess(len(result), len(content))

    def test_preserves_head_and_tail(self):
        """Head and tail content are preserved in collapsed output."""
        head = "=== FILE HEADER ===\n"
        tail = "\n=== END OF FILE ==="
        middle = "x" * 5000
        content = head + middle + tail

        result, _ = collapse_tool_content(
            content, max_chars=2000, head_chars=len(head), tail_chars=len(tail),
        )
        self.assertIn(head, result)
        self.assertIn(tail, result)

    def test_exact_max_chars_unchanged(self):
        """Content exactly at max_chars is not collapsed."""
        content = "x" * 2000
        result, was_collapsed = collapse_tool_content(content, max_chars=2000)
        self.assertFalse(was_collapsed)

    def test_empty_content(self):
        """Empty content returns empty."""
        result, was_collapsed = collapse_tool_content("", max_chars=100)
        self.assertEqual(result, "")
        self.assertFalse(was_collapsed)

    def test_none_content(self):
        """None content returns None."""
        result, was_collapsed = collapse_tool_content(None, max_chars=100)
        self.assertIsNone(result)
        self.assertFalse(was_collapsed)


class TestCollapseMessages(unittest.TestCase):
    """Test the collapse_messages function."""

    def _make_tool_message(self, content, name="terminal"):
        return {"role": "tool", "content": content, "name": name}

    def test_collapses_long_results(self):
        """Long tool results are collapsed."""
        messages = [
            {"role": "user", "content": "read file"},
            {"role": "assistant", "content": "reading..."},
            self._make_tool_message("x" * 5000),
        ]

        result, stats = collapse_messages(messages, max_chars=2000)
        self.assertEqual(stats["collapsed_count"], 1)
        self.assertLess(len(result[2]["content"]), 5000)
        self.assertIn("... [", result[2]["content"])

    def test_leaves_short_results(self):
        """Short tool results are not collapsed."""
        messages = [
            self._make_tool_message("short output"),
        ]

        result, stats = collapse_messages(messages, max_chars=2000)
        self.assertEqual(stats["collapsed_count"], 0)
        self.assertEqual(result[0]["content"], "short output")

    def test_mixed_messages(self):
        """Handles mix of message types."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            self._make_tool_message("x" * 5000),
            {"role": "assistant", "content": "got it"},
            self._make_tool_message("short"),
            self._make_tool_message("y" * 3000),
        ]

        result, stats = collapse_messages(messages, max_chars=2000)
        self.assertEqual(stats["collapsed_count"], 2)  # x*5000 and y*3000
        self.assertEqual(stats["total_checked"], 3)

    def test_anthropic_format(self):
        """Handles Anthropic format (tool_result blocks in user messages)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "z" * 5000,
                        "tool_use_id": "tu_1",
                    },
                ],
            },
        ]

        result, stats = collapse_messages(messages, max_chars=2000)
        self.assertEqual(stats["collapsed_count"], 1)

    def test_chars_saved(self):
        """Stats correctly report chars saved."""
        content = "a" * 5000
        messages = [self._make_tool_message(content)]

        result, stats = collapse_messages(messages, max_chars=2000)
        self.assertGreater(stats["chars_saved"], 0)

    def test_custom_head_tail(self):
        """Custom head/tail chars are respected."""
        content = "A" * 100 + "B" * 5000 + "C" * 100
        messages = [self._make_tool_message(content)]

        result, stats = collapse_messages(
            messages, max_chars=2000, head_chars=100, tail_chars=100,
        )
        collapsed = result[0]["content"]
        self.assertIn("A" * 100, collapsed)
        self.assertIn("C" * 100, collapsed)


if __name__ == "__main__":
    unittest.main()
