#!/usr/bin/env python3
"""
Tests for micro-compact: lightweight per-turn tool result pruning.

Run with:  python -m pytest tests/agent/test_micro_compact.py -v
"""

import unittest
from agent.micro_compact import (
    microcompact_messages,
    estimate_context_tokens,
    CLEARED_PLACEHOLDER,
)


class TestMicrocompactMessages(unittest.TestCase):
    """Test the microcompact_messages function."""

    def _make_tool_message(self, content, name="terminal", tool_call_id="tc_1"):
        """Create an OpenAI-format tool message."""
        return {
            "role": "tool",
            "content": content,
            "name": name,
            "tool_call_id": tool_call_id,
        }

    def _make_user_message(self, text="hello"):
        return {"role": "user", "content": text}

    def _make_assistant_message(self, text="response"):
        return {"role": "assistant", "content": text}

    def test_no_compaction_when_few_results(self):
        """No compaction when tool results <= keep_recent."""
        messages = [
            self._make_user_message(),
            self._make_assistant_message(),
            self._make_tool_message("result 1"),
            self._make_assistant_message(),
            self._make_tool_message("result 2"),
        ]
        result, stats = microcompact_messages(messages, keep_recent=5)
        self.assertEqual(stats["compacted_count"], 0)
        self.assertEqual(stats["total_tool_results"], 2)

    def test_compacts_old_results(self):
        """Old tool results beyond keep_recent are compacted."""
        messages = []
        for i in range(8):
            messages.append(self._make_user_message(f"q{i}"))
            messages.append(self._make_assistant_message(f"a{i}"))
            messages.append(self._make_tool_message(f"output {'x' * 500} from tool {i}"))

        result, stats = microcompact_messages(messages, keep_recent=3, min_result_chars=100)
        # 8 tool results, keep 3 recent → compact 5
        self.assertEqual(stats["compacted_count"], 5)
        self.assertEqual(stats["total_tool_results"], 8)
        self.assertEqual(stats["kept_recent"], 3)

        # Verify: last 3 tool results should be intact
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        for tm in tool_msgs[-3:]:
            self.assertIn("output", tm["content"])
            self.assertNotIn("cleared", tm["content"])

        # Verify: first 5 tool results should be compacted
        for tm in tool_msgs[:5]:
            self.assertEqual(tm["content"], CLEARED_PLACEHOLDER)

    def test_skips_small_results(self):
        """Results shorter than min_result_chars are not compacted."""
        messages = []
        for i in range(8):
            messages.append(self._make_user_message(f"q{i}"))
            messages.append(self._make_assistant_message(f"a{i}"))
            messages.append(self._make_tool_message(f"short"))  # 5 chars

        result, stats = microcompact_messages(messages, keep_recent=3, min_result_chars=200)
        # All results are too short to compact
        self.assertEqual(stats["compacted_count"], 0)

    def test_chars_saved(self):
        """Stats correctly report chars saved."""
        big_content = "x" * 1000
        messages = []
        for i in range(6):
            messages.append(self._make_tool_message(big_content))

        result, stats = microcompact_messages(messages, keep_recent=2, min_result_chars=100)
        self.assertEqual(stats["compacted_count"], 4)
        # Each compacted result saves ~1000 - len(placeholder) chars
        expected_saved_per = len(big_content) - len(CLEARED_PLACEHOLDER)
        self.assertEqual(stats["chars_saved"], 4 * expected_saved_per)

    def test_mixed_message_types(self):
        """Handles mix of user, assistant, and tool messages."""
        messages = [
            self._make_user_message("what is python?"),
            self._make_assistant_message("Python is..."),
            self._make_tool_message("result " + "y" * 500, name="web_search"),
            self._make_assistant_message("Based on the search..."),
            self._make_user_message("tell me more"),
            self._make_assistant_message("Sure..."),
            self._make_tool_message("data " + "z" * 500, name="read_file"),
            self._make_assistant_message("Here's what I found..."),
            self._make_user_message("another question"),
            self._make_assistant_message("answer"),
            self._make_tool_message("output " + "w" * 500, name="terminal"),
        ]

        result, stats = microcompact_messages(messages, keep_recent=2, min_result_chars=100)
        # 3 tool results, keep 2 → compact 1
        self.assertEqual(stats["compacted_count"], 1)
        self.assertEqual(stats["total_tool_results"], 3)

    def test_empty_messages(self):
        """Empty message list returns empty stats."""
        result, stats = microcompact_messages([])
        self.assertEqual(stats["compacted_count"], 0)
        self.assertEqual(stats["total_tool_results"], 0)

    def test_keep_recent_zero(self):
        """keep_recent=0 compacts all results."""
        messages = [
            self._make_tool_message("a" * 500),
            self._make_tool_message("b" * 500),
            self._make_tool_message("c" * 500),
        ]
        result, stats = microcompact_messages(messages, keep_recent=0, min_result_chars=100)
        self.assertEqual(stats["compacted_count"], 3)

    def test_default_keep_recent(self):
        """Default keep_recent=5 works correctly."""
        messages = []
        for i in range(10):
            messages.append(self._make_tool_message(f"output {i} " + "x" * 300))

        result, stats = microcompact_messages(messages)
        # 10 results, keep 5 → compact 5
        self.assertEqual(stats["compacted_count"], 5)


class TestEstimateContextTokens(unittest.TestCase):
    """Test the estimate_context_tokens function."""

    def test_basic_estimate(self):
        """Rough token estimate for simple messages."""
        messages = [
            {"role": "user", "content": "hello world"},  # 11 chars
            {"role": "assistant", "content": "hi there"},  # 8 chars
        ]
        tokens = estimate_context_tokens(messages)
        # (11 + 8 + 100 overhead) / 4 ≈ 29
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, 100)

    def test_empty_messages(self):
        """Empty messages return 0."""
        self.assertEqual(estimate_context_tokens([]), 0)

    def test_multimodal_content(self):
        """Handles list content (multimodal)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello " * 100},
                ],
            },
        ]
        tokens = estimate_context_tokens(messages)
        self.assertGreater(tokens, 0)


if __name__ == "__main__":
    unittest.main()
