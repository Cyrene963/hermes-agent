#!/usr/bin/env python3
"""
Tests for per-tool prompt system.

Run with:  python -m pytest tests/agent/test_tool_prompts.py -v
"""

import unittest
from agent.tool_prompts import (
    get_tool_prompt,
    get_tool_prompts_for_available_tools,
    list_registered_tools,
    _REGISTRY,
)


class TestToolPrompts(unittest.TestCase):
    """Test the per-tool prompt registry."""

    def test_registered_tools(self):
        """All expected tools have registered prompts."""
        registered = set(list_registered_tools())
        expected = {
            "terminal", "read_file", "write_file", "patch",
            "search_files", "web_search", "web_extract",
            "memory", "session_search", "delegate_task",
            "execute_code", "skill_view", "skill_manage",
            "browser_navigate", "cronjob", "send_message",
        }
        self.assertTrue(expected.issubset(registered),
                        f"Missing: {expected - registered}")

    def test_get_tool_prompt_returns_string(self):
        """get_tool_prompt returns a non-empty string for registered tools."""
        for tool in ["terminal", "read_file", "write_file", "memory"]:
            prompt = get_tool_prompt(tool)
            self.assertIsNotNone(prompt, f"{tool} should have a prompt")
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 50, f"{tool} prompt too short")

    def test_get_tool_prompt_unregistered(self):
        """get_tool_prompt returns None for unregistered tools."""
        self.assertIsNone(get_tool_prompt("nonexistent_tool"))

    def test_terminal_prompt_contains_anti_patterns(self):
        """Terminal prompt includes anti-pattern warnings."""
        prompt = get_tool_prompt("terminal")
        self.assertIn("Anti-pattern", prompt)
        self.assertIn("read_file", prompt)  # should mention read_file as alternative

    def test_read_file_prompt_contains_usage(self):
        """Read file prompt includes usage instructions."""
        prompt = get_tool_prompt("read_file")
        self.assertIn("offset", prompt)
        self.assertIn("limit", prompt)

    def test_memory_prompt_contains_rules(self):
        """Memory prompt includes writing rules."""
        prompt = get_tool_prompt("memory")
        self.assertIn("declarative", prompt.lower())
        self.assertIn("User prefers", prompt)

    def test_get_prompts_for_available_tools(self):
        """Only returns prompts for available tools."""
        available = {"terminal", "read_file", "nonexistent"}
        prompts = get_tool_prompts_for_available_tools(available)
        self.assertEqual(len(prompts), 2)  # terminal + read_file
        for p in prompts:
            self.assertIsInstance(p, str)
            self.assertGreater(len(p), 50)

    def test_get_prompts_empty_set(self):
        """Empty tool set returns empty prompts."""
        prompts = get_tool_prompts_for_available_tools(set())
        self.assertEqual(prompts, [])

    def test_all_prompts_are_strings(self):
        """All registered prompts return strings."""
        for tool_name in list_registered_tools():
            prompt = get_tool_prompt(tool_name)
            if prompt is not None:
                self.assertIsInstance(prompt, str, f"{tool_name} prompt is not a string")


if __name__ == "__main__":
    unittest.main()
