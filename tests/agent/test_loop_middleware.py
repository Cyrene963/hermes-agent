#!/usr/bin/env python3
"""
Tests for loop middleware: compliance check extraction.

Run with:  python -m pytest tests/agent/test_loop_middleware.py -v
"""

import unittest
from agent.loop_middleware import (
    LoopMiddleware,
    HookResult,
    tool_permission_hook,
)


class TestLoopMiddleware(unittest.TestCase):
    """Test the LoopMiddleware class."""

    def setUp(self):
        self.mw = LoopMiddleware()

    def test_register_pre(self):
        """Pre-hooks are registered and sorted by priority."""
        self.mw.register_pre("a", lambda t, a, c: HookResult(), priority=20)
        self.mw.register_pre("b", lambda t, a, c: HookResult(), priority=10)
        self.assertEqual(self.mw.pre_hook_names, ["b", "a"])  # sorted by priority

    def test_register_post(self):
        """Post-hooks are registered and sorted by priority."""
        self.mw.register_post("a", lambda t, r, c: None, priority=20)
        self.mw.register_post("b", lambda t, r, c: None, priority=10)
        self.assertEqual(self.mw.post_hook_names, ["b", "a"])

    def test_pre_hook_blocks(self):
        """Pre-hook can block tool execution."""
        def blocker(tool_name, args, context):
            return HookResult(blocked=True, error_message="blocked!")

        self.mw.register_pre("blocker", blocker)
        result = self.mw.run_pre("terminal", {"command": "ls"})
        self.assertTrue(result.blocked)
        self.assertEqual(result.error_message, "blocked!")

    def test_pre_hook_modifies_args(self):
        """Pre-hook can modify tool args."""
        def modifier(tool_name, args, context):
            args["command"] = args["command"] + " --color=never"
            return HookResult(modified_args=args)

        self.mw.register_pre("modifier", modifier)
        result = self.mw.run_pre("terminal", {"command": "ls"})
        self.assertEqual(result.modified_args["command"], "ls --color=never")

    def test_pre_hook_chain(self):
        """Multiple pre-hooks run in priority order."""
        calls = []
        def hook_a(tool_name, args, context):
            calls.append("a")
            return HookResult()
        def hook_b(tool_name, args, context):
            calls.append("b")
            return HookResult()

        self.mw.register_pre("a", hook_a, priority=20)
        self.mw.register_pre("b", hook_b, priority=10)
        self.mw.run_pre("test", {})
        self.assertEqual(calls, ["b", "a"])

    def test_pre_hook_stops_on_block(self):
        """Pre-hook chain stops when a hook blocks."""
        calls = []
        def hook_a(tool_name, args, context):
            calls.append("a")
            return HookResult(blocked=True, error_message="nope")
        def hook_b(tool_name, args, context):
            calls.append("b")
            return HookResult()

        self.mw.register_pre("a", hook_a, priority=10)
        self.mw.register_pre("b", hook_b, priority=20)
        self.mw.run_pre("test", {})
        self.assertEqual(calls, ["a"])  # b never runs

    def test_disabled_hook_skipped(self):
        """Disabled hooks are skipped."""
        calls = []
        def hook(tool_name, args, context):
            calls.append("called")
            return HookResult()

        self.mw.register_pre("hook", hook)
        self.mw.disable("hook")
        self.mw.run_pre("test", {})
        self.assertEqual(calls, [])

    def test_enable_hook(self):
        """Re-enabling a hook makes it run again."""
        calls = []
        def hook(tool_name, args, context):
            calls.append("called")
            return HookResult()

        self.mw.register_pre("hook", hook, enabled=False)
        self.mw.run_pre("test", {})
        self.assertEqual(calls, [])

        self.mw.enable("hook")
        self.mw.run_pre("test", {})
        self.assertEqual(calls, ["called"])

    def test_post_hook_runs(self):
        """Post-hooks run after tool execution."""
        calls = []
        def tracker(tool_name, result, context):
            calls.append((tool_name, result))

        self.mw.register_post("tracker", tracker)
        self.mw.run_post("terminal", "output")
        self.assertEqual(calls, [("terminal", "output")])

    def test_post_hook_exception_swallowed(self):
        """Post-hook exceptions are swallowed (non-fatal)."""
        def bad_hook(tool_name, result, context):
            raise ValueError("oops")

        self.mw.register_post("bad", bad_hook)
        # Should not raise
        self.mw.run_post("test", "result")

    def test_unregister(self):
        """Unregistering a hook removes it."""
        self.mw.register_pre("hook", lambda t, a, c: HookResult())
        self.assertTrue(self.mw.unregister("hook"))
        self.assertEqual(self.mw.pre_hook_names, [])

    def test_stats(self):
        """Stats reports hook counts."""
        self.mw.register_pre("a", lambda t, a, c: HookResult())
        self.mw.register_post("b", lambda t, r, c: None)
        stats = self.mw.stats()
        self.assertEqual(stats["pre_hooks"], 1)
        self.assertEqual(stats["post_hooks"], 1)


class TestToolPermissionHook(unittest.TestCase):
    """Test the built-in tool_permission_hook."""

    def test_safe_tool_passes(self):
        """SAFE tools pass through."""
        result = tool_permission_hook("read_file", {}, {})
        self.assertFalse(result.blocked)

    def test_harmful_command_blocked(self):
        """Harmful commands are blocked."""
        result = tool_permission_hook(
            "terminal", {"command": "rm -rf /"}, {},
        )
        self.assertTrue(result.blocked)

    def test_high_tool_metadata(self):
        """HIGH tools get needs_approval metadata in interactive mode."""
        result = tool_permission_hook(
            "terminal", {"command": "ls"}, {"interactive": True},
        )
        self.assertFalse(result.blocked)
        self.assertTrue(result.metadata.get("needs_approval"))

    def test_yolo_mode_passes(self):
        """YOLO mode auto-approves non-blocked tools."""
        result = tool_permission_hook(
            "terminal", {"command": "ls"}, {"yolo_mode": True},
        )
        self.assertFalse(result.blocked)
        self.assertFalse(result.metadata.get("needs_approval"))


if __name__ == "__main__":
    unittest.main()
