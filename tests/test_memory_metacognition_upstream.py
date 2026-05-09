#!/usr/bin/env python3
"""
Upstream Framework Tests for Memory Metacognition.
Tests generic interfaces, default no-op behavior, policy loading, fallbacks.
NO user-specific data (no user IDs, no platform-specific rules, no language-specific mappings).

Suitable for PR to NousResearch/hermes-agent.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Repo root: tests/ -> hermes-agent/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agent"))

# Disable local policy to test pure default behavior
os.environ["HERMES_MEMORY_POLICY_DISABLE_LOCAL"] = "1"

from agent.memory_metacognition import (
    MemoryIndexProvider,
    RecallQueryExpander,
    MemoryPreflightPolicy,
    PreflightResult,
    PreflightCheck,
    NoOpIndexProvider,
    PassthroughExpander,
    NoOpPreflightPolicy,
    PolicyQueryExpander,
    PolicyPreflightPolicy,
    ScriptIndexProvider,
    load_policy,
    build_index_provider,
    build_query_expander,
    build_preflight_policy,
    _deep_merge,
)


class TestInterfaces(unittest.TestCase):
    """Test that abstract interfaces are properly defined."""

    def test_memory_index_provider_is_abstract(self):
        self.assertTrue(hasattr(MemoryIndexProvider, 'build_index'))

    def test_recall_query_expander_is_abstract(self):
        self.assertTrue(hasattr(RecallQueryExpander, 'expand'))

    def test_memory_preflight_policy_is_abstract(self):
        self.assertTrue(hasattr(MemoryPreflightPolicy, 'get_task_type'))
        self.assertTrue(hasattr(MemoryPreflightPolicy, 'run_checks'))

    def test_preflight_result_dataclass(self):
        r = PreflightResult(decision="allow", reason="test", task_type="x")
        self.assertEqual(r.decision, "allow")
        self.assertEqual(r.checks, [])

    def test_preflight_check_dataclass(self):
        c = PreflightCheck(
            check_type="identity", query="q", required=True,
            found=False, status="FAIL", message="err"
        )
        self.assertEqual(c.status, "FAIL")


class TestDefaultNoOp(unittest.TestCase):
    """Test that default implementations are no-op."""

    def test_noop_index_returns_empty(self):
        provider = NoOpIndexProvider()
        self.assertEqual(provider.build_index(), "")

    def test_passthrough_expander_returns_original(self):
        expander = PassthroughExpander()
        result = expander.expand("hello world")
        self.assertEqual(result, ["hello world"])

    def test_passthrough_expander_empty_message(self):
        expander = PassthroughExpander()
        self.assertEqual(expander.expand(""), [])
        self.assertEqual(expander.expand(None), [])

    def test_noop_preflight_allows_all(self):
        policy = NoOpPreflightPolicy()
        self.assertIsNone(policy.get_task_type("send_message", {}))
        self.assertIsNone(policy.get_task_type("terminal", {"command": "rm -rf /"}))
        result = policy.run_checks("anything", {})
        self.assertEqual(result.decision, "allow")


class TestPolicyQueryExpander(unittest.TestCase):
    """Test PolicyQueryExpander with generic test data."""

    def test_expands_with_mappings(self):
        expansions = {"hello": ["greeting", "hi"]}
        expander = PolicyQueryExpander(expansions=expansions)
        result = expander.expand("say hello")
        self.assertIn("say hello", result)
        self.assertIn("greeting", result)
        self.assertIn("hi", result)

    def test_respects_max_queries(self):
        expansions = {f"k{i}": [f"v{i}"] for i in range(20)}
        expander = PolicyQueryExpander(expansions=expansions, max_queries=3)
        result = expander.expand("k0 k1 k2 k3 k4 k5 k6 k7 k8 k9")
        self.assertLessEqual(len(result), 3)

    def test_empty_expansions_passthrough(self):
        expander = PolicyQueryExpander(expansions={})
        self.assertEqual(expander.expand("test"), ["test"])

    def test_case_insensitive_matching(self):
        expansions = {"Hello": ["world"]}
        expander = PolicyQueryExpander(expansions=expansions)
        result = expander.expand("HELLO there")
        self.assertIn("world", result)

    def test_deduplication(self):
        expansions = {"test": ["test", "testing"]}
        expander = PolicyQueryExpander(expansions=expansions)
        result = expander.expand("test")
        # "test" appears once (original), "testing" added
        self.assertEqual(result.count("test"), 1)


class TestPolicyPreflightPolicy(unittest.TestCase):
    """Test PolicyPreflightPolicy with generic test data."""

    def test_no_rules_allows_all(self):
        policy = PolicyPreflightPolicy(rules=[])
        self.assertIsNone(policy.get_task_type("any_tool", {}))

    def test_matching_tool_returns_task_type(self):
        rules = [{"tool": "my_tool", "task_type": "my_task"}]
        policy = PolicyPreflightPolicy(rules=rules)
        self.assertEqual(policy.get_task_type("my_tool", {}), "my_task")

    def test_nonmatching_tool_returns_none(self):
        rules = [{"tool": "my_tool", "task_type": "my_task"}]
        policy = PolicyPreflightPolicy(rules=rules)
        self.assertIsNone(policy.get_task_type("other_tool", {}))

    def test_trigger_patterns_activate(self):
        rules = [{
            "tool": "terminal",
            "task_type": "dangerous",
            "trigger_patterns": ["rm -rf", "drop table"],
        }]
        policy = PolicyPreflightPolicy(rules=rules)
        self.assertEqual(
            policy.get_task_type("terminal", {"command": "rm -rf /tmp"}),
            "dangerous"
        )
        self.assertIsNone(
            policy.get_task_type("terminal", {"command": "ls -la"})
        )

    def test_run_checks_no_rule_allows(self):
        policy = PolicyPreflightPolicy(rules=[])
        result = policy.run_checks("unknown_task", {})
        self.assertEqual(result.decision, "allow")


class TestPolicyLoader(unittest.TestCase):
    """Test YAML policy loading and merging."""

    def test_load_policy_returns_dict(self):
        policy = load_policy(force_reload=True)
        self.assertIsInstance(policy, dict)

    def test_deep_merge_overlay_wins(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        overlay = {"b": {"c": 99}, "e": 5}
        result = _deep_merge(base, overlay)
        self.assertEqual(result["b"]["c"], 99)
        self.assertEqual(result["b"]["d"], 3)
        self.assertEqual(result["e"], 5)

    def test_build_expander_returns_object(self):
        expander = build_query_expander()
        self.assertIsInstance(expander, RecallQueryExpander)

    def test_build_preflight_returns_object(self):
        policy = build_preflight_policy()
        self.assertIsInstance(policy, MemoryPreflightPolicy)

    def test_build_index_returns_object(self):
        provider = build_index_provider()
        self.assertIsInstance(provider, MemoryIndexProvider)

    def test_default_policy_is_noop(self):
        """Without local policy, defaults should be no-op."""
        expander = build_query_expander()
        result = expander.expand("any message")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 1)


class TestIntegrationPoints(unittest.TestCase):
    """Test that prompt_builder integration points work."""

    def test_expand_recall_queries_callable(self):
        from agent.prompt_builder import expand_recall_queries
        result = expand_recall_queries("test message")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 1)

    def test_expand_recall_queries_empty(self):
        from agent.prompt_builder import expand_recall_queries
        self.assertEqual(expand_recall_queries(""), [])
        self.assertEqual(expand_recall_queries(None), [])

    def test_build_memory_index_block_callable(self):
        from agent.prompt_builder import build_memory_index_block
        result = build_memory_index_block()
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
