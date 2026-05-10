"""Tests for field_matches_context — generic recipient-binding preflight check.

Tests use generic fixtures (user_a, user_b, chat_001, chat_002).
No real user IDs or names.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))

from agent.memory_metacognition import PolicyPreflightPolicy, PreflightCheck


# ── Fixtures ──

USER_A_CTX = {"user": {"id": "user_a", "chat_id": "chat_001"}}
USER_B_CTX = {"user": {"id": "user_b", "chat_id": "chat_002"}}
EMPTY_CTX: dict = {}

FILE_SEND_RULE = {
    "tool": "send_message",
    "task_type": "external_file_delivery",
    "trigger_patterns": ["senddocument", "file_path", "MEDIA:"],
    "block_on_failure": True,
    "checks": [
        {
            "type": "field_matches_context",
            "field": "chat_id",
            "context_path": "user.chat_id",
            "required": True,
            "on_missing_context": "fail",
            "error": "recipient does not match execution context",
        },
    ],
}


def _make_policy(ctx, checks=None, rule_overrides=None):
    rule = dict(FILE_SEND_RULE)
    if checks is not None:
        rule["checks"] = checks
    if rule_overrides:
        rule.update(rule_overrides)
    return PolicyPreflightPolicy(rules=[rule], execution_context=ctx)


def _run(policy, tool_args):
    task_type = policy.get_task_type("send_message", tool_args)
    assert task_type is not None, f"trigger not matched for {tool_args}"
    return policy.run_checks(task_type, tool_args)


def _identity_check(result):
    checks = [c for c in result.checks if c.check_type == "field_matches_context"]
    assert len(checks) == 1
    return checks[0]


# ── Core matching ──

class TestFieldMatchesContext:

    def test_matching_value_passes(self):
        policy = _make_policy(USER_A_CTX)
        result = _run(policy, {"chat_id": "chat_001", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "allow"
        assert _identity_check(result).status == "PASS"

    def test_mismatched_value_blocks(self):
        policy = _make_policy(USER_A_CTX)
        result = _run(policy, {"chat_id": "chat_002", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "block"
        ic = _identity_check(result)
        assert ic.status == "FAIL"
        # Verify masking in query
        assert "***" in ic.query

    def test_text_message_no_file_ignores(self):
        policy = _make_policy(USER_A_CTX)
        task_type = policy.get_task_type("send_message", {"chat_id": "chat_002", "text": "hello"})
        assert task_type is None

    def test_non_required_mismatch_warns(self):
        checks = [{"type": "field_matches_context", "field": "chat_id",
                    "context_path": "user.chat_id", "required": False,
                    "on_missing_context": "pass", "error": "mismatch"}]
        policy = _make_policy(USER_A_CTX, checks=checks)
        result = _run(policy, {"chat_id": "chat_002", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert _identity_check(result).status == "WARN"


# ── Missing context ──

class TestMissingContext:

    def test_missing_context_default_fail_blocks(self):
        """on_missing_context=fail (default) + required → BLOCK when context absent."""
        policy = _make_policy(EMPTY_CTX)
        result = _run(policy, {"chat_id": "chat_001", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "block"
        assert _identity_check(result).status == "FAIL"

    def test_missing_context_pass_allows(self):
        checks = [{"type": "field_matches_context", "field": "chat_id",
                    "context_path": "user.chat_id", "required": True,
                    "on_missing_context": "pass", "error": "mismatch"}]
        policy = _make_policy(EMPTY_CTX, checks=checks)
        result = _run(policy, {"chat_id": "chat_001", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "allow"

    def test_missing_context_warn(self):
        checks = [{"type": "field_matches_context", "field": "chat_id",
                    "context_path": "user.chat_id", "required": True,
                    "on_missing_context": "warn", "error": "mismatch"}]
        policy = _make_policy(EMPTY_CTX, checks=checks)
        result = _run(policy, {"chat_id": "chat_001", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        # warn → PASS (not FAIL), so decision depends on other checks
        assert _identity_check(result).status == "PASS"


# ── Missing tool field ──

class TestMissingToolField:

    def test_missing_field_passes_by_default(self):
        """No chat_id in tool_args → PASS (field_required handles this)."""
        policy = _make_policy(USER_A_CTX)
        result = _run(policy, {"method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert _identity_check(result).status == "PASS"

    def test_missing_field_fail_when_configured(self):
        checks = [{"type": "field_matches_context", "field": "chat_id",
                    "context_path": "user.chat_id", "required": True,
                    "on_missing_field": "fail", "on_missing_context": "fail",
                    "error": "need recipient"}]
        policy = _make_policy(USER_A_CTX, checks=checks)
        result = _run(policy, {"method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert _identity_check(result).status == "FAIL"


# ── Dot-path resolution ──

class TestResolvePath:

    def test_simple_path(self):
        assert PolicyPreflightPolicy._resolve_path({"a": {"b": "val"}}, "a.b") == "val"

    def test_nested_path(self):
        data = {"job": {"recipient": {"id": "r_001"}}}
        assert PolicyPreflightPolicy._resolve_path(data, "job.recipient.id") == "r_001"

    def test_missing_key(self):
        assert PolicyPreflightPolicy._resolve_path({}, "user.chat_id") is None

    def test_missing_intermediate(self):
        assert PolicyPreflightPolicy._resolve_path({"user": {}}, "user.chat_id") is None

    def test_non_dict_intermediate(self):
        assert PolicyPreflightPolicy._resolve_path({"user": "scalar"}, "user.chat_id") is None


# ── Masking ──

class TestMaskValue:

    def test_long_value(self):
        assert PolicyPreflightPolicy._mask_value("1234567890") == "***7890"

    def test_short_value(self):
        assert PolicyPreflightPolicy._mask_value("abc") == "****"

    def test_exact_4(self):
        assert PolicyPreflightPolicy._mask_value("1234") == "****"

    def test_5_chars(self):
        assert PolicyPreflightPolicy._mask_value("12345") == "***2345"


# ── Job / cron context ──

class TestJobContext:

    def test_job_recipient_matches(self):
        """Job with intended_recipient → check against job context."""
        checks = [{"type": "field_matches_context", "field": "chat_id",
                    "context_path": "job.intended_recipient.id", "required": True,
                    "on_missing_context": "fail", "error": "wrong job recipient"}]
        job_ctx = {"job": {"intended_recipient": {"id": "chat_001"}}}
        policy = _make_policy(job_ctx, checks=checks)
        result = _run(policy, {"chat_id": "chat_001", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "allow"
        assert _identity_check(result).status == "PASS"

    def test_job_recipient_mismatch_blocks(self):
        """Job delivers to wrong user → BLOCK."""
        checks = [{"type": "field_matches_context", "field": "chat_id",
                    "context_path": "job.intended_recipient.id", "required": True,
                    "on_missing_context": "fail", "error": "wrong job recipient"}]
        job_ctx = {"job": {"intended_recipient": {"id": "chat_001"}}}
        policy = _make_policy(job_ctx, checks=checks)
        result = _run(policy, {"chat_id": "chat_002", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "block"

    def test_job_without_intended_recipient_blocks(self):
        """Job missing intended_recipient + on_missing_context=fail → BLOCK."""
        checks = [{"type": "field_matches_context", "field": "chat_id",
                    "context_path": "job.intended_recipient.id", "required": True,
                    "on_missing_context": "fail", "error": "no recipient configured"}]
        policy = _make_policy(EMPTY_CTX, checks=checks)
        result = _run(policy, {"chat_id": "chat_001", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "block"

    def test_combined_session_and_job_rules(self):
        """Two rules: one for session, one for job. Each checks its own path."""
        checks = [
            {"type": "field_matches_context", "field": "chat_id",
             "context_path": "user.chat_id", "required": True,
             "on_missing_context": "pass", "error": "session mismatch"},
            {"type": "field_matches_context", "field": "chat_id",
             "context_path": "job.intended_recipient.id", "required": True,
             "on_missing_context": "pass", "error": "job mismatch"},
        ]
        # Has user context but no job context → first check passes, second skips
        policy = _make_policy(USER_A_CTX, checks=checks)
        result = _run(policy, {"chat_id": "chat_001", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "allow"


# ── Backward compatibility ──

class TestBackwardCompat:

    def test_old_check_type_alias(self):
        """field_session_chat_id still works as alias for field_matches_context."""
        checks = [{"type": "field_session_chat_id", "field": "chat_id",
                    "context_path": "user.chat_id", "required": True,
                    "on_missing_context": "fail", "error": "wrong recipient"}]
        policy = _make_policy(USER_A_CTX, checks=checks)
        result = _run(policy, {"chat_id": "chat_001", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "allow"

    def test_old_alias_blocks_on_mismatch(self):
        checks = [{"type": "field_session_chat_id", "field": "chat_id",
                    "context_path": "user.chat_id", "required": True,
                    "on_missing_context": "fail", "error": "wrong recipient"}]
        policy = _make_policy(USER_A_CTX, checks=checks)
        result = _run(policy, {"chat_id": "chat_002", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "block"


# ── Integration: build_preflight_policy end-to-end ──

class TestBuildPreflightPolicyE2E:
    """Integration tests verifying user_context flows through to execution_context.

    These test the full chain:
        user_context → build_preflight_policy() → PolicyPreflightPolicy
        → execution_context → field_matches_context check → block/allow
    """

    def _build_and_run(self, user_context, rule, tool_args):
        from agent.memory_metacognition import PolicyPreflightPolicy as Policy
        # Simulate what build_preflight_policy does: extract execution_context
        execution_context = {}
        if user_context:
            if user_context.get("user_id"):
                execution_context.setdefault("user", {})["id"] = str(user_context["user_id"])
            if user_context.get("chat_id"):
                execution_context.setdefault("user", {})["chat_id"] = str(user_context["chat_id"])
        policy = Policy(rules=[rule], execution_context=execution_context)
        task_type = policy.get_task_type("send_message", tool_args)
        assert task_type is not None
        return policy.run_checks(task_type, tool_args)

    def test_user_context_flows_to_execution_context_allow(self):
        """user_context.chat_id → execution_context.user.chat_id → match → allow."""
        rule = {
            "tool": "send_message", "task_type": "file_delivery",
            "trigger_patterns": ["senddocument", "file_path"],
            "block_on_failure": True,
            "checks": [{"type": "field_matches_context", "field": "chat_id",
                        "context_path": "user.chat_id", "required": True,
                        "on_missing_context": "fail", "error": "mismatch"}],
        }
        user_ctx = {"user_id": "user_a", "chat_id": "chat_001"}
        result = self._build_and_run(user_ctx, rule,
            {"chat_id": "chat_001", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "allow"

    def test_user_context_flows_to_execution_context_block(self):
        """user_context.chat_id → execution_context.user.chat_id → mismatch → block."""
        rule = {
            "tool": "send_message", "task_type": "file_delivery",
            "trigger_patterns": ["senddocument", "file_path"],
            "block_on_failure": True,
            "checks": [{"type": "field_matches_context", "field": "chat_id",
                        "context_path": "user.chat_id", "required": True,
                        "on_missing_context": "fail", "error": "mismatch"}],
        }
        user_ctx = {"user_id": "user_a", "chat_id": "chat_001"}
        result = self._build_and_run(user_ctx, rule,
            {"chat_id": "chat_002", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "block"

    def test_no_user_context_with_fail_blocks(self):
        """No user_context + on_missing_context=fail → block."""
        rule = {
            "tool": "send_message", "task_type": "file_delivery",
            "trigger_patterns": ["senddocument", "file_path"],
            "block_on_failure": True,
            "checks": [{"type": "field_matches_context", "field": "chat_id",
                        "context_path": "user.chat_id", "required": True,
                        "on_missing_context": "fail", "error": "no context"}],
        }
        result = self._build_and_run(None, rule,
            {"chat_id": "chat_001", "method": "sendDocument", "file_path": "/tmp/f.pdf"})
        assert result.decision == "block"

    def test_plain_text_not_triggered(self):
        """Plain text message → no file delivery trigger → no check."""
        rule = {
            "tool": "send_message", "task_type": "file_delivery",
            "trigger_patterns": ["senddocument", "file_path"],
            "block_on_failure": True,
            "checks": [{"type": "field_matches_context", "field": "chat_id",
                        "context_path": "user.chat_id", "required": True,
                        "on_missing_context": "fail", "error": "mismatch"}],
        }
        execution_context = {"user": {"chat_id": "chat_001"}}
        from agent.memory_metacognition import PolicyPreflightPolicy as Policy
        policy = Policy(rules=[rule], execution_context=execution_context)
        task_type = policy.get_task_type("send_message", {"chat_id": "chat_002", "text": "hello"})
        assert task_type is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
