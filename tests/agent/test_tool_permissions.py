#!/usr/bin/env python3
"""
Tests for per-tool permission system.

Run with:  python -m pytest tests/agent/test_tool_permissions.py -v
"""

import unittest
from agent.tool_permissions import (
    check_tool_permission,
    get_tool_risk,
    list_tools_by_risk,
    ToolRisk,
    PermissionDecision,
)


class TestToolRiskClassification(unittest.TestCase):
    """Test tool risk level assignments."""

    def test_safe_tools(self):
        """Read-only tools are classified as SAFE."""
        safe_tools = ["read_file", "search_files", "web_search", "session_search",
                      "skills_list", "skill_view", "browser_snapshot"]
        for tool in safe_tools:
            self.assertEqual(get_tool_risk(tool), ToolRisk.SAFE, f"{tool} should be SAFE")

    def test_low_tools(self):
        """Minor side-effect tools are classified as LOW."""
        low_tools = ["memory", "hindsight_retain", "skill_manage", "send_message"]
        for tool in low_tools:
            self.assertEqual(get_tool_risk(tool), ToolRisk.LOW, f"{tool} should be LOW")

    def test_medium_tools(self):
        """Significant side-effect tools are classified as MEDIUM."""
        medium_tools = ["write_file", "patch", "browser_click", "browser_navigate"]
        for tool in medium_tools:
            self.assertEqual(get_tool_risk(tool), ToolRisk.MEDIUM, f"{tool} should be MEDIUM")

    def test_high_tools(self):
        """Destructive/privileged tools are classified as HIGH."""
        high_tools = ["terminal", "execute_code", "delegate_task", "cronjob"]
        for tool in high_tools:
            self.assertEqual(get_tool_risk(tool), ToolRisk.HIGH, f"{tool} should be HIGH")

    def test_unknown_tool_is_high(self):
        """Unknown tools default to HIGH risk."""
        self.assertEqual(get_tool_risk("some_unknown_tool"), ToolRisk.HIGH)

    def test_list_tools_by_risk(self):
        """list_tools_by_risk returns correct sets."""
        safe = list_tools_by_risk(ToolRisk.SAFE)
        self.assertIn("read_file", safe)
        self.assertNotIn("terminal", safe)


class TestCheckToolPermission(unittest.TestCase):
    """Test the check_tool_permission function."""

    def test_safe_tool_auto_approves(self):
        """SAFE tools are always auto-approved."""
        decision = check_tool_permission("read_file")
        self.assertTrue(decision.auto_approve)
        self.assertFalse(decision.blocked)
        self.assertFalse(decision.needs_approval)

    def test_low_tool_auto_approves(self):
        """LOW tools are auto-approved."""
        decision = check_tool_permission("memory")
        self.assertTrue(decision.auto_approve)

    def test_medium_tool_needs_approval_interactive(self):
        """MEDIUM tools need approval in interactive mode."""
        decision = check_tool_permission("write_file", interactive=True)
        self.assertTrue(decision.needs_approval)
        self.assertFalse(decision.auto_approve)

    def test_medium_tool_auto_approves_non_interactive(self):
        """MEDIUM tools auto-approve in non-interactive mode."""
        decision = check_tool_permission("write_file", interactive=False)
        self.assertTrue(decision.auto_approve)
        self.assertFalse(decision.needs_approval)

    def test_high_tool_needs_approval_interactive(self):
        """HIGH tools need approval in interactive mode."""
        decision = check_tool_permission("terminal", interactive=True)
        self.assertTrue(decision.needs_approval)

    def test_high_tool_auto_approves_non_interactive(self):
        """HIGH tools auto-approve in non-interactive mode (user opted in)."""
        decision = check_tool_permission("terminal", interactive=False)
        self.assertTrue(decision.auto_approve)

    def test_yolo_mode_auto_approves_all(self):
        """YOLO mode auto-approves everything except BLOCKED."""
        for tool in ["read_file", "write_file", "terminal", "execute_code"]:
            decision = check_tool_permission(tool, yolo_mode=True)
            self.assertTrue(decision.auto_approve, f"{tool} should auto-approve in YOLO")

    def test_harmful_command_blocked(self):
        """Harmful terminal commands are blocked even in YOLO mode."""
        harmful_commands = [
            "rm -rf /",
            "rm -rf /*",
            "mkfs.ext4 /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
            ":(){:|:&};:",
            "shutdown -h now",
        ]
        for cmd in harmful_commands:
            decision = check_tool_permission(
                "terminal", args={"command": cmd}, yolo_mode=True,
            )
            self.assertTrue(
                decision.blocked,
                f"Command '{cmd}' should be BLOCKED even in YOLO",
            )

    def test_safe_terminal_command_allowed(self):
        """Safe terminal commands are not blocked."""
        safe_commands = [
            "ls -la",
            "cat /etc/hostname",
            "python3 --version",
            "git status",
        ]
        for cmd in safe_commands:
            decision = check_tool_permission(
                "terminal", args={"command": cmd}, yolo_mode=True,
            )
            self.assertFalse(decision.blocked, f"Command '{cmd}' should NOT be blocked")

    def test_execute_code_harmful_blocked(self):
        """Harmful execute_code commands are blocked."""
        decision = check_tool_permission(
            "execute_code", args={"command": "rm -rf /"}, yolo_mode=True,
        )
        self.assertTrue(decision.blocked)

    def test_decision_has_risk(self):
        """Decision always includes risk level."""
        decision = check_tool_permission("read_file")
        self.assertEqual(decision.risk, ToolRisk.SAFE)

    def test_decision_has_reason_when_blocked(self):
        """Blocked decisions include a reason."""
        decision = check_tool_permission(
            "terminal", args={"command": "rm -rf /"},
        )
        self.assertTrue(decision.blocked)
        self.assertIn("blocked", decision.reason.lower())

    def test_decision_has_reason_when_needs_approval(self):
        """Approval-needed decisions include a reason."""
        decision = check_tool_permission("terminal", interactive=True)
        self.assertTrue(decision.needs_approval)
        self.assertIn("HIGH", decision.reason)


class TestPermissionDecision(unittest.TestCase):
    """Test the PermissionDecision dataclass."""

    def test_auto_approve_when_not_blocked_and_not_needs_approval(self):
        d = PermissionDecision(tool_name="test", risk=ToolRisk.SAFE)
        self.assertTrue(d.auto_approve)

    def test_not_auto_approve_when_blocked(self):
        d = PermissionDecision(tool_name="test", risk=ToolRisk.BLOCKED, blocked=True)
        self.assertFalse(d.auto_approve)

    def test_not_auto_approve_when_needs_approval(self):
        d = PermissionDecision(tool_name="test", risk=ToolRisk.HIGH, needs_approval=True)
        self.assertFalse(d.auto_approve)


if __name__ == "__main__":
    unittest.main()
