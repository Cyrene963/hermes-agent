"""Per-tool permission system with risk-based classification.

Inspired by Claude Code's canUseTool() — classifies each tool by risk
level and provides a single check function that determines whether a
tool call should be auto-approved, require confirmation, or be blocked.

This module is a lightweight layer ON TOP of the existing approval.py
system, not a replacement. It adds tool-level risk classification that
the agent loop can consult before executing each tool call.

Risk levels:
  SAFE      — Read-only, no side effects. Always auto-approved.
  LOW       — Minor side effects (memory, skills, todo). Auto-approved
              in most modes.
  MEDIUM    — Significant side effects (file writes, patches). Requires
              confirmation in interactive mode.
  HIGH      — Destructive or privileged (terminal, execute_code, delegate).
              Always requires confirmation (unless YOLO mode).
  BLOCKED   — Never allowed (hardline blocklist).

Usage:
    from agent.tool_permissions import check_tool_permission, ToolRisk

    decision = check_tool_permission("terminal", args={"command": "rm -rf /"})
    if decision.blocked:
        return f"BLOCKED: {decision.reason}"
    if decision.needs_approval:
        # prompt user for confirmation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class ToolRisk(Enum):
    """Risk classification for tools."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


# ── Tool → Risk mapping ──────────────────────────────────────────────
# Based on Claude Code's per-tool canUseTool pattern and Hermes's
# existing tool_guardrails.py classifications.

TOOL_RISK_MAP: Dict[str, ToolRisk] = {
    # SAFE: Read-only, no side effects
    "read_file": ToolRisk.SAFE,
    "search_files": ToolRisk.SAFE,
    "web_search": ToolRisk.SAFE,
    "web_extract": ToolRisk.SAFE,
    "session_search": ToolRisk.SAFE,
    "hindsight_recall": ToolRisk.SAFE,
    "hindsight_reflect": ToolRisk.SAFE,
    "skills_list": ToolRisk.SAFE,
    "skill_view": ToolRisk.SAFE,
    "browser_snapshot": ToolRisk.SAFE,
    "browser_console": ToolRisk.SAFE,
    "browser_get_images": ToolRisk.SAFE,
    "vision_analyze": ToolRisk.SAFE,
    "browser_back": ToolRisk.SAFE,
    "todo": ToolRisk.LOW,  # read-only when no args

    # LOW: Minor side effects
    "memory": ToolRisk.LOW,
    "hindsight_retain": ToolRisk.LOW,
    "skill_manage": ToolRisk.LOW,
    "clarify": ToolRisk.LOW,
    "send_message": ToolRisk.LOW,

    # MEDIUM: Significant side effects
    "write_file": ToolRisk.MEDIUM,
    "patch": ToolRisk.MEDIUM,
    "browser_click": ToolRisk.MEDIUM,
    "browser_type": ToolRisk.MEDIUM,
    "browser_press": ToolRisk.MEDIUM,
    "browser_scroll": ToolRisk.MEDIUM,
    "browser_navigate": ToolRisk.MEDIUM,
    "browser_vision": ToolRisk.MEDIUM,
    "text_to_speech": ToolRisk.MEDIUM,
    "process": ToolRisk.MEDIUM,

    # HIGH: Destructive or privileged
    "terminal": ToolRisk.HIGH,
    "execute_code": ToolRisk.HIGH,
    "delegate_task": ToolRisk.HIGH,
    "cronjob": ToolRisk.HIGH,
}

# ── Hardline blocklist ────────────────────────────────────────────────
# Commands that should NEVER execute, regardless of mode.
# Mirrors approval.py's DANGEROUS_PATTERNS but at the tool level.
_HARMLINE_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs.",
    "dd if=/dev/zero",
    "dd if=/dev/random",
    ":(){:|:&};:",  # fork bomb
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
]


@dataclass
class PermissionDecision:
    """Result of a tool permission check."""
    tool_name: str
    risk: ToolRisk
    blocked: bool = False
    needs_approval: bool = False
    reason: str = ""

    @property
    def auto_approve(self) -> bool:
        """Whether this tool call can be auto-approved."""
        return not self.blocked and not self.needs_approval


def check_tool_permission(
    tool_name: str,
    args: Dict[str, Any] = None,
    yolo_mode: bool = False,
    interactive: bool = True,
) -> PermissionDecision:
    """Check whether a tool call should be allowed.

    Args:
        tool_name: Name of the tool being called.
        args: Tool arguments (used for dangerous command detection).
        yolo_mode: If True, auto-approve everything except BLOCKED.
        interactive: If False (cron, gateway non-interactive), auto-approve
                     MEDIUM and below.

    Returns:
        PermissionDecision with blocked/needs_approval/auto_approve.
    """
    risk = TOOL_RISK_MAP.get(tool_name, ToolRisk.HIGH)  # unknown = HIGH

    # BLOCKED: never allowed
    if risk == ToolRisk.BLOCKED:
        return PermissionDecision(
            tool_name=tool_name,
            risk=risk,
            blocked=True,
            reason=f"Tool '{tool_name}' is permanently blocked.",
        )

    # Check for dangerous commands in terminal/execute_code
    if tool_name in ("terminal", "execute_code") and args:
        # terminal uses "command", execute_code uses "code"
        command = args.get("command", "") or args.get("code", "")
        if _is_harmful_command(command):
            return PermissionDecision(
                tool_name=tool_name,
                risk=ToolRisk.BLOCKED,
                blocked=True,
                reason=f"Command blocked by hardline policy: {_truncate(command, 80)}",
            )

    # YOLO mode: auto-approve everything except BLOCKED
    if yolo_mode:
        return PermissionDecision(
            tool_name=tool_name,
            risk=risk,
        )

    # Risk-based decision
    if risk == ToolRisk.SAFE:
        return PermissionDecision(tool_name=tool_name, risk=risk)

    if risk == ToolRisk.LOW:
        return PermissionDecision(tool_name=tool_name, risk=risk)

    if risk == ToolRisk.MEDIUM:
        if not interactive:
            # Non-interactive mode (cron, gateway): auto-approve MEDIUM
            return PermissionDecision(tool_name=tool_name, risk=risk)
        return PermissionDecision(
            tool_name=tool_name,
            risk=risk,
            needs_approval=True,
            reason=f"Tool '{tool_name}' has MEDIUM risk — requires confirmation.",
        )

    if risk == ToolRisk.HIGH:
        if not interactive:
            # Non-interactive mode: still approve HIGH (user opted in)
            return PermissionDecision(tool_name=tool_name, risk=risk)
        return PermissionDecision(
            tool_name=tool_name,
            risk=risk,
            needs_approval=True,
            reason=f"Tool '{tool_name}' has HIGH risk — requires confirmation.",
        )

    # Fallback: require approval
    return PermissionDecision(
        tool_name=tool_name,
        risk=risk,
        needs_approval=True,
        reason=f"Unknown risk level for '{tool_name}'.",
    )


def _is_harmful_command(command: str) -> bool:
    """Check if a command matches hardline blocklist patterns."""
    cmd_lower = command.lower().strip()
    for pattern in _HARMLINE_PATTERNS:
        if pattern.lower() in cmd_lower:
            return True
    return False


def _truncate(s: str, max_len: int) -> str:
    """Truncate string with ellipsis."""
    return s[:max_len] + "..." if len(s) > max_len else s


def get_tool_risk(tool_name: str) -> ToolRisk:
    """Get the risk level for a tool."""
    return TOOL_RISK_MAP.get(tool_name, ToolRisk.HIGH)


def list_tools_by_risk(risk: ToolRisk) -> Set[str]:
    """List all tools with a given risk level."""
    return {name for name, r in TOOL_RISK_MAP.items() if r == risk}
