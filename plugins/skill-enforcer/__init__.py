"""skill-enforcer plugin — periodic compliance checkpoints during agent execution.

Addresses the "having rules ≠ following rules" problem.

PR #18316 (hybrid mode) already selects relevant skills and injects them into
the system prompt. This plugin doesn't duplicate that. Instead, it forces
periodic compliance checkpoints to make sure the agent is actually FOLLOWING
the rules in the loaded skills, not just having them in context.

Design:
- Tracks action tool call count per session
- Every N action tool calls, BLOCK with a compliance checkpoint message
- The checkpoint asks: "are you following the rules in your loaded skills?"
- Agent acknowledges by calling any self-check tool (skill_view, hindsight_recall, etc.)
- Counter resets after acknowledgment
- Non-action tools (read, search, explore) don't count
- This is NOT a one-time gate — it's periodic enforcement throughout the session
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# How many action tool calls before forcing a compliance checkpoint
# Increased from 8 to 25 — 8 was too aggressive, causing frequent interruptions
# and wasted tokens on hindsight_recall. 25 strikes a better balance between
# compliance checking and task flow.
_CHECKPOINT_INTERVAL = 25

# Tools that are "actions" — they do things, not just read/search
_ACTION_TOOLS = frozenset({
    "terminal",
    "write_file",
    "patch",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "delegate_task",
    "cronjob",
    "send_message",
    "text_to_speech",
    "execute_code",
})

# Tools that count as "compliance acknowledgment"
_ACKNOWLEDGMENT_TOOLS = frozenset({
    "skill_view",
    "hindsight_recall",
    "hindsight_reflect",
    "session_search",
    "memory",
})

# Per-session state
_session_action_count: Dict[str, int] = {}  # session_id -> action call count
_session_since_ack: Dict[str, int] = {}     # session_id -> calls since last ack
_lock = threading.Lock()


def _session_key(task_id: str, session_id: str) -> str:
    return task_id or session_id or "default"


def _on_pre_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Pre-tool-call hook: periodic compliance checkpoints."""

    key = _session_key(task_id, session_id)

    # Acknowledgment tools: reset counter and pass
    if tool_name in _ACKNOWLEDGMENT_TOOLS:
        with _lock:
            _session_since_ack[key] = 0
        return None

    # Non-action tools always pass
    if tool_name not in _ACTION_TOOLS:
        return None

    # Action tool: increment counter
    with _lock:
        count = _session_action_count.get(key, 0) + 1
        _session_action_count[key] = count
        since_ack = _session_since_ack.get(key, 0) + 1
        _session_since_ack[key] = since_ack

    # Check if checkpoint is due
    if since_ack >= _CHECKPOINT_INTERVAL:
        with _lock:
            _session_since_ack[key] = 0  # Reset to prevent immediate re-block

        msg = (
            f"COMPLIANCE CHECKPOINT (after {count} action calls in this session): "
            f"You have executed {since_ack} action tool calls since your last "
            f"context check. Before proceeding, verify:\n"
            f"1. Are you following the rules in your loaded skills?\n"
            f"2. Are you fabricating data or guessing? If uncertain, search first.\n"
            f"3. Are you handling errors per your error recovery rules?\n"
            f"4. Have you verified deployment results (curl, test)?\n\n"
            f"Call skill_view(name), hindsight_recall(query), or "
            f"session_search(query) to acknowledge compliance, then retry "
            f"your original tool call."
        )

        logger.info(
            "skill-enforcer: checkpoint at call #%d for session %s",
            count, key,
        )
        return {"action": "block", "message": msg}

    return None


def register(ctx) -> None:
    """Register the pre_tool_call hook."""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    logger.info("skill-enforcer plugin registered (periodic checkpoints)")
