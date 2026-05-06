"""
from __future__ import annotations
import logging
from typing import Dict

logger = logging.getLogger(__name__)

    "cronjob",          # Cron management is OK
})

# Per-session tracking: which sessions have completed skill evaluation
# Key: session_id (or task_id as fallback), Value: True if evaluated
_evaluated_sessions: Dict[str, bool] = {}


def get_skill_eval_instruction() -> str:
    """Return the mandatory instruction to inject into system prompt."""
    return SKILL_EVAL_INSTRUCTION


def should_block_tool(tool_name: str, session_id: str = "") -> bool:
    """Check if a tool should be blocked pending skill evaluation.

    Returns True if the tool call should be blocked. Returns False if:
    - The tool is read-only (allowed before evaluation)
    - The session has already completed skill evaluation
    - No session tracking is available (fail-open for safety)
    """
    # Read-only tools are always allowed
    if tool_name in _READ_ONLY_TOOLS:
        return False

    # No session ID — can't track, fail-open
    if not session_id:
        return False

    # Already evaluated this session — allow everything
    if _evaluated_sessions.get(session_id, False):
        return False

    # First action tool call in this session — block it
    logger.info(
        "skill_eval_gate: blocking '%s' for session '%s' — no skill_view() called yet",
        tool_name, session_id,
    )
    return True


def mark_evaluated(session_id: str) -> None:
    """Mark a session as having completed skill evaluation.

    Called when the agent calls skill_view() for the first time.
    """
    if session_id and not _evaluated_sessions.get(session_id, False):
        _evaluated_sessions[session_id] = True
        logger.info("skill_eval_gate: session '%s' now marked as evaluated", session_id)


def is_skill_view_call(tool_name: str) -> bool:
    """Check if this tool call satisfies the skill evaluation requirement."""
    return tool_name == "skill_view"


def reset_session(session_id: str) -> None:
    """Reset evaluation state for a session (e.g., on new conversation)."""
    _evaluated_sessions.pop(session_id, None)