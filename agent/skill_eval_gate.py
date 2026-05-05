"""
Skill Evaluation Gate — forces the agent to evaluate and load relevant skills
before taking any action.

This is a code-level enforcement mechanism. It does NOT use keyword matching.
The agent sees all skill names and descriptions in the system prompt (via
build_skills_system_prompt), and uses its own semantic understanding to decide
which skills are relevant.

Enforcement:
  - Before the first action tool call, the agent MUST call skill_view() for
    at least one relevant skill, OR explicitly acknowledge no skills are needed.
  - If the agent skips this step, the tool call is blocked with a reminder.
  - After skill_view() is called once, the gate opens and all tools proceed.

This replaces the keyword-based auto_inject approach with a universal,
language-agnostic mechanism that works for any task in any language.
"""
from __future__ import annotations
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# The mandatory instruction injected into the system prompt.
SKILL_EVAL_INSTRUCTION = """## Mandatory Skill Evaluation

Before your FIRST action (any tool call beyond read-only tools like read_file,
search_files, browser_snapshot, or hindsight_recall), you MUST:

1. Review the skill index in your system prompt (listed above)
2. Call `skill_view()` for any skills whose description matches the user's task
3. If no skills match, proceed without loading — but you MUST have considered them

This is code-enforced. Skipping this step will cause your tool calls to be blocked.
The skill index contains names and descriptions — use your understanding of the
user's task to determine relevance. This applies to ALL languages and task types."""

# Tools that are allowed before skill evaluation (read-only / info tools)
_READ_ONLY_TOOLS = frozenset({
    "read_file",
    "search_files",
    "browser_snapshot",
    "browser_get_images",
    "browser_console",
    "hindsight_recall",
    "hindsight_reflect",
    "session_search",
    "skills_list",
    "skill_view",       # Allow skill_view itself (this is what we want!)
    "clarify",          # Asking user a question is always OK
    "todo",             # Planning is OK
    "memory",           # Memory read is OK
    "send_message",     # Messaging is OK (for gateway)
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
