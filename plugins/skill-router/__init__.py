"""skill-router plugin — auto-load skills on first action tool call.

Solves the core problem: "agent has skills but doesn't load them."
Instead of relying on agent self-discipline, this plugin forces a
skill-loading checkpoint on the FIRST action tool call in a session.

How it works:
1. Tracks whether any skill has been loaded this session
2. On the first action tool call, BLOCKS with a checkpoint
3. The checkpoint asks the agent to load task-orchestrator before proceeding
4. task-orchestrator then tells the agent which other skills to load
5. Once any skill_view is called, the gate is satisfied for this session

This is NOT a replacement for skill-enforcer. It's complementary:
- skill-router: fires ONCE at session start -> ensures skills are loaded
- skill-enforcer: fires every N calls -> ensures skills are followed

Together they form a complete enforcement chain:
  Load skills (router) -> Follow skills (enforcer) -> Periodic verification (enforcer)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# How many action tool calls before the gate fires (0 = first call)
_GATE_ON_CALL = 0

# Tools that count as "action" -- they modify state
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

# Tools that count as "skill loading" -- they load context
_SKILL_LOADING_TOOLS = frozenset({
    "skill_view",
    "skill_manage",
    "skills_list",
})

# Per-session state
_session_action_count: Dict[str, int] = {}
_session_gated: Dict[str, bool] = {}
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
    """Pre-tool-call hook: force skill loading on first action."""

    key = _session_key(task_id, session_id)

    # Skill-loading tools always pass and satisfy the gate
    if tool_name in _SKILL_LOADING_TOOLS:
        with _lock:
            _session_gated[key] = True
        return None

    # Non-action tools always pass (read, search, etc.)
    if tool_name not in _ACTION_TOOLS:
        return None

    # Action tool: check if gate is satisfied
    with _lock:
        if _session_gated.get(key, False):
            return None  # Gate already satisfied

        count = _session_action_count.get(key, 0) + 1
        _session_action_count[key] = count

        if count > _GATE_ON_CALL:
            return None  # Past the gate window

        # Gate fires here -- BLOCK
        _session_gated[key] = False

    msg = (
        "SKILL ROUTER CHECKPOINT: This is your first action in this session. "
        "Before executing any tool, you MUST analyze the task complexity:\n\n"
        "1. Is this task expected to take >10 minutes or >15 tool calls?\n"
        "2. Does it involve multiple chapters, files, or steps?\n"
        "3. Did the user give a time budget (\"you have N hours\", \"take your time\")?\n\n"
        "If ANY answer is YES:\n"
        "  -> Call skill_view(name='task-orchestrator') to load the orchestration skill\n"
        "  -> It will tell you which additional skills to load\n\n"
        "If ALL answers are NO (simple/quick task):\n"
        "  -> You may proceed directly, but call skill_view if unsure\n\n"
        "After loading skills (or confirming trivial task), retry your original tool call."
    )

    logger.info("skill-router: gate check #%d for session %s", count, key)
    return {"action": "block", "message": msg}


def register(ctx) -> None:
    """Register the pre_tool_call hook."""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    logger.info("skill-router plugin registered (first-call gate)")
