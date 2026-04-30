#!/usr/bin/env python3
"""
Goal Tool Module - Persistent Goal Tracking

Provides CRUD tools for goals persisted in SessionDB's state_meta table.
Goals are stored as JSON keyed by ``goal:<goal_id>``.

Tools:
  - create_goal: create a new goal with a description
  - get_goal:    retrieve a goal by id
  - update_goal: mark a goal as complete (ONLY valid status transition)

Design:
- Each goal has: id, description, status (active|complete), created_at, completed_at
- update_goal can ONLY set status to 'complete' — the model cannot
  pause, resume, or clear goals.  This keeps the tool surface minimal
  and prevents the agent from gaming the goal lifecycle.
- Persistence uses hermes_state.SessionDB get_meta / set_meta.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────

_DB_CACHE: Dict[str, Any] = {}


def _get_session_db():
    """Return a SessionDB instance for the current HERMES_HOME.

    Cached per hermes_home path so profile switches work correctly.
    Returns None if SessionDB is unavailable.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        home = str(get_hermes_home())
    except Exception as exc:
        logger.debug("goal_tool: SessionDB bootstrap failed (%s)", exc)
        return None

    cached = _DB_CACHE.get(home)
    if cached is not None:
        return cached
    try:
        db = SessionDB()
    except Exception as exc:
        logger.debug("goal_tool: SessionDB() raised (%s)", exc)
        return None
    _DB_CACHE[home] = db
    return db


def _meta_key(goal_id: str) -> str:
    return f"goal:{goal_id}"


def _load_goal(goal_id: str) -> Optional[Dict[str, Any]]:
    """Load a goal dict from SessionDB, or None."""
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(goal_id))
    except Exception as exc:
        logger.debug("goal_tool: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as exc:
        logger.warning("goal_tool: could not parse goal %s: %s", goal_id, exc)
        return None


def _save_goal(goal_id: str, goal: Dict[str, Any]) -> bool:
    """Persist a goal dict to SessionDB. Returns True on success."""
    db = _get_session_db()
    if db is None:
        return False
    try:
        db.set_meta(_meta_key(goal_id), json.dumps(goal, ensure_ascii=False))
        return True
    except Exception as exc:
        logger.debug("goal_tool: set_meta failed: %s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────
# Handler functions
# ──────────────────────────────────────────────────────────────────────


def _create_goal(args: Dict[str, Any]) -> str:
    """Create a new goal."""
    description = (args.get("description") or "").strip()
    if not description:
        return tool_error("description is required")

    goal_id = uuid.uuid4().hex[:12]
    now = time.time()
    goal = {
        "id": goal_id,
        "description": description,
        "status": "active",
        "created_at": now,
        "completed_at": None,
    }

    if not _save_goal(goal_id, goal):
        return tool_error("Failed to persist goal — SessionDB unavailable")

    return json.dumps({
        "status": "ok",
        "goal": goal,
    }, ensure_ascii=False)


def _get_goal(args: Dict[str, Any]) -> str:
    """Retrieve a goal by id."""
    goal_id = (args.get("goal_id") or "").strip()
    if not goal_id:
        return tool_error("goal_id is required")

    goal = _load_goal(goal_id)
    if goal is None:
        return tool_error(f"Goal '{goal_id}' not found")

    return json.dumps({
        "status": "ok",
        "goal": goal,
    }, ensure_ascii=False)


def _update_goal(args: Dict[str, Any]) -> str:
    """Update a goal. Only status='complete' is allowed."""
    goal_id = (args.get("goal_id") or "").strip()
    if not goal_id:
        return tool_error("goal_id is required")

    status = (args.get("status") or "").strip().lower()
    if status != "complete":
        return tool_error(
            "Only status='complete' is allowed via update_goal. "
            "Cannot pause, resume, or clear goals."
        )

    goal = _load_goal(goal_id)
    if goal is None:
        return tool_error(f"Goal '{goal_id}' not found")

    if goal["status"] == "complete":
        return json.dumps({
            "status": "ok",
            "goal": goal,
            "message": "Goal was already complete.",
        }, ensure_ascii=False)

    goal["status"] = "complete"
    goal["completed_at"] = time.time()

    if not _save_goal(goal_id, goal):
        return tool_error("Failed to persist goal update — SessionDB unavailable")

    return json.dumps({
        "status": "ok",
        "goal": goal,
    }, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────
# Requirements check
# ──────────────────────────────────────────────────────────────────────


def check_goal_requirements() -> bool:
    """Goal tools require SessionDB to be importable."""
    try:
        from hermes_state import SessionDB  # noqa: F401
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# OpenAI Function-Calling Schemas
# ──────────────────────────────────────────────────────────────────────

CREATE_GOAL_SCHEMA = {
    "name": "create_goal",
    "description": (
        "Create a new goal to track a high-level objective. "
        "Returns the goal with a unique id. Use get_goal to check "
        "progress and update_goal to mark it complete."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "A clear description of the goal to achieve.",
            },
        },
        "required": ["description"],
    },
}

GET_GOAL_SCHEMA = {
    "name": "get_goal",
    "description": (
        "Retrieve a goal by its id. Returns the full goal state "
        "including status and timestamps."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal_id": {
                "type": "string",
                "description": "The unique id of the goal to retrieve.",
            },
        },
        "required": ["goal_id"],
    },
}

UPDATE_GOAL_SCHEMA = {
    "name": "update_goal",
    "description": (
        "Update a goal's status. The ONLY allowed transition is "
        "setting status to 'complete'. You cannot pause, resume, "
        "or clear goals through this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal_id": {
                "type": "string",
                "description": "The unique id of the goal to update.",
            },
            "status": {
                "type": "string",
                "enum": ["complete"],
                "description": "Must be 'complete'.",
            },
        },
        "required": ["goal_id", "status"],
    },
}


# ──────────────────────────────────────────────────────────────────────
# Tool error helper (imported from registry)
# ──────────────────────────────────────────────────────────────────────

from tools.registry import registry, tool_error  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

registry.register(
    name="create_goal",
    toolset="goal",
    schema=CREATE_GOAL_SCHEMA,
    handler=lambda args, **kw: _create_goal(args),
    check_fn=check_goal_requirements,
    emoji="🎯",
)

registry.register(
    name="get_goal",
    toolset="goal",
    schema=GET_GOAL_SCHEMA,
    handler=lambda args, **kw: _get_goal(args),
    check_fn=check_goal_requirements,
    emoji="🎯",
)

registry.register(
    name="update_goal",
    toolset="goal",
    schema=UPDATE_GOAL_SCHEMA,
    handler=lambda args, **kw: _update_goal(args),
    check_fn=check_goal_requirements,
    emoji="🎯",
)
