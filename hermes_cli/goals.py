"""Persistent session goals — the Ralph loop for Hermes.

A goal is a free-form user objective that stays active across turns. After
each turn completes, a small judge call asks an auxiliary model "is this
goal satisfied by the assistant's last response?". If not, Hermes feeds a
continuation prompt back into the same session and keeps working until the
goal is done, turn budget is exhausted, the user pauses/clears it, or the
user sends a new message (which takes priority and pauses the goal loop).

State is persisted in SessionDB's ``state_meta`` table keyed by
``goal:<session_id>`` so ``/resume`` picks it up.

Design notes / invariants:

- The continuation prompt is just a normal user message appended to the
  session via ``run_conversation``. No system-prompt mutation, no toolset
  swap — prompt caching stays intact.
- Judge failures are fail-OPEN: ``continue``. A broken judge must not wedge
  progress; the turn budget is the backstop.
- When a real user message arrives mid-loop it preempts the continuation
  prompt and also pauses the goal loop for that turn (we still re-judge
  after, so if the user's message happens to complete the goal the judge
  will say ``done``).
- This module has zero hard dependency on ``cli.HermesCLI`` or the gateway
  runner — both wire the same ``GoalManager`` in.

Nothing in this module touches the agent's system prompt or toolset.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TURNS = 20
DEFAULT_JUDGE_TIMEOUT = 30.0
# Cap how much of the last response + recent messages we send to the judge.
_JUDGE_RESPONSE_SNIPPET_CHARS = 4000
# Anti-laziness: suppress continuation after N consecutive turns with 0 tool calls.
MAX_IDLE_TURNS = 3
# Wrap-up steering: inject budget warning when this fraction is consumed.
WRAP_UP_FRACTION = 0.9

# Judge retry: how many times to retry on transient failures before giving up.
_JUDGE_MAX_RETRIES = 2


def _is_transient_judge_error(exc: Exception) -> bool:
    """Return True if the exception is likely transient and worth retrying."""
    from openai import (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
    )

    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    # Some providers wrap transient errors in generic APIStatusError
    # with 429 or 5xx status codes.
    if hasattr(exc, "status_code"):
        code = getattr(exc, "status_code", 0)
        if code == 429 or code >= 500:
            return True
    return False


CONTINUATION_PROMPT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "{budget_info}"
    "Continue working toward this goal. Take the next concrete step. "
    "If you believe the goal is complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly and stop."
    "{wrap_up}"
)


JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. You receive the goal text and the "
    "agent's most recent response. Your only job is to decide whether "
    "the goal is fully satisfied based on that response.\n\n"
    "A goal is DONE only when:\n"
    "- The response explicitly confirms the goal was completed, OR\n"
    "- The response clearly shows the final deliverable was produced, OR\n"
    "- The response explains the goal is unachievable / blocked / needs "
    "user input (treat this as DONE with reason describing the block).\n\n"
    "Otherwise the goal is NOT done — CONTINUE.\n\n"
    "Reply ONLY with a single JSON object on one line:\n"
    '{\"done\": <true|false>, \"reason\": \"<one-sentence rationale>\"}'
)


JUDGE_USER_PROMPT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "Is the goal satisfied?"
)


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GoalState:
    """Serializable goal state stored per session."""

    goal: str
    status: str = "active"          # active | paused | done | cleared | budget_limited
    turns_used: int = 0
    max_turns: int = DEFAULT_MAX_TURNS
    created_at: float = 0.0
    last_turn_at: float = 0.0
    last_verdict: Optional[str] = None        # "done" | "continue" | "skipped"
    last_reason: Optional[str] = None
    paused_reason: Optional[str] = None       # why we auto-paused (budget, etc.)
    # Codex /goal enhancements
    token_budget: Optional[int] = None        # token limit (None = unlimited)
    tokens_used: int = 0                      # approximate tokens consumed
    time_used_seconds: float = 0.0            # wall-clock time
    tool_calls_in_turn: int = 0               # tool calls in the last turn
    consecutive_idle_turns: int = 0           # turns with 0 tool calls in a row
    goal_suppressed: bool = False             # anti-laziness: stop continuation

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "GoalState":
        data = json.loads(raw)
        return cls(
            goal=data.get("goal", ""),
            status=data.get("status", "active"),
            turns_used=int(data.get("turns_used", 0) or 0),
            max_turns=int(data.get("max_turns", DEFAULT_MAX_TURNS) or DEFAULT_MAX_TURNS),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_turn_at=float(data.get("last_turn_at", 0.0) or 0.0),
            last_verdict=data.get("last_verdict"),
            last_reason=data.get("last_reason"),
            paused_reason=data.get("paused_reason"),
            token_budget=data.get("token_budget"),
            tokens_used=int(data.get("tokens_used", 0) or 0),
            time_used_seconds=float(data.get("time_used_seconds", 0.0) or 0.0),
            tool_calls_in_turn=int(data.get("tool_calls_in_turn", 0) or 0),
            consecutive_idle_turns=int(data.get("consecutive_idle_turns", 0) or 0),
            goal_suppressed=bool(data.get("goal_suppressed", False)),
        )


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta)
# ──────────────────────────────────────────────────────────────────────


def _meta_key(session_id: str) -> str:
    return f"goal:{session_id}"


_DB_CACHE: Dict[str, Any] = {}


def _get_session_db() -> Optional[Any]:
    """Return a SessionDB instance for the current HERMES_HOME.

    SessionDB has no built-in singleton, but opening a new connection per
    /goal call would thrash the file. We cache one instance per
    ``hermes_home`` path so profile switches still pick up the right DB.
    Defensive against import/instantiation failures so tests and
    non-standard launchers can still use the GoalManager.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        home = str(get_hermes_home())
    except Exception as exc:  # pragma: no cover
        logger.debug("GoalManager: SessionDB bootstrap failed (%s)", exc)
        return None

    cached = _DB_CACHE.get(home)
    if cached is not None:
        return cached
    try:
        db = SessionDB()
    except Exception as exc:  # pragma: no cover
        logger.debug("GoalManager: SessionDB() raised (%s)", exc)
        return None
    _DB_CACHE[home] = db
    return db


def load_goal(session_id: str) -> Optional[GoalState]:
    """Load the goal for a session, or None if none exists."""
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("GoalManager: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        return GoalState.from_json(raw)
    except Exception as exc:
        logger.warning("GoalManager: could not parse stored goal for %s: %s", session_id, exc)
        return None


def save_goal(session_id: str, state: GoalState) -> None:
    """Persist a goal to SessionDB. No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_meta_key(session_id), state.to_json())
    except Exception as exc:
        logger.debug("GoalManager: set_meta failed: %s", exc)


def clear_goal(session_id: str) -> None:
    """Mark a goal cleared in the DB (preserved for audit, status=cleared)."""
    state = load_goal(session_id)
    if state is None:
        return
    state.status = "cleared"
    save_goal(session_id, state)


# ──────────────────────────────────────────────────────────────────────
# Judge
# ──────────────────────────────────────────────────────────────────────


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_judge_response(raw: str) -> Tuple[bool, str]:
    """Parse the judge's reply. Fail-open to ``(False, "<reason>")``.

    Returns ``(done, reason)``.
    """
    if not raw:
        return False, "judge returned empty response"

    text = raw.strip()

    # Strip markdown code fences the model may wrap JSON in.
    if text.startswith("```"):
        text = text.strip("`")
        # Peel off leading json/JSON/etc tag
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]

    # First try: parse the whole blob.
    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(text)
    except Exception:
        # Second try: pull the first JSON object out.
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        return False, f"judge reply was not JSON: {_truncate(raw, 200)!r}"

    done_val = data.get("done")
    if isinstance(done_val, str):
        done = done_val.strip().lower() in ("true", "yes", "1", "done")
    else:
        done = bool(done_val)
    reason = str(data.get("reason") or "").strip()
    if not reason:
        reason = "no reason provided"
    return done, reason


def judge_goal(
    goal: str,
    last_response: str,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
) -> Tuple[str, str]:
    """Ask the auxiliary model whether the goal is satisfied.

    Returns ``(verdict, reason)`` where verdict is ``"done"``, ``"continue"``,
    or ``"judge_unavailable"`` (when the judge couldn't be reached after retries).

    Transient failures (connection errors, timeouts, rate limits) are retried
    up to `_JUDGE_MAX_RETRIES` times.  If all retries are exhausted, or a
    non-recoverable error occurs (import failure, misconfigured client), the
    verdict is ``"judge_unavailable"`` so the caller can pause the goal
    instead of silently continuing.
    """
    if not goal.strip():
        return "skipped", "empty goal"
    if not last_response.strip():
        # No substantive reply this turn — almost certainly not done yet.
        return "continue", "empty response (nothing to evaluate)"

    try:
        from agent.auxiliary_client import get_text_auxiliary_client
    except Exception as exc:
        logger.debug("goal judge: auxiliary client import failed: %s", exc)
        return "judge_unavailable", f"judge import failed: {exc}"

    try:
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception as exc:
        logger.debug("goal judge: get_text_auxiliary_client failed: %s", exc)
        return "judge_unavailable", f"judge client init failed: {exc}"

    if client is None or not model:
        return "judge_unavailable", "no auxiliary client configured"

    prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        goal=_truncate(goal, 2000),
        response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
    )

    last_exc: Optional[Exception] = None
    for attempt in range(_JUDGE_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=200,
                timeout=timeout,
            )
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if _is_transient_judge_error(exc) and attempt < _JUDGE_MAX_RETRIES:
                logger.info(
                    "goal judge: transient error (attempt %d/%d): %s",
                    attempt + 1, _JUDGE_MAX_RETRIES + 1, exc,
                )
                continue
            # Non-transient or last attempt — no more retries
            break

    if last_exc is not None:
        logger.warning("goal judge: API call failed after %d attempt(s): %s", _JUDGE_MAX_RETRIES + 1, last_exc)
        return "judge_unavailable", f"judge error after {_JUDGE_MAX_RETRIES + 1} attempts: {type(last_exc).__name__}: {last_exc}"

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    done, reason = _parse_judge_response(raw)
    verdict = "done" if done else "continue"
    logger.info("goal judge: verdict=%s reason=%s", verdict, _truncate(reason, 120))
    return verdict, reason


# ──────────────────────────────────────────────────────────────────────
# GoalManager — the orchestration surface CLI + gateway talk to
# ──────────────────────────────────────────────────────────────────────


class GoalManager:
    """Per-session goal state + continuation decisions.

    The CLI and gateway each hold one ``GoalManager`` per live session.

    Methods:

    - ``set(goal)`` — start a new standing goal.
    - ``clear()`` — remove the active goal.
    - ``pause()`` / ``resume()`` — explicit user controls.
    - ``status()`` — printable one-liner.
    - ``evaluate_after_turn(last_response)`` — call the judge, update state,
      and return a decision dict the caller uses to drive the next turn.
    - ``next_continuation_prompt()`` — the canonical user-role message to
      feed back into ``run_conversation``.
    """

    def __init__(self, session_id: str, *, default_max_turns: int = DEFAULT_MAX_TURNS):
        self.session_id = session_id
        self.default_max_turns = int(default_max_turns or DEFAULT_MAX_TURNS)
        self._state: Optional[GoalState] = load_goal(session_id)

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[GoalState]:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_goal(self) -> bool:
        return self._state is not None and self._state.status in ("active", "paused")

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status in ("cleared",):
            return "No active goal. Set one with /goal <text>."
        turns = f"{s.turns_used}/{s.max_turns} turns"
        tokens = ""
        if s.token_budget:
            tokens = f", {s.tokens_used}/{s.token_budget} tokens"
        elif s.tokens_used > 0:
            tokens = f", {s.tokens_used} tokens"
        time_info = f", {s.time_used_seconds:.0f}s" if s.time_used_seconds > 0 else ""
        if s.status == "active":
            idle_warn = f" ⚠ idle×{s.consecutive_idle_turns}" if s.consecutive_idle_turns > 0 else ""
            return f"⊙ Goal (active, {turns}{tokens}{time_info}{idle_warn}): {s.goal}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Goal (paused, {turns}{tokens}{time_info}{extra}): {s.goal}"
        if s.status == "done":
            return f"✓ Goal done ({turns}{tokens}{time_info}): {s.goal}"
        if s.status == "budget_limited":
            return f"⏸ Goal budget-limited ({turns}{tokens}{time_info}): {s.goal}"
        return f"Goal ({s.status}, {turns}{tokens}{time_info}): {s.goal}"

    # --- mutation -----------------------------------------------------

    def set(self, goal: str, *, max_turns: Optional[int] = None, token_budget: Optional[int] = None) -> GoalState:
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("goal text is empty")
        state = GoalState(
            goal=goal,
            status="active",
            turns_used=0,
            max_turns=int(max_turns) if max_turns else self.default_max_turns,
            created_at=time.time(),
            token_budget=token_budget,
            last_turn_at=0.0,
        )
        self._state = state
        save_goal(self.session_id, state)
        return state

    def pause(self, reason: str = "user-paused") -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "paused"
        self._state.paused_reason = reason
        save_goal(self.session_id, self._state)
        return self._state

    def resume(self, *, reset_budget: bool = True) -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "active"
        self._state.paused_reason = None
        if reset_budget:
            self._state.turns_used = 0
        save_goal(self.session_id, self._state)
        return self._state

    def clear(self) -> None:
        if self._state is None:
            return
        self._state.status = "cleared"
        save_goal(self.session_id, self._state)
        self._state = None

    def mark_done(self, reason: str) -> None:
        if not self._state:
            return
        self._state.status = "done"
        self._state.last_verdict = "done"
        self._state.last_reason = reason
        save_goal(self.session_id, self._state)

    # --- the main entry point called after every turn -----------------

    def evaluate_after_turn(
        self,
        last_response: str,
        *,
        user_initiated: bool = True,
        tool_calls_count: int = 0,
        tokens_used: int = 0,
    ) -> Dict[str, Any]:
        """Run the judge and update state. Return a decision dict.

        ``user_initiated`` distinguishes a real user prompt (True) from a
        continuation prompt we fed ourselves (False). Both increment
        ``turns_used`` because both consume model budget.

        ``tool_calls_count`` is the number of tool calls the model made in
        this turn. Used for anti-laziness detection.

        ``tokens_used`` is the approximate token count for this turn. Used
        for token budget tracking.

        Decision keys:
          - ``status``: current goal status after update
          - ``should_continue``: bool — caller should fire another turn
          - ``continuation_prompt``: str or None
          - ``verdict``: "done" | "continue" | "skipped" | "inactive"
          - ``reason``: str
          - ``message``: user-visible one-liner to print/send
        """
        state = self._state
        if state is None or state.status != "active":
            return {
                "status": state.status if state else None,
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "inactive",
                "reason": "no active goal",
                "message": "",
            }

        # Count the turn that just finished.
        state.turns_used += 1
        state.last_turn_at = time.time()

        # ── Token budget tracking (Codex /goal) ──────────────────────
        if tokens_used > 0:
            state.tokens_used += tokens_used
        if state.created_at > 0:
            state.time_used_seconds = time.time() - state.created_at
        state.tool_calls_in_turn = tool_calls_count

        # ── Anti-laziness detection (Codex /goal) ────────────────────
        # If this is a continuation turn (not user-initiated) and the model
        # made zero tool calls, it's "idle" — suppress further continuation.
        if not user_initiated and tool_calls_count == 0:
            state.consecutive_idle_turns += 1
            if state.consecutive_idle_turns >= MAX_IDLE_TURNS:
                state.goal_suppressed = True
                save_goal(self.session_id, state)
                return {
                    "status": "paused",
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "continue",
                    "reason": "anti-laziness: no tool calls for consecutive turns",
                    "message": (
                        f"⏸ Goal paused — no progress for {state.consecutive_idle_turns} turns. "
                        "The agent made no tool calls in continuation turns. "
                        "Use /goal resume to retry, or /goal clear to stop."
                    ),
                }
        else:
            # Reset idle counter on productive turn
            state.consecutive_idle_turns = 0
            state.goal_suppressed = False

        # ── Token budget check (Codex /goal) ─────────────────────────
        if state.token_budget and state.tokens_used >= state.token_budget:
            state.status = "budget_limited"
            save_goal(self.session_id, state)
            return {
                "status": "budget_limited",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": f"token budget exhausted ({state.tokens_used}/{state.token_budget})",
                "message": (
                    f"⏸ Goal paused — token budget exhausted "
                    f"({state.tokens_used}/{state.token_budget} tokens, "
                    f"{state.time_used_seconds:.0f}s elapsed). "
                    "Use /goal resume to continue with a new budget, or /goal clear to stop."
                ),
            }

        # ── Judge evaluation ────────────────────────────────────────
        verdict, reason = judge_goal(state.goal, last_response)
        state.last_verdict = verdict
        state.last_reason = reason

        # Judge unavailable (API down after retries) → fail-closed: pause
        # so the user can resume when the judge is back.  This is safer than
        # fail-open which silently continues forever.
        if verdict == "judge_unavailable":
            state.status = "paused"
            state.paused_reason = f"judge unavailable: {reason}"
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "judge_unavailable",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — judge API unavailable after {_JUDGE_MAX_RETRIES + 1} attempts. "
                    "Use /goal resume when the API is back, or /goal clear to stop."
                ),
            }

        if verdict == "done":
            state.status = "done"
            save_goal(self.session_id, state)
            return {
                "status": "done",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "done",
                "reason": reason,
                "message": f"✓ Goal achieved: {reason}",
            }

        if state.turns_used >= state.max_turns:
            state.status = "paused"
            state.paused_reason = f"turn budget exhausted ({state.turns_used}/{state.max_turns})"
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — {state.turns_used}/{state.max_turns} turns used. "
                    "Use /goal resume to keep going, or /goal clear to stop."
                ),
            }

        save_goal(self.session_id, state)
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": self.next_continuation_prompt(),
            "verdict": "continue",
            "reason": reason,
            "message": (
                f"↻ Continuing toward goal ({state.turns_used}/{state.max_turns}): {reason}"
            ),
        }

    def next_continuation_prompt(self) -> Optional[str]:
        if not self._state or self._state.status != "active":
            return None
        s = self._state
        # Build budget info string
        budget_info = ""
        if s.token_budget:
            remaining = max(0, s.token_budget - s.tokens_used)
            budget_info = (
                f"Budget: {s.tokens_used}/{s.token_budget} tokens used, "
                f"{remaining} remaining, {s.time_used_seconds:.0f}s elapsed\n\n"
            )
        else:
            budget_info = (
                f"Tokens used: {s.tokens_used}, "
                f"Time: {s.time_used_seconds:.0f}s elapsed\n\n"
            )
        # Wrap-up steering when near budget limit
        wrap_up = ""
        if s.token_budget and s.tokens_used >= s.token_budget * WRAP_UP_FRACTION:
            wrap_up = (
                "\n\nWRAP-UP: You are near the token budget limit. "
                "Finish current work, summarize progress, and stop."
            )
        return CONTINUATION_PROMPT_TEMPLATE.format(
            goal=s.goal, budget_info=budget_info, wrap_up=wrap_up,
        )


__all__ = [
    "GoalState",
    "GoalManager",
    "CONTINUATION_PROMPT_TEMPLATE",
    "DEFAULT_MAX_TURNS",
    "MAX_IDLE_TURNS",
    "WRAP_UP_FRACTION",
    "load_goal",
    "save_goal",
    "clear_goal",
    "judge_goal",
]
