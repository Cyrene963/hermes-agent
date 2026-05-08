"""Tests for hermes_cli/goals.py — persistent cross-turn goals."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal-module's DB cache for each test so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# _parse_judge_response
# ──────────────────────────────────────────────────────────────────────


class TestParseJudgeResponse:
    def test_clean_json_done(self):
        from hermes_cli.goals import _parse_judge_response

        done, reason = _parse_judge_response('{"done": true, "reason": "all good"}')
        assert done is True
        assert reason == "all good"

    def test_clean_json_continue(self):
        from hermes_cli.goals import _parse_judge_response

        done, reason = _parse_judge_response('{"done": false, "reason": "more work needed"}')
        assert done is False
        assert reason == "more work needed"

    def test_json_in_markdown_fence(self):
        from hermes_cli.goals import _parse_judge_response

        raw = '```json\n{"done": true, "reason": "done"}\n```'
        done, reason = _parse_judge_response(raw)
        assert done is True
        assert "done" in reason

    def test_json_embedded_in_prose(self):
        """Some models prefix reasoning before emitting JSON — we extract it."""
        from hermes_cli.goals import _parse_judge_response

        raw = 'Looking at this... the agent says X. Verdict: {"done": false, "reason": "partial"}'
        done, reason = _parse_judge_response(raw)
        assert done is False
        assert reason == "partial"

    def test_string_done_values(self):
        from hermes_cli.goals import _parse_judge_response

        for s in ("true", "yes", "done", "1"):
            done, _ = _parse_judge_response(f'{{"done": "{s}", "reason": "r"}}')
            assert done is True
        for s in ("false", "no", "not yet"):
            done, _ = _parse_judge_response(f'{{"done": "{s}", "reason": "r"}}')
            assert done is False

    def test_malformed_json_fails_open(self):
        """Non-JSON → not done, with error-ish reason (so judge_goal can map to continue)."""
        from hermes_cli.goals import _parse_judge_response

        done, reason = _parse_judge_response("this is not json at all")
        assert done is False
        assert reason  # non-empty

    def test_empty_response(self):
        from hermes_cli.goals import _parse_judge_response

        done, reason = _parse_judge_response("")
        assert done is False
        assert reason


# ──────────────────────────────────────────────────────────────────────
# judge_goal — fail-open semantics
# ──────────────────────────────────────────────────────────────────────


class TestJudgeGoal:
    def test_empty_goal_skipped(self):
        from hermes_cli.goals import judge_goal

        verdict, _ = judge_goal("", "some response")
        assert verdict == "skipped"

    def test_empty_response_continues(self):
        from hermes_cli.goals import judge_goal

        verdict, _ = judge_goal("ship the thing", "")
        assert verdict == "continue"

    def test_no_aux_client_returns_judge_unavailable(self):
        """No aux client → judge_unavailable (fail-closed)."""
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(None, None),
        ):
            verdict, _ = goals.judge_goal("my goal", "my response")
        assert verdict == "judge_unavailable"

    def test_api_error_returns_judge_unavailable(self):
        """Judge exception after retries → fail-closed (judge_unavailable)."""
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason = goals.judge_goal("goal", "response")
        # RuntimeError is non-transient → no retries, immediate judge_unavailable
        assert verdict == "judge_unavailable"
        assert "judge error" in reason.lower()

    def test_transient_error_retries_then_fails(self):
        """Transient errors (connection, timeout) are retried before giving up."""
        from hermes_cli import goals
        from openai import APIConnectionError

        fake_client = MagicMock()
        # All 3 attempts (1 initial + 2 retries) fail with connection error
        fake_client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason = goals.judge_goal("goal", "response")
        assert verdict == "judge_unavailable"
        assert "3 attempts" in reason
        # Should have tried 3 times (initial + 2 retries)
        assert fake_client.chat.completions.create.call_count == 3

    def test_transient_error_succeeds_on_retry(self):
        """If a transient error recovers on retry, we get a normal verdict."""
        from hermes_cli import goals
        from openai import APITimeoutError

        fake_client = MagicMock()
        # First attempt fails, second succeeds
        fake_client.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            MagicMock(choices=[MagicMock(message=MagicMock(
                content='{"done": true, "reason": "recovered"}'
            ))]),
        ]
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason = goals.judge_goal("goal", "response")
        assert verdict == "done"
        assert reason == "recovered"
        assert fake_client.chat.completions.create.call_count == 2

    def test_judge_says_done(self):
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='{"done": true, "reason": "achieved"}')
                )
            ]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason = goals.judge_goal("goal", "agent response")
        assert verdict == "done"
        assert reason == "achieved"

    def test_judge_says_continue(self):
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='{"done": false, "reason": "not yet"}')
                )
            ]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason = goals.judge_goal("goal", "agent response")
        assert verdict == "continue"
        assert reason == "not yet"


# ──────────────────────────────────────────────────────────────────────
# GoalManager lifecycle + persistence
# ──────────────────────────────────────────────────────────────────────


class TestGoalManager:
    def test_no_goal_initial(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-1")
        assert mgr.state is None
        assert not mgr.is_active()
        assert not mgr.has_goal()
        assert "No active goal" in mgr.status_line()

    def test_set_then_status(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-2", default_max_turns=5)
        state = mgr.set("port the thing")
        assert state.goal == "port the thing"
        assert state.status == "active"
        assert state.max_turns == 5
        assert state.turns_used == 0
        assert mgr.is_active()
        assert "active" in mgr.status_line().lower()
        assert "port the thing" in mgr.status_line()

    def test_set_rejects_empty(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-3")
        with pytest.raises(ValueError):
            mgr.set("")
        with pytest.raises(ValueError):
            mgr.set("   ")

    def test_pause_and_resume(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-4")
        mgr.set("goal text")
        mgr.pause(reason="user-paused")
        assert mgr.state.status == "paused"
        assert not mgr.is_active()
        assert mgr.has_goal()

        mgr.resume()
        assert mgr.state.status == "active"
        assert mgr.is_active()

    def test_clear(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-5")
        mgr.set("goal")
        mgr.clear()
        assert mgr.state is None
        assert not mgr.is_active()

    def test_persistence_across_managers(self, hermes_home):
        """Key invariant: a second manager on the same session sees the goal.

        This is what makes /resume work — each session rebinds its
        GoalManager and picks up the saved state.
        """
        from hermes_cli.goals import GoalManager

        mgr1 = GoalManager(session_id="persist-sid")
        mgr1.set("do the thing")

        mgr2 = GoalManager(session_id="persist-sid")
        assert mgr2.state is not None
        assert mgr2.state.goal == "do the thing"
        assert mgr2.is_active()

    def test_evaluate_after_turn_done(self, hermes_home):
        """Judge says done → status=done, no continuation."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-1")
        mgr.set("ship it")

        with patch.object(goals, "judge_goal", return_value=("done", "shipped")):
            decision = mgr.evaluate_after_turn("I shipped the feature.")

        assert decision["verdict"] == "done"
        assert decision["should_continue"] is False
        assert decision["continuation_prompt"] is None
        assert mgr.state.status == "done"
        assert mgr.state.turns_used == 1

    def test_evaluate_after_turn_continue_under_budget(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-2", default_max_turns=5)
        mgr.set("a long goal")

        with patch.object(goals, "judge_goal", return_value=("continue", "more work")):
            decision = mgr.evaluate_after_turn("made some progress")

        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert decision["continuation_prompt"] is not None
        assert "a long goal" in decision["continuation_prompt"]
        assert mgr.state.status == "active"
        assert mgr.state.turns_used == 1

    def test_evaluate_after_turn_budget_exhausted(self, hermes_home):
        """When turn budget hits ceiling, auto-pause instead of continuing."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-3", default_max_turns=2)
        mgr.set("hard goal")

        with patch.object(goals, "judge_goal", return_value=("continue", "not yet")):
            d1 = mgr.evaluate_after_turn("step 1")
            assert d1["should_continue"] is True
            assert mgr.state.turns_used == 1
            assert mgr.state.status == "active"

            d2 = mgr.evaluate_after_turn("step 2")
            # turns_used is now 2 which equals max_turns → paused
            assert d2["should_continue"] is False
            assert mgr.state.status == "paused"
            assert mgr.state.turns_used == 2
            assert "budget" in (mgr.state.paused_reason or "").lower()

    def test_evaluate_after_turn_inactive(self, hermes_home):
        """evaluate_after_turn is a no-op when goal isn't active."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-4")
        d = mgr.evaluate_after_turn("anything")
        assert d["verdict"] == "inactive"
        assert d["should_continue"] is False

        mgr.set("a goal")
        mgr.pause()
        d2 = mgr.evaluate_after_turn("anything")
        assert d2["verdict"] == "inactive"
        assert d2["should_continue"] is False

    def test_continuation_prompt_shape(self, hermes_home):
        """The continuation prompt must include the goal text verbatim —
        and must be safe to inject as a user-role message (prompt-cache
        invariants: no system-prompt mutation)."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="cont-sid")
        mgr.set("port goal command to hermes")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "port goal command to hermes" in prompt
        assert prompt.strip()  # non-empty

    def test_evaluate_after_turn_judge_unavailable_pauses(self, hermes_home):
        """When the judge API is unavailable (after retries), the goal pauses
        instead of continuing silently.  This is fail-closed behavior.

        Previously this was fail-open (always continue), which caused goals
        to run forever when the judge API was down.
        """
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-judge-unavail")
        mgr.set("build the widget")

        # Judge returns "judge_unavailable" after exhausting retries.
        resp = "I've finished building the widget. Goal is complete."
        with _patch_judge("judge_unavailable", "judge error after 3 attempts: APITimeoutError: timeout"):
            d = mgr.evaluate_after_turn(resp)
        # Goal is paused, not done — user must manually resume when judge is back
        assert d["verdict"] == "judge_unavailable"
        assert d["should_continue"] is False
        assert mgr.state.status == "paused"
        assert "judge error" in d["reason"]
        assert "/goal resume" in d["message"]

    def test_evaluate_after_turn_judge_unavailable_even_with_completion_text(self, hermes_home):
        """Even if the agent's response says 'goal complete', we don't trust
        it when the judge is unavailable.  Only the judge can confirm done.

        This prevents the agent from self-declaring victory without verification.
        """
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-judge-unavail-text")
        mgr.set("修复这个bug")

        resp = "所有代码已提交。目标已完成。"
        with _patch_judge("judge_unavailable", "judge error after 3 attempts: ConnectionError"):
            d = mgr.evaluate_after_turn(resp)
        # Paused — not done.  The agent saying "done" isn't enough.
        assert d["verdict"] == "judge_unavailable"
        assert d["should_continue"] is False
        assert mgr.state.status == "paused"

    def test_evaluate_after_turn_normal_continue(self, hermes_home):
        """Normal flow: judge says continue → goal stays active."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-cont")
        mgr.set("refactor the login handler")

        # Response is progress but NOT completion
        resp = "I've started refactoring the login handler. More work needed."
        # Judge returns continue with a real evaluation (not an error)
        with _patch_judge("continue", "still in progress"):
            d = mgr.evaluate_after_turn(resp)
        assert d["verdict"] == "continue"
        assert d["should_continue"] is True
        assert mgr.state.status == "active"

    def test_evaluate_after_turn_judge_done(self, hermes_home):
        """When the judge says done, the goal completes."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-judge-done")
        mgr.set("build the widget")

        resp = "I've finished building the widget. All tests pass."
        with _patch_judge("done", "goal requirements met"):
            d = mgr.evaluate_after_turn(resp)
        assert d["verdict"] == "done"
        assert d["should_continue"] is False
        assert mgr.state.status == "done"

    def test_evaluate_after_turn_judge_says_continue_overrides_text(self, hermes_home):
        """When the judge works and says continue, we trust it — even if the
        agent's response contains completion language.

        This is the core principle: the judge is authoritative, not the agent.
        """
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-judge-override")
        mgr.set("build the widget")

        # Agent says "goal complete" but judge says "still needs testing"
        resp = "I've finished building the widget. Goal is complete."
        with _patch_judge("continue", "still needs testing"):
            d = mgr.evaluate_after_turn(resp)
        # Judge's verdict wins
        assert d["verdict"] == "continue"
        assert d["should_continue"] is True
        assert mgr.state.status == "active"


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _patch_judge(verdict: str, reason: str):
    """Context manager that patches judge_goal to return a fixed verdict."""
    from unittest.mock import patch
    return patch(
        "hermes_cli.goals.judge_goal",
        return_value=(verdict, reason),
    )


# ──────────────────────────────────────────────────────────────────────
# Smoke: CommandDef is wired
# ──────────────────────────────────────────────────────────────────────


def test_goal_command_in_registry():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("goal")
    assert cmd is not None
    assert cmd.name == "goal"


def test_goal_command_dispatches_in_cli_registry_helpers():
    """goal shows up in autocomplete / help categories alongside other Session cmds."""
    from hermes_cli.commands import COMMANDS, COMMANDS_BY_CATEGORY

    assert "/goal" in COMMANDS
    session_cmds = COMMANDS_BY_CATEGORY.get("Session", {})
    assert "/goal" in session_cmds
