"""Tests for emergency compression before max_iterations exhaustion.

Covers the three reviewer asks:
1. Unit test for IterationBudget.reset()
2. Integration test showing emergency compression fires and resets budget
3. Boundary test verifying the cap at 3 emergency compressions
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent, IterationBudget
import run_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None, usage=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = SimpleNamespace(**usage) if usage else None
    return resp


def _tool_call(id: str, name: str = "web_search", args: str = "{}"):
    return SimpleNamespace(
        id=id,
        function=SimpleNamespace(name=name, arguments=args),
        type="function",
    )


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


# ---------------------------------------------------------------------------
# Common agent fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=2,  # Tiny budget so emergency fires after 1 iteration
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = True
        a.save_trajectories = False
        a.context_compressor.threshold_tokens = 100
        return a


# ---------------------------------------------------------------------------
# Task 1: Unit test for IterationBudget.reset()
# ---------------------------------------------------------------------------

class TestIterationBudgetReset:
    """IterationBudget.reset() should correctly reset state mid-consumption."""

    def test_reset_after_partial_consumption(self):
        budget = IterationBudget(max_total=10)
        for _ in range(3):
            assert budget.consume() is True
        assert budget.used == 3
        assert budget.remaining == 7
        budget.reset()
        assert budget.used == 0
        assert budget.remaining == 10

    def test_reset_when_full(self):
        budget = IterationBudget(max_total=10)
        budget.reset()
        assert budget.used == 0
        assert budget.remaining == 10

    def test_reset_after_full_exhaustion(self):
        budget = IterationBudget(max_total=3)
        for _ in range(3):
            budget.consume()
        assert budget.consume() is False
        assert budget.used == 3
        assert budget.remaining == 0
        budget.reset()
        assert budget.used == 0
        assert budget.remaining == 3
        assert budget.consume() is True

    def test_reset_is_thread_safe(self):
        import threading
        budget = IterationBudget(max_total=100)
        def consumer():
            for _ in range(50):
                budget.consume()
        def reseter():
            for _ in range(5):
                budget.reset()
        threads = [
            threading.Thread(target=consumer),
            threading.Thread(target=reseter),
            threading.Thread(target=consumer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert 0 <= budget.used <= 100
        assert budget.remaining == 100 - budget.used


# ---------------------------------------------------------------------------
# Task 2: Integration test — emergency compression fires and resets budget
# ---------------------------------------------------------------------------

class TestEmergencyCompressionTriggers:
    """Emergency compression fires when budget is nearly exhausted
    and context is still large, then resets budget so agent continues."""

    def test_emergency_fires_and_resets_budget(self, agent):
        """First call returns tool_calls (consumes budget), then emergency
        fires, budget resets, and a stop call completes the conversation."""
        tool_resp = _mock_response(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[_tool_call("tc_1")],
        )
        stop_resp = _mock_response(content="Answer", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [tool_resp, stop_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.estimate_messages_tokens_rough", return_value=500),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello")

        # Emergency compression fired exactly once
        assert agent._emergency_compression_count == 1

        # Budget was reset (2) then consumed once by the final call (1)
        assert agent.iteration_budget.remaining == 1

        # Conversation completed
        assert result["completed"] is True
        assert result["final_response"] == "Answer"

    def test_agent_continues_after_emergency(self, agent):
        """After emergency compression resets budget, the agent continues
        making progress (remaining budget is consumed normally)."""
        tool_resp = _mock_response(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[_tool_call("tc_1")],
        )
        stop_resp = _mock_response(content="Done", finish_reason="stop")
        # Pattern: 1 tool_call to consume budget, then stop to finish
        agent.client.chat.completions.create.side_effect = [tool_resp, stop_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.estimate_messages_tokens_rough", return_value=500),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello")

        assert agent._emergency_compression_count == 1
        assert agent.iteration_budget.remaining == 1
        assert result["completed"] is True
        assert result["final_response"] == "Done"


# ---------------------------------------------------------------------------
# Task 3: Boundary test — cap at 3 emergency compressions
# ---------------------------------------------------------------------------

class TestEmergencyCompressionCap:
    """After 3 emergency compressions, the 4th exhaustion does NOT trigger
    another compression — the agent hits max_iterations."""

    def test_cap_at_three(self, agent):
        """Compression fires at most 3 times; 4th exhaustion hits max_iterations."""
        tool_resp = _mock_response(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[_tool_call("tc_1")],
        )
        # Each cycle: 2 tool_calls (one to consume, then emg check fires).
        # 3 cycles = 6, then 2 fallthrough = 8 total.
        agent.client.chat.completions.create.side_effect = [
            tool_resp, tool_resp,  # cycle 1
            tool_resp, tool_resp,  # cycle 2
            tool_resp, tool_resp,  # cycle 3
            tool_resp,             # fallthrough iter 1
            tool_resp,             # fallthrough iter 2 → max_iterations
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.estimate_messages_tokens_rough", return_value=500),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello")

        # Exactly 3 emergency compressions (the cap)
        assert agent._emergency_compression_count == 3

        # After cap reached, agent hits max_iterations
        assert result["completed"] is False
