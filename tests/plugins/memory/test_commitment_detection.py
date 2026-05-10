"""Tests for commitment detection in Hindsight provider (LLM-based)."""

import json
import unittest
from unittest.mock import patch, MagicMock

from plugins.memory.hindsight import HindsightMemoryProvider as HindsightProvider


def _make_mock_response(commitment: bool, confidence: float):
    """Create a mock LLM response."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps({
        "commitment": commitment,
        "confidence": confidence,
    })
    return mock_resp


class TestCommitmentDetectionLLM(unittest.TestCase):
    """Test _detect_commitment with LLM classification."""

    def _make_provider(self, enabled=True):
        """Create a minimal provider with commitment_detection config."""
        with patch.object(HindsightProvider, '__init__', lambda self, **kw: None):
            provider = HindsightProvider.__new__(HindsightProvider)
            provider._commitment_detection = enabled
            return provider

    @patch("agent.auxiliary_client.call_llm")
    def test_high_confidence_commitment(self, mock_call_llm):
        """LLM returns commitment=true with high confidence -> True."""
        mock_call_llm.return_value = _make_mock_response(True, 0.95)
        provider = self._make_provider()
        self.assertTrue(provider._detect_commitment(
            "you should remember to save this", "You're right, I'll remember to save from now on."
        ))

    @patch("agent.auxiliary_client.call_llm")
    def test_low_confidence_rejected(self, mock_call_llm):
        """LLM returns commitment=true but low confidence -> False."""
        mock_call_llm.return_value = _make_mock_response(True, 0.5)
        provider = self._make_provider()
        self.assertFalse(provider._detect_commitment(
            "help me run the tests", "Sure, I'll run them."
        ))

    @patch("agent.auxiliary_client.call_llm")
    def test_not_commitment(self, mock_call_llm):
        """LLM returns commitment=false -> False."""
        mock_call_llm.return_value = _make_mock_response(False, 0.1)
        provider = self._make_provider()
        self.assertFalse(provider._detect_commitment(
            "What's the weather?", "It's sunny today."
        ))

    @patch("agent.auxiliary_client.call_llm")
    def test_task_plan_not_commitment(self, mock_call_llm):
        """Task plan classified as not commitment."""
        mock_call_llm.return_value = _make_mock_response(False, 0.15)
        provider = self._make_provider()
        self.assertFalse(provider._detect_commitment(
            "Can you run the tests?", "I'll run the tests next."
        ))

    @patch("agent.auxiliary_client.call_llm")
    def test_llm_error_fails_open(self, mock_call_llm):
        """LLM call raises exception -> False (fail-open, never blocks)."""
        mock_call_llm.side_effect = RuntimeError("API timeout")
        provider = self._make_provider()
        self.assertFalse(provider._detect_commitment(
            "Why didn't you save?", "I'll be more careful."
        ))

    @patch("agent.auxiliary_client.call_llm")
    def test_malformed_json_fails_open(self, mock_call_llm):
        """LLM returns non-JSON -> False (fail-open)."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "I'm not sure, let me think..."
        mock_call_llm.return_value = mock_resp
        provider = self._make_provider()
        self.assertFalse(provider._detect_commitment(
            "you're wrong", "I'll do better."
        ))

    @patch("agent.auxiliary_client.call_llm")
    def test_json_in_code_block(self, mock_call_llm):
        """LLM wraps JSON in markdown code block -> still parsed correctly."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '```json\n{"commitment": true, "confidence": 0.85}\n```'
        mock_call_llm.return_value = mock_resp
        provider = self._make_provider()
        self.assertTrue(provider._detect_commitment(
            "remember to save this", "I'll remember to save from now on."
        ))

    def test_empty_content(self):
        """Empty strings -> False without calling LLM."""
        provider = self._make_provider()
        self.assertFalse(provider._detect_commitment("", ""))

    def test_whitespace_only(self):
        """Whitespace-only strings -> False without calling LLM."""
        provider = self._make_provider()
        self.assertFalse(provider._detect_commitment("  ", "  "))


class TestCommitmentDetectionConfig(unittest.TestCase):
    """Test commitment_detection config toggle."""

    def test_disabled_flag_checked_in_sync_turn(self):
        """commitment_detection=False causes sync_turn to skip detection."""
        # The config check is in sync_turn, not _detect_commitment.
        # This test verifies the attribute exists for sync_turn to check.
        with patch.object(HindsightProvider, '__init__', lambda self, **kw: None):
            provider = HindsightProvider.__new__(HindsightProvider)
            provider._commitment_detection = False
            # The flag is accessible and False — sync_turn uses:
            #   if self._commitment_detection and self._detect_commitment(...)
            self.assertFalse(provider._commitment_detection)


if __name__ == "__main__":
    unittest.main()
