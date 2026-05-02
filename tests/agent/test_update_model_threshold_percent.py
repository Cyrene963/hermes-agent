"""Tests for ContextCompressor.update_model() threshold_percent handling.

Regression test for #18617: threshold_percent not updated on model switch.
"""
import pytest
from unittest.mock import patch


def _make_compressor(model="model-a", threshold_percent=0.70, context_length=200000):
    """Create a ContextCompressor with controlled context length."""
    from agent.context_compressor import ContextCompressor
    with patch("agent.context_compressor.get_model_context_length", return_value=context_length):
        return ContextCompressor(
            model=model,
            threshold_percent=threshold_percent,
            quiet_mode=True,
        )


class TestUpdateModelThresholdPercent:
    """update_model() should accept and apply threshold_percent."""

    def test_update_model_preserves_threshold_percent_when_not_passed(self):
        """When threshold_percent is not passed, the existing value is kept."""
        cc = _make_compressor(threshold_percent=0.70)
        assert cc.threshold_percent == 0.70

        cc.update_model(model="model-b", context_length=100000)
        # threshold_percent unchanged
        assert cc.threshold_percent == 0.70
        # threshold_tokens recalculated from new context_length
        # 100000 * 0.70 = 70000 > MINIMUM_CONTEXT_LENGTH (64000)
        assert cc.threshold_tokens == 70000

    def test_update_model_accepts_threshold_percent_parameter(self):
        """update_model() should accept threshold_percent and update it."""
        cc = _make_compressor(threshold_percent=0.70)
        cc.update_model(model="model-b", context_length=100000, threshold_percent=0.50)
        assert cc.threshold_percent == 0.50

    def test_update_model_recalculates_threshold_tokens_with_new_percent(self):
        """After threshold_percent update, threshold_tokens uses the new value."""
        cc = _make_compressor(threshold_percent=0.70, context_length=200000)
        assert cc.threshold_percent == 0.70
        # 200000 * 0.70 = 140000
        assert cc.threshold_tokens == 140000

        cc.update_model(model="model-b", context_length=100000, threshold_percent=0.80)
        # 100000 * 0.80 = 80000 > MINIMUM_CONTEXT_LENGTH
        assert cc.threshold_percent == 0.80
        assert cc.threshold_tokens == 80000

    def test_update_model_preserves_percent_on_context_change_only(self):
        """Model switch with same percent should still recalculate tokens."""
        cc = _make_compressor(threshold_percent=0.60, context_length=200000)
        assert cc.threshold_tokens == 120000

        cc.update_model(model="model-b", context_length=300000)
        assert cc.threshold_percent == 0.60
        # 300000 * 0.60 = 180000
        assert cc.threshold_tokens == 180000

    def test_threshold_floor_still_applies(self):
        """MINIMUM_CONTEXT_LENGTH floor should still apply after percent update."""
        cc = _make_compressor(threshold_percent=0.70, context_length=200000)
        # Switch to small model with low percent that would be below floor
        cc.update_model(model="small-model", context_length=80000, threshold_percent=0.50)
        # 80000 * 0.50 = 40000 < 64000 (MINIMUM_CONTEXT_LENGTH)
        # Floor should kick in
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        assert cc.threshold_tokens == MINIMUM_CONTEXT_LENGTH
