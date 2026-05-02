"""Tests for get_hermes_home() profile-mode fallback safety.

Validates that when HERMES_HOME is unset but an active_profile file exists
(indicating profile mode), get_hermes_home() raises ValueError instead of
silently falling back to ~/.hermes — which would cause cross-profile data
corruption.

Regression test for https://github.com/NousResearch/hermes-agent/issues/18594
"""

import os
from pathlib import Path

import pytest


class TestGetHermesHomeProfileFallback:
    """get_hermes_home() should raise in profile mode when HERMES_HOME is unset."""

    def test_classic_mode_no_active_profile_returns_default(self, tmp_path, monkeypatch):
        """Classic mode: no HERMES_HOME, no active_profile → returns ~/.hermes."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # No active_profile file exists — classic mode
        from hermes_constants import get_hermes_home
        result = get_hermes_home()
        assert result == tmp_path / ".hermes"

    def test_profile_mode_raises_when_hermes_home_unset(self, tmp_path, monkeypatch):
        """Profile mode: active_profile exists but HERMES_HOME is unset → ValueError."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Create active_profile file indicating a named profile is active
        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()
        (hermes_dir / "active_profile").write_text("coder\n")
        from hermes_constants import get_hermes_home
        with pytest.raises(ValueError, match="HERMES_HOME.*profile"):
            get_hermes_home()

    def test_profile_mode_default_profile_returns_ok(self, tmp_path, monkeypatch):
        """Profile mode with 'default' active_profile → returns ~/.hermes (safe)."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()
        (hermes_dir / "active_profile").write_text("default\n")
        from hermes_constants import get_hermes_home
        result = get_hermes_home()
        assert result == tmp_path / ".hermes"

    def test_hermes_home_env_set_returns_env_value(self, tmp_path, monkeypatch):
        """When HERMES_HOME is set, returns it regardless of active_profile."""
        profile_dir = tmp_path / "profiles" / "coder"
        profile_dir.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        from hermes_constants import get_hermes_home
        result = get_hermes_home()
        assert result == profile_dir

    def test_profile_mode_empty_active_profile_returns_ok(self, tmp_path, monkeypatch):
        """Empty active_profile file → treated as classic mode, returns default."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()
        (hermes_dir / "active_profile").write_text("")
        from hermes_constants import get_hermes_home
        result = get_hermes_home()
        assert result == tmp_path / ".hermes"
