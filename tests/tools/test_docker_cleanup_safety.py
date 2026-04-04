"""Tests for Docker cleanup no longer using shell=True.

Verifies that DockerEnvironment.cleanup() uses list-based subprocess
calls (via a background thread) instead of shell=True, conforming to
the Hermes Agent contribution guide's security requirements.
"""

import subprocess
import threading
from unittest.mock import patch, MagicMock, call

import pytest


class TestDockerCleanupNoShell:
    """Verify cleanup() uses list-based subprocess calls, not shell=True."""

    def _make_env(self):
        """Create a minimal DockerEnvironment-like object for testing cleanup."""
        from types import SimpleNamespace

        env = SimpleNamespace()
        env._container_id = "abc123deadbeef"
        env._docker_exe = "/usr/bin/docker"
        env._persistent = False
        env._workspace_dir = None
        env._home_dir = None

        # Bind the real cleanup method
        from tools.environments.docker import DockerEnvironment
        env.cleanup = DockerEnvironment.cleanup.__get__(env, type(env))
        return env

    @patch("subprocess.run")
    def test_cleanup_uses_list_args(self, mock_run):
        """cleanup() must use list-based subprocess.run, not shell=True."""
        mock_run.return_value = MagicMock(returncode=0)
        env = self._make_env()

        env.cleanup()

        # Wait a bit for the background thread
        import time
        time.sleep(0.5)

        # Verify subprocess.run was called with list args (no shell=True)
        for c in mock_run.call_args_list:
            args, kwargs = c
            # First positional arg should be a list
            cmd = args[0] if args else kwargs.get("args")
            assert isinstance(cmd, list), (
                f"cleanup() called subprocess with non-list args: {cmd}"
            )
            assert kwargs.get("shell") is not True, (
                "cleanup() must not use shell=True"
            )

    @patch("subprocess.run")
    def test_cleanup_clears_container_id(self, mock_run):
        """cleanup() must set _container_id to None."""
        mock_run.return_value = MagicMock(returncode=0)
        env = self._make_env()

        env.cleanup()

        assert env._container_id is None

    @patch("subprocess.run")
    def test_cleanup_calls_docker_stop(self, mock_run):
        """cleanup() should call 'docker stop <container_id>'."""
        mock_run.return_value = MagicMock(returncode=0)
        env = self._make_env()

        env.cleanup()

        import time
        time.sleep(0.5)

        # Find the stop call
        stop_calls = [
            c for c in mock_run.call_args_list
            if "stop" in (c[0][0] if c[0] else [])
        ]
        assert len(stop_calls) >= 1, "cleanup() should call docker stop"

    @patch("subprocess.run")
    def test_cleanup_nonpersistent_calls_rm(self, mock_run):
        """Non-persistent cleanup should also call 'docker rm -f'."""
        mock_run.return_value = MagicMock(returncode=0)
        env = self._make_env()
        env._persistent = False

        env.cleanup()

        import time
        time.sleep(0.5)

        rm_calls = [
            c for c in mock_run.call_args_list
            if "rm" in (c[0][0] if c[0] else [])
        ]
        assert len(rm_calls) >= 1, (
            "Non-persistent cleanup should call docker rm"
        )

    @patch("subprocess.run")
    def test_cleanup_persistent_skips_rm(self, mock_run):
        """Persistent cleanup should NOT call 'docker rm'."""
        mock_run.return_value = MagicMock(returncode=0)
        env = self._make_env()
        env._persistent = True

        env.cleanup()

        import time
        time.sleep(0.5)

        rm_calls = [
            c for c in mock_run.call_args_list
            if "rm" in (c[0][0] if c[0] else [])
        ]
        assert len(rm_calls) == 0, (
            "Persistent cleanup should skip docker rm"
        )

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker stop", 60))
    def test_cleanup_handles_stop_timeout(self, mock_run):
        """If docker stop times out, cleanup should force rm and not crash."""
        env = self._make_env()

        # Should not raise
        env.cleanup()

        import time
        time.sleep(0.5)

        assert env._container_id is None
