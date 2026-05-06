"""Tests for the update check mechanism in hermes_cli.banner."""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_version_string_no_v_prefix():
    """__version__ should be bare semver without a 'v' prefix."""
    from hermes_cli import __version__
    assert not __version__.startswith("v"), f"__version__ should not start with 'v', got {__version__!r}"


def test_check_for_updates_uses_cache(tmp_path, monkeypatch):
    """When cache is fresh AND HEAD-bound, check_for_updates returns cached value without git fetch."""
    import hermes_cli.banner as banner

    # Create a fake git repo and fresh cache bound to the current HEAD
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({
        "ts": time.time(), "behind": 3, "rev": None, "local_head": "abc12345",
    }))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(banner, "_current_local_head_short", lambda: "abc12345")
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        result = banner.check_for_updates()

    assert result == 3
    mock_run.assert_not_called()


def test_check_for_updates_legacy_cache_invalidated(tmp_path, monkeypatch):
    """Caches without a ``local_head`` field are treated as stale so the
    HEAD-binding contract takes effect on the very next read."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    # Legacy format — no local_head field
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({"ts": time.time(), "behind": 3, "rev": None}))

    mock_result = MagicMock(returncode=0, stdout="7\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(banner, "_current_local_head_short", lambda: "deadbeef")
    with patch("hermes_cli.banner.subprocess.run", return_value=mock_result) as mock_run:
        result = banner.check_for_updates()

    assert result == 7
    assert mock_run.call_count >= 1  # fetch + rev-list (+ rev-parse for new HEAD field)


def test_check_for_updates_invalidates_when_head_moved(tmp_path, monkeypatch):
    """A `git pull` outside `hermes update` moves HEAD without deleting the
    cache file. The HEAD-bound cache must reject the stale entry on the
    next read so the banner reflects the new commit."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({
        "ts": time.time(), "behind": 5, "rev": None, "local_head": "OLDHEAD1",
    }))

    mock_result = MagicMock(returncode=0, stdout="0\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(banner, "_current_local_head_short", lambda: "NEWHEAD2")
    with patch("hermes_cli.banner.subprocess.run", return_value=mock_result) as mock_run:
        result = banner.check_for_updates()

    assert result == 0  # post-update count, not the stale `5`
    assert mock_run.call_count >= 1


def test_check_for_updates_writes_local_head_to_cache(tmp_path, monkeypatch):
    """A fresh check writes the HEAD SHA into the cache so subsequent reads
    can confirm the cache still matches the current commit."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    cache_file = tmp_path / ".update_check"

    # subprocess.run is hit three times: fetch, rev-list, rev-parse(HEAD)
    def _fake_run(cmd, *args, **kwargs):
        if "rev-list" in cmd:
            return MagicMock(returncode=0, stdout="2\n")
        if "rev-parse" in cmd:
            return MagicMock(returncode=0, stdout="ABCDEF12\n")
        return MagicMock(returncode=0, stdout="")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run", side_effect=_fake_run):
        result = banner.check_for_updates()

    assert result == 2
    cached = json.loads(cache_file.read_text())
    assert cached["local_head"] == "ABCDEF12"
    assert cached["behind"] == 2


def test_check_for_updates_expired_cache(tmp_path, monkeypatch):
    """When cache is expired, check_for_updates should call git fetch."""
    from hermes_cli.banner import check_for_updates

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({
        "ts": 0, "behind": 1, "rev": None, "local_head": "abc12345",
    }))

    mock_result = MagicMock(returncode=0, stdout="5\n")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run", return_value=mock_result) as mock_run:
        result = check_for_updates()

    assert result == 5
    # fetch + rev-list + rev-parse(HEAD) — last one was added with HEAD-binding
    assert mock_run.call_count == 3


def test_check_for_updates_no_git_dir(tmp_path, monkeypatch):
    """Returns None when .git directory doesn't exist anywhere."""
    import hermes_cli.banner as banner

    # Create a fake banner.py so the fallback path also has no .git
    fake_banner = tmp_path / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()

    monkeypatch.setattr(banner, "__file__", str(fake_banner))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        result = banner.check_for_updates()
    assert result is None
    mock_run.assert_not_called()


def test_check_for_updates_fallback_to_project_root(tmp_path, monkeypatch):
    """Dev install: falls back to Path(__file__).parent.parent when HERMES_HOME has no git repo."""
    import hermes_cli.banner as banner

    project_root = Path(banner.__file__).parent.parent.resolve()
    if not (project_root / ".git").exists():
        pytest.skip("Not running from a git checkout")

    # Point HERMES_HOME at a temp dir with no hermes-agent/.git
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="0\n")
        result = banner.check_for_updates()
    # Should have fallen back to project root and run git commands
    assert mock_run.call_count >= 1


def test_prefetch_non_blocking():
    """prefetch_update_check() should return immediately without blocking."""
    import hermes_cli.banner as banner

    # Reset module state
    banner._update_result = None
    banner._update_check_done = threading.Event()
    banner._update_result_head = None

    with patch.object(banner, "check_for_updates", return_value=5), \
         patch.object(banner, "_current_local_head_short", return_value="HEAD0001"):
        start = time.monotonic()
        banner.prefetch_update_check()
        elapsed = time.monotonic() - start

        # Should return almost immediately (well under 1 second)
        assert elapsed < 1.0

        # Wait for the background thread to finish
        banner._update_check_done.wait(timeout=5)
        assert banner._update_result == 5
        assert banner._update_result_head == "HEAD0001"


def test_get_update_result_refreshes_after_head_moves():
    """Long-running TUI/gateway: when local HEAD has moved since the
    prefetch (e.g. ``hermes update`` ran in a sibling process), the next
    ``get_update_result`` call should kick a non-blocking refresh so
    later renders pick up the new value. The refresh runs in a daemon
    thread on purpose so the caller is never blocked."""
    import hermes_cli.banner as banner

    # Seed module state as if a prefetch ran when behind=2982 at HEAD=OLD
    banner._update_result = 2982
    banner._update_check_done = threading.Event()
    banner._update_check_done.set()
    banner._update_result_head = "OLDHEAD1"
    banner._update_refresh_lock = threading.Lock()

    release_check = threading.Event()
    refresh_done = threading.Event()
    refresh_calls = []

    def _fake_check():
        # Hold the refresh thread until the test releases it so we can
        # observe both the pre-refresh return value and the post-refresh
        # state without racing.
        release_check.wait(timeout=5)
        refresh_calls.append(time.time())
        try:
            return 0
        finally:
            refresh_done.set()

    with patch.object(banner, "check_for_updates", side_effect=_fake_check), \
         patch.object(banner, "_current_local_head_short", return_value="NEWHEAD2"):
        # Caller returns immediately with the stale snapshot
        first = banner.get_update_result(timeout=0.0)
        assert first == 2982

        # Now let the daemon refresh complete and observe the new value
        release_check.set()
        assert refresh_done.wait(timeout=5)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if banner._update_result == 0:
                break
            time.sleep(0.02)

    assert banner._update_result == 0
    assert banner._update_result_head == "NEWHEAD2"
    assert len(refresh_calls) == 1


def test_get_update_result_does_not_block_caller():
    """The refresh path must not block the caller — even if the underlying
    update check stalls, ``get_update_result`` returns the cached snapshot
    in well under a second."""
    import hermes_cli.banner as banner

    banner._update_result = 99
    banner._update_check_done = threading.Event()
    banner._update_check_done.set()
    banner._update_result_head = "OLDHEAD1"
    banner._update_refresh_lock = threading.Lock()

    block_forever = threading.Event()
    try:
        with patch.object(
            banner, "check_for_updates",
            side_effect=lambda: block_forever.wait(timeout=10) or 0,
        ), patch.object(banner, "_current_local_head_short", return_value="NEWHEAD2"):
            start = time.monotonic()
            result = banner.get_update_result(timeout=0.0)
            elapsed = time.monotonic() - start

        assert result == 99
        assert elapsed < 0.5
    finally:
        # Let the daemon thread exit cleanly so it doesn't leak between tests
        block_forever.set()


def test_get_update_result_no_refresh_when_head_unchanged():
    """When HEAD hasn't moved, get_update_result must not spawn a refresh
    thread — the cached value is still bound to the right commit."""
    import hermes_cli.banner as banner

    banner._update_result = 7
    banner._update_check_done = threading.Event()
    banner._update_check_done.set()
    banner._update_result_head = "STABLE01"
    banner._update_refresh_lock = threading.Lock()

    with patch.object(banner, "check_for_updates") as mock_check, \
         patch.object(banner, "_current_local_head_short", return_value="STABLE01"):
        result = banner.get_update_result(timeout=0.0)

    assert result == 7
    mock_check.assert_not_called()


def test_invalidate_update_cache_clears_all_profiles(tmp_path):
    """_invalidate_update_cache() should delete .update_check from ALL profiles."""
    from hermes_cli.main import _invalidate_update_cache

    # Build a fake ~/.hermes with default + two named profiles
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / ".update_check").write_text('{"ts":1,"behind":50}')

    profiles_root = default_home / "profiles"
    for name in ("ops", "dev"):
        p = profiles_root / name
        p.mkdir(parents=True)
        (p / ".update_check").write_text('{"ts":1,"behind":50}')

    with patch.object(Path, "home", return_value=tmp_path), \
         patch.dict(os.environ, {"HERMES_HOME": str(default_home)}):
        _invalidate_update_cache()

    # All three caches should be gone
    assert not (default_home / ".update_check").exists(), "default profile cache not cleared"
    assert not (profiles_root / "ops" / ".update_check").exists(), "ops profile cache not cleared"
    assert not (profiles_root / "dev" / ".update_check").exists(), "dev profile cache not cleared"


def test_invalidate_update_cache_no_profiles_dir(tmp_path):
    """Works fine when no profiles directory exists (single-profile setup)."""
    from hermes_cli.main import _invalidate_update_cache

    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / ".update_check").write_text('{"ts":1,"behind":5}')

    with patch.object(Path, "home", return_value=tmp_path), \
         patch.dict(os.environ, {"HERMES_HOME": str(default_home)}):
        _invalidate_update_cache()

    assert not (default_home / ".update_check").exists()
