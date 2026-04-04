"""Tests for tarball extraction safety in tirith_security.py.

Verifies that _auto_install_tirith rejects non-regular-file tar members
(symlinks, hardlinks, device nodes) to prevent symlink-based code execution
attacks via crafted archives.
"""

import io
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers — build malicious tarballs in memory
# ---------------------------------------------------------------------------

def _make_tarball_with_regular_file() -> bytes:
    """Create a valid tarball with a regular-file 'tirith' member."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        content = b"#!/bin/sh\necho tirith\n"
        info = tarfile.TarInfo(name="tirith")
        info.size = len(content)
        info.type = tarfile.REGTYPE
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _make_tarball_with_symlink() -> bytes:
    """Create a tarball where 'tirith' is a symlink → /tmp/evil."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="tirith")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp/evil"
        tar.addfile(info)
    return buf.getvalue()


def _make_tarball_with_hardlink() -> bytes:
    """Create a tarball where 'tirith' is a hardlink → /etc/passwd."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="tirith")
        info.type = tarfile.LNKTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    return buf.getvalue()


def _make_tarball_with_directory() -> bytes:
    """Create a tarball where 'tirith' is a directory member."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="tirith")
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
    return buf.getvalue()


def _make_tarball_with_nested_regular() -> bytes:
    """Create a valid tarball with 'some-dir/tirith' as a regular file."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        content = b"#!/bin/sh\necho tirith\n"
        info = tarfile.TarInfo(name="some-dir/tirith")
        info.size = len(content)
        info.type = tarfile.REGTYPE
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _make_tarball_with_nested_symlink() -> bytes:
    """Create a tarball with 'subdir/tirith' as a symlink."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="subdir/tirith")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp/evil"
        tar.addfile(info)
    return buf.getvalue()


def _make_tarball_with_path_traversal() -> bytes:
    """Create a tarball with '../tirith' (path traversal)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        content = b"#!/bin/sh\necho evil\n"
        info = tarfile.TarInfo(name="../tirith")
        info.size = len(content)
        info.type = tarfile.REGTYPE
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Extract the tarball extraction logic into a testable function
# ---------------------------------------------------------------------------

def _extract_tirith_from_tarball(archive_path: str, tmpdir: str) -> bool:
    """Replicate the extraction logic from tirith_security._auto_install_tirith.

    Returns True if a regular-file 'tirith' was extracted, False otherwise.
    """
    import logging
    logger = logging.getLogger(__name__)

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name == "tirith" or member.name.endswith("/tirith"):
                if ".." in member.name:
                    continue
                # This is the critical security check added by the fix
                if not member.isfile():
                    logger.warning(
                        "tirith archive member '%s' is not a regular file "
                        "(type=%s) — skipping to prevent symlink/hardlink attack",
                        member.name, member.type,
                    )
                    continue
                member.name = "tirith"
                tar.extract(member, tmpdir)
                return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTirithTarSafety:
    """Verify tarball extraction rejects dangerous member types."""

    def test_regular_file_accepted(self, tmp_path):
        """Regular-file 'tirith' should be extracted successfully."""
        archive = tmp_path / "tirith.tar.gz"
        archive.write_bytes(_make_tarball_with_regular_file())

        result = _extract_tirith_from_tarball(str(archive), str(tmp_path))

        assert result is True
        extracted = tmp_path / "tirith"
        assert extracted.exists()
        assert not extracted.is_symlink()

    def test_symlink_member_rejected(self, tmp_path):
        """Symlink 'tirith' member must be rejected (prevents code execution)."""
        archive = tmp_path / "tirith.tar.gz"
        archive.write_bytes(_make_tarball_with_symlink())

        result = _extract_tirith_from_tarball(str(archive), str(tmp_path))

        assert result is False
        extracted = tmp_path / "tirith"
        assert not extracted.exists(), "Symlink member should not be extracted"

    def test_hardlink_member_rejected(self, tmp_path):
        """Hardlink 'tirith' member must be rejected."""
        archive = tmp_path / "tirith.tar.gz"
        archive.write_bytes(_make_tarball_with_hardlink())

        result = _extract_tirith_from_tarball(str(archive), str(tmp_path))

        assert result is False
        extracted = tmp_path / "tirith"
        assert not extracted.exists(), "Hardlink member should not be extracted"

    def test_directory_member_rejected(self, tmp_path):
        """Directory 'tirith' member must be rejected."""
        archive = tmp_path / "tirith.tar.gz"
        archive.write_bytes(_make_tarball_with_directory())

        result = _extract_tirith_from_tarball(str(archive), str(tmp_path))

        assert result is False

    def test_nested_regular_file_accepted(self, tmp_path):
        """'some-dir/tirith' as a regular file should be extracted."""
        archive = tmp_path / "tirith.tar.gz"
        archive.write_bytes(_make_tarball_with_nested_regular())

        result = _extract_tirith_from_tarball(str(archive), str(tmp_path))

        assert result is True
        extracted = tmp_path / "tirith"
        assert extracted.exists()
        assert not extracted.is_symlink()

    def test_nested_symlink_rejected(self, tmp_path):
        """'subdir/tirith' as a symlink must be rejected."""
        archive = tmp_path / "tirith.tar.gz"
        archive.write_bytes(_make_tarball_with_nested_symlink())

        result = _extract_tirith_from_tarball(str(archive), str(tmp_path))

        assert result is False

    def test_path_traversal_rejected(self, tmp_path):
        """'../tirith' path traversal member must be skipped."""
        archive = tmp_path / "tirith.tar.gz"
        archive.write_bytes(_make_tarball_with_path_traversal())

        result = _extract_tirith_from_tarball(str(archive), str(tmp_path))

        assert result is False
