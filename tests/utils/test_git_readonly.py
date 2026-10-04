"""Provenance Git calls must not rewrite a shared checkout's index."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.utils.git_readonly import (
    readonly_git_command,
    readonly_git_diff_command,
    readonly_git_env,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _stale_repo(root: Path) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    tracked = root / "tracked.txt"
    tracked.write_text("same content\n")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "init")
    index = root / ".git" / "index"
    old = index.stat().st_mtime - 100
    os.utime(index, (old, old))
    # Content unchanged, stat data changed: what another host leaves behind.
    later = old + 50
    os.utime(tracked, (later, later))
    return index


@pytest.mark.parametrize(
    "args",
    [("status", "--porcelain"), ("diff-index", "-p", "--no-color", "HEAD", "--"),
     ("rev-parse", "HEAD")],
)
def test_readonly_git_leaves_a_stale_index_untouched(tmp_path: Path, args: tuple[str, ...]) -> None:
    index = _stale_repo(tmp_path)
    before = index.stat().st_mtime_ns
    result = subprocess.run(
        readonly_git_command(tmp_path, *args),
        env=readonly_git_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert index.stat().st_mtime_ns == before
    if args[0] in ("status", "diff-index"):
        assert result.stdout == ""


def test_plain_status_would_rewrite_it(tmp_path: Path) -> None:
    """The mechanism the helper avoids, so the guard above is not vacuous."""
    index = _stale_repo(tmp_path)
    before = index.stat().st_mtime_ns
    subprocess.run(["git", "-C", str(tmp_path), "status", "--porcelain"], check=True,
                   capture_output=True)
    assert index.stat().st_mtime_ns != before


def test_env_keeps_the_base_environment() -> None:
    env = readonly_git_env({"PATH": "/usr/bin", "GIT_OPTIONAL_LOCKS": "1"})
    assert env == {"PATH": "/usr/bin", "GIT_OPTIONAL_LOCKS": "0"}


def test_diff_command_reports_real_changes(tmp_path: Path) -> None:
    _stale_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("changed\n")
    patch = subprocess.run(readonly_git_diff_command(tmp_path), env=readonly_git_env(),
                           check=True, capture_output=True, text=True).stdout
    assert "-same content" in patch and "+changed" in patch
