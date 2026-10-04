"""A canonical client can dispatch a scoped commit without touching its checkout."""
import subprocess
import tarfile

import pytest

from experiments.jobs import fleet, inbox


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def repository(path):
    path.mkdir()
    git(path, "init", "--initial-branch=main")
    (path / "runner.py").write_text("original\n")
    git(path, "add", "runner.py")
    git(path, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-m", "seed")
    return path


def test_scoped_snapshot_uses_worktree_commit_and_preserves_canonical_dirty_file(tmp_path):
    root = repository(tmp_path / "canonical")
    scoped = tmp_path / "scoped"
    git(root, "worktree", "add", "-b", "codex/fixture", str(scoped))
    (scoped / "runner.py").write_text("reviewed\n")
    git(scoped, "add", "runner.py")
    git(scoped, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-m", "fix")
    (root / "runner.py").write_text("unrelated dirty work\n")
    selected = inbox.snapshot_root(root, scoped)
    archive, _, metadata = fleet._build_snapshot(selected)
    try:
        with tarfile.open(archive) as stream:
            assert stream.extractfile("runner.py").read() == b"reviewed\n"
        assert metadata["git_head"] != fleet._git(root, "rev-parse", "HEAD").stdout.decode().strip()
        assert (root / "runner.py").read_text() == "unrelated dirty work\n"
    finally:
        archive.unlink()


def test_source_worktree_rejects_another_repository_and_subdirectories(tmp_path):
    root = repository(tmp_path / "canonical")
    other = repository(tmp_path / "other")
    child = root / "child"
    child.mkdir()
    for selected in (other, child):
        with pytest.raises(fleet.FleetError, match="same repository"):
            inbox.snapshot_root(root, selected)


def test_default_snapshot_keeps_canonical_root(tmp_path):
    assert inbox.snapshot_root(tmp_path, None) == tmp_path


def test_selected_archive_records_scope_and_omits_unneeded_files(tmp_path):
    root = repository(tmp_path / "repo")
    (root / "irrelevant.txt").write_text("large unrelated context\n")
    git(root, "add", "irrelevant.txt")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-m", "context")
    archive, _, metadata = fleet._build_snapshot(root, include_paths=("runner.py",))
    try:
        with tarfile.open(archive) as stream:
            assert stream.getnames() == ["runner.py"]
        assert metadata["tracked_archive_paths_selected"] == ["runner.py"]
    finally:
        archive.unlink()
    for invalid in ("../repo", "/tmp", ":(glob)**", "demo/outputs", "."):
        with pytest.raises(fleet.FleetError):
            fleet._build_snapshot(root, include_paths=(invalid,))
