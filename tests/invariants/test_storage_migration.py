import hashlib
import json
from pathlib import Path
import time

import pytest

from scripts.storage_layout import (
    HOSTS,
    MigrationError,
    identity,
    make_plan,
    rename_noreplace,
    transact,
)


@pytest.fixture
def layout(tmp_path):
    home = tmp_path / "home"
    (home / "Models").mkdir(parents=True)
    old = home / "Datasets/pointstream-data"
    (old / "outputs").mkdir(parents=True)
    (old / "outputs/result.json").write_text('{"preserve": true}')
    return home


def ready(home):
    return {
        "home": str(home),
        "checked_at": time.time(),
        "hosts": [{"alias": host, "shared_visible": True, "busy": []} for host in HOSTS],
    }


def test_moves_preserve_inode_and_bytes_and_rollback(layout):
    source = layout / "Datasets/pointstream-data/outputs/result.json"
    original = identity(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = make_plan(layout)
    journal = layout / "Datasets/migration.jsonl"
    transact(plan, journal, ready(layout))
    target = layout / "Datasets/outputs/result.json"
    assert source.read_bytes() == target.read_bytes()
    assert source.stat().st_ino == original["ino"]
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    transact(plan, journal, ready(layout))  # acknowledgement retry, never replay
    transact(plan, journal, ready(layout), rollback=True)
    assert not source.parent.is_symlink()
    assert not target.exists()
    assert source.stat().st_ino == original["ino"]


def test_unreachable_busy_or_stale_checks_block_every_move(layout):
    plan = make_plan(layout)
    source = Path(plan["moves"][0]["source"])
    for mutation in ("unreachable", "busy", "stale", "missing_host"):
        report = ready(layout)
        if mutation == "unreachable":
            report["hosts"][0]["shared_visible"] = False
        elif mutation == "busy":
            report["hosts"][0]["busy"] = [{"pid": 12, "name": "python"}]
        elif mutation == "missing_host":
            report["hosts"].pop()
        else:
            report["checked_at"] -= 61
        with pytest.raises(MigrationError):
            transact(plan, layout / "Datasets/migration.jsonl", report)
        assert source.is_dir() and not source.is_symlink()


def test_collision_and_concurrent_destination_are_never_overwritten(layout):
    target = layout / "Datasets/outputs"
    plan = make_plan(layout)
    target.mkdir()
    (target / "owned.txt").write_text("someone else")
    with pytest.raises(MigrationError, match="destination appeared"):
        transact(plan, layout / "Datasets/migration.jsonl", ready(layout))
    assert (target / "owned.txt").read_text() == "someone else"
    assert make_plan(layout)["conflicts"] == []  # merge plans only missing children
    collision = target / "result.json"
    collision.write_text("different")
    conflicted = make_plan(layout)
    assert conflicted["conflicts"]
    with pytest.raises(MigrationError, match="collision"):
        transact(conflicted, layout / "Datasets/migration2.jsonl", ready(layout))
    assert collision.read_text() == "different"


def test_disconnect_after_move_reconciles_only_recorded_identity(layout):
    plan = make_plan(layout)
    journal = layout / "Datasets/migration.jsonl"
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    journal.write_text(json.dumps({"event": "intent", "index": 0, "plan_sha256": digest}) + "\n")
    move = plan["moves"][0]
    source, target = Path(move["source"]), Path(move["target"])
    rename_noreplace(source, target)
    transact(plan, journal, ready(layout))
    assert source.is_symlink() and source.resolve() == target
    assert identity(target)["ino"] == move["identity"]["ino"]


def test_ambiguous_launch_identity_and_existing_lock_require_attention(layout):
    plan = make_plan(layout)
    journal = layout / "Datasets/migration.jsonl"
    lock = journal.with_suffix(".jsonl.lock")
    lock.mkdir()
    with pytest.raises(FileExistsError):
        transact(plan, journal, ready(layout))
    assert lock.exists()
    lock.rmdir()
    source = Path(plan["moves"][0]["source"])
    (source / "new.txt").write_text("changed")
    with pytest.raises(MigrationError, match="source changed"):
        transact(plan, journal, ready(layout))


def test_native_no_clobber_rename_protects_existing_file(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_text("source")
    target.write_text("target")
    with pytest.raises(OSError):
        rename_noreplace(source, target)
    assert source.read_text() == "source"
    assert target.read_text() == "target"


def test_model_repo_and_checkpoints_share_family_without_overwriting(layout):
    old = layout / "Datasets/pointstream-data"
    repo = old / "third_party/HOPformer"
    (repo / ".git").mkdir(parents=True)
    (repo / "README.md").write_text("model code")
    weights = old / "weights/hopformer"
    weights.mkdir(parents=True)
    (weights / "epoch.ckpt").write_bytes(b"weights")
    plan = make_plan(layout)
    assert not plan["conflicts"]
    transact(plan, layout / "Datasets/migration.jsonl", ready(layout))
    assert (layout / "Models/HOPformer/README.md").read_text() == "model code"
    assert (layout / "Models/HOPformer/epoch.ckpt").read_bytes() == b"weights"
    assert (weights / "epoch.ckpt").resolve() == layout / "Models/HOPformer/epoch.ckpt"


def test_rollback_wont_remove_an_alias_replaced_by_someone_else(layout):
    plan = make_plan(layout)
    journal = layout / "Datasets/migration.jsonl"
    transact(plan, journal, ready(layout))
    source = Path(plan["moves"][0]["source"])
    source.unlink()
    source.mkdir()
    (source / "new.txt").write_text("keep")
    with pytest.raises(MigrationError, match="source was recreated"):
        transact(plan, journal, ready(layout), rollback=True)
    assert (source / "new.txt").read_text() == "keep"
