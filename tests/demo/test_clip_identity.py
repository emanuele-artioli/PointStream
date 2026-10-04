"""Source selection must be explicit and stable before any model is loaded."""

import hashlib
import json

import pytest

from demo.experiments.clip_identity import CLIP_IDS, load_clip_manifest


def manifest(tmp_path):
    rows = []
    for clip_id in CLIP_IDS:
        source = tmp_path / f"{clip_id}.mp4"
        source.write_bytes(clip_id.encode())
        rows.append(
            {
                "clip_id": clip_id,
                "path": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(rows))
    return path, rows


def test_reordered_manifest_keeps_explicit_clip_bindings(tmp_path):
    path, rows = manifest(tmp_path)
    path.write_text(json.dumps(rows[::-1]))
    result = load_clip_manifest(path)
    assert tuple(clip.clip_id for clip in result.clips) == CLIP_IDS
    assert result.clips[0].path == tmp_path / "clip_01.mp4"
    assert result.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("fault", ["legacy", "duplicate", "missing", "wrong_hash", "same_source"])
def test_ambiguous_or_changed_source_manifest_rejected(tmp_path, fault):
    path, rows = manifest(tmp_path)
    if fault == "legacy":
        rows[0].pop("sha256")
    elif fault == "duplicate":
        rows[1]["clip_id"] = rows[0]["clip_id"]
    elif fault == "missing":
        rows.pop()
    elif fault == "wrong_hash":
        rows[0]["sha256"] = "0" * 64
    elif fault == "same_source":
        rows[1].update(path=rows[0]["path"], sha256=rows[0]["sha256"])
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        load_clip_manifest(path)


def test_source_change_after_selection_blocks_completion(tmp_path):
    path, _ = manifest(tmp_path)
    clip = load_clip_manifest(path).clips[0]
    clip.path.write_bytes(b"another source")
    with pytest.raises(ValueError, match="source identity mismatch"):
        clip.verify()


def test_outputs_cannot_replace_old_runs_or_write_inside_code(tmp_path):
    from demo.experiments.clip_identity import check_output_paths

    source = tmp_path / "pointstream"
    source.mkdir()
    saved = tmp_path / "Datasets" / "saved.json"
    saved.parent.mkdir()
    saved.write_text("immutable result")
    for path in (source / "outputs", saved):
        with pytest.raises(ValueError):
            check_output_paths((path,), source_root=source)
    assert saved.read_text() == "immutable result"
    check_output_paths((saved.parent / "new-run",), source_root=source)


def test_output_alias_cannot_hide_a_source_tree_write(tmp_path):
    from demo.experiments.clip_identity import check_output_paths

    source = tmp_path / "pointstream"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        check_output_paths((alias / "new-output",), source_root=source)
