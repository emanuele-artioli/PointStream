"""Recovery preserves completed evidence and merges bounded chunks at original shot boundaries."""
from __future__ import annotations

import argparse
import io
import json
import tarfile

import pytest

from experiments.tennis import k8, k8_recover as kr
from experiments.visor.b1 import file_sha256


def write_tar(path, files):
    with tarfile.open(path, "w") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def rows(start, end):
    return [{"frame": i, "registered": True, "chamfer_px": 1.0} for i in range(start, end)]


def test_chunks_partition_only_missing_frames_and_balance_groups(monkeypatch):
    monkeypatch.setattr(kr, "CHUNK", 3)
    manifest = {"missing": [[1, 4, 10], [3, 20, 24]]}
    items = kr.chunks(manifest)
    assert [f for c in items for f in range(c["start"], c["end"])] == list(range(4, 10)) + list(range(20, 24))
    assert all(c["end"] - c["start"] <= 3 for c in items)
    divided = kr.groups(items, 3)
    assert sorted(c["id"] for g in divided for c in g) == sorted(c["id"] for c in items)
    loads = [sum(c["end"] - c["start"] for c in group) for group in divided]
    assert max(loads) - min(loads) <= 3


def test_recovery_assembly_reuses_complete_shots_and_requires_all_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(kr, "CHUNK", 3)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"immutable source fixture")
    image_roles = ("first", "middle", "last")
    old = {"id": "shot-0000", "start": 0, "end": 4, "registration": rows(0, 4),
           "images": [f"images/shot-0000-{role}.jpg" for role in image_roles], **k8.shot_prediction(rows(0, 4))}
    partial = tmp_path / "partial.tar"
    write_tar(partial, {"publish/shots/shot-0000.json": json.dumps(old).encode(),
                        **{f"publish/{p}": b"old frame" for p in old["images"]}})
    old_hash = file_sha256(partial)
    manifest = {"base_report": {"frames": 10, "source_frames": 10, "fps": 10, "scene_threshold": .25,
                               "all_spans": [[0, 4], [4, 10]], "source": {"video_sha256": file_sha256(source)}},
                "completed_archives": [{"path": str(partial), "sha256": old_hash}], "missing": [[1, 4, 10]]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    recovered = []
    for item, roles in zip(kr.chunks(manifest), [("first",), ("middle", "last")]):
        data = {**item, "parent_shot": item["parent"], "registration": rows(item["start"], item["end"]),
                "extra_images": {role: f"images/shot-0001-{role}.jpg" for role in roles}, "conversion_command": []}
        recovered.append(data)
    report = {"chunks": recovered, "manifest_sha256": file_sha256(manifest_path), "job_dir": "fixture",
              "native_tools": {}, "registration_code_sha256": "r", "worker_code_sha256": "w", "runtime_seconds": 1}
    recovery = tmp_path / "published.tar"
    files = {"publish/recovery-report.json": json.dumps(report).encode()}
    files.update({f"publish/{name}": b"recovered frame" for c in recovered for name in c["extra_images"].values()})
    write_tar(recovery, files)
    args = argparse.Namespace(manifest=manifest_path, archive=[recovery], source=source, out=tmp_path / "review")
    assert kr.assemble(args) == 0
    final = json.loads((args.out / "k8-report.json").read_text())
    assert [s["id"] for s in final["shots"]] == ["shot-0000", "shot-0001"]
    assert [r["frame"] for s in final["shots"] for r in s["registration"]] == list(range(10))
    assert final["shots"][1]["predicted"] == "court"
    assert all((args.out / p).exists() for s in final["shots"] for p in s["images"])
    assert file_sha256(partial) == old_hash
    assert source.read_bytes() == b"immutable source fixture"
    report["chunks"] = recovered[:1]
    files["publish/recovery-report.json"] = json.dumps(report).encode()
    incomplete = tmp_path / "incomplete.tar"
    write_tar(incomplete, files)
    args.archive, args.out = [incomplete], tmp_path / "incomplete-review"
    with pytest.raises(ValueError, match="missing recovery chunks"):
        kr.assemble(args)
    assert not (args.out / "READY.json").exists()
