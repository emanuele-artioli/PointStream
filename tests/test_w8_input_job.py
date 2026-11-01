import numpy as np
import pytest

from experiments.tennis.w8_input_job import judged_bitmap, validate


def test_declared_recipe_has_exact_eighty_ticks():
    ticks = np.unpackbits(np.frombuffer(judged_bitmap("0:240:3"), np.uint8)).nonzero()[0]
    assert ticks.tolist() == list(range(0, 240, 3))
    with pytest.raises(ValueError):
        judged_bitmap("1:240:3")


def test_validator_fails_closed_without_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_SCRATCH_DIR", str(tmp_path))
    # Avoid imposing a process CPU limit on the pytest runner itself.
    monkeypatch.setattr("experiments.tennis.w8_input_job.cpu_limit", lambda _: None)
    result = validate(tmp_path)
    assert not result["passed"]
    assert not result["checks"]["validator_completed"]
    assert "FileNotFoundError" in result["details"]["error"]


def test_denied_probes_must_exist_and_read(tmp_path):
    from experiments.tennis.w8_input_job import _verified_denials

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"a")
    with pytest.raises(ValueError, match="not a file"):
        _verified_denials([first, second])
    second.write_bytes(b"b")
    _verified_denials([first, second])
    with pytest.raises(ValueError, match="Distinct"):
        _verified_denials([first, first])


def test_allowlist_rejects_forbidden_inside_project(tmp_path):
    from pathlib import Path
    from experiments.tennis.w8_input_job import _allowed_read_paths

    project_file = Path(__file__).resolve()
    with pytest.raises(ValueError, match="forbidden"):
        _allowed_read_paths(tmp_path, tmp_path, [project_file])


def test_stage_validation_survives_scratch_removal(tmp_path, monkeypatch):
    import errno
    import io
    import json
    import shutil
    import tarfile
    from experiments.tennis import w8_input_job as job

    stage = tmp_path / "stage"
    packets = stage / "packets"
    decoded = stage / "decoded"
    packets.mkdir(parents=True)
    decoded.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    temp = tmp_path / "host-tmp"
    temp.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(job, "cpu_limit", lambda _: None)
    monkeypatch.setattr(job, "validate_golden", lambda *_: None)
    contents = {"keypoints.kps": b"keypoints", **{f"{n}.png.ref": n.encode() for n in job.UNITS}}
    for name, content in contents.items():
        (packets / name).write_bytes(content)
    packet_shas = {name: job.digest(content) for name, content in contents.items()}
    monkeypatch.setattr(job, "PACKET_SHAS", packet_shas)
    prepared = stage / "prepared.tar"
    with tarfile.open(prepared, "w") as tar:
        for name in job.UNITS:
            content = b"golden"
            member = tarfile.TarInfo(f"publish/units/{name}.npz")
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(job, "PREPARED_SHA", job.digest(prepared.read_bytes()))
    hashes = {}
    for name in job.UNITS:
        path = decoded / f"{name}.npz"
        np.savez_compressed(path, reference=np.zeros(1), reference_box=np.zeros(4), t=np.arange(240),
                            run=np.zeros(240), box=np.zeros((240, 4)), pose=np.zeros(1),
                            judged=np.arange(240) % 3 == 0)
        hashes[name] = job.digest(path.read_bytes())
    forbidden = ["/golden", "/prepared-copy", "/probe", "/source"]
    (stage / "receipt.json").write_text(json.dumps({"worker_returncode": 0, "packet_dir": "packets", "prepared": "prepared.tar",
                                                    "packets": packet_shas, "recipe": job.RECIPE, "parent_cpu_s": 0, "forbidden": forbidden}))
    import cv2
    (decoded / "worker.json").write_text(json.dumps({"isolation": {"landlock_abi": 1, "denied": {p: errno.EACCES for p in forbidden}},
                                                       "recipe": job.RECIPE, "numpy": np.__version__, "opencv": cv2.__version__,
                                                       "cpu_s": 0, "hashes": hashes}))
    shutil.rmtree(scratch)
    result = job.validate(stage)
    assert result["passed"], result
    assert not scratch.exists()
    manifest = job.evidence_manifest(stage)
    (packets / "keypoints.kps").write_bytes(b"tampered")
    assert manifest != job.evidence_manifest(stage)
    assert not job.validate(stage)["passed"]
