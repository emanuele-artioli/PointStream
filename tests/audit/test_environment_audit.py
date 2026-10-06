"""The environment audit's own code: DCVC container, MANO conversion, smoke helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from experiments.audit import env_smoke
from src.codecs import dcvc_uf_worker as dcvc
from src.segmentation import sam31
from tools.models import mano_dechumpy


def test_dcvc_container_round_trip_and_rejects_corruption() -> None:
    packed = dcvc.pack_container(b"native", structure="hts", frame_count=9)
    header, native = dcvc.unpack_container(packed)
    assert header == {"structure": "hts", "frame_count": 9}
    assert native == b"native"
    with pytest.raises(ValueError):
        dcvc.unpack_container(b"XXXX" + packed[4:])
    with pytest.raises(ValueError):
        dcvc.pack_container(b"", structure="hts", frame_count=0)


def test_dcvc_display_schedule_matches_frame_delay() -> None:
    assert dcvc.display_schedule(9, 8) == [[0], list(range(1, 9))]
    assert dcvc.display_schedule(11, 8) == [[0], list(range(1, 9)), [9, 10]]
    assert dcvc.display_schedule(3, 1) == [[0], [1], [2]]


def test_dcvc_command_runs_the_vendored_tree(tmp_path: Path) -> None:
    command, env, cwd = dcvc.dcvc_command(tmp_path, "encode", tmp_path / "p.json", tmp_path / "r.json")
    assert command[0] == str(tmp_path / "bin" / "python")
    assert command[1].endswith("dcvc_uf_worker.py") and command[2] == "encode"
    assert cwd == tmp_path / "opt" / "DCVC" and env["PYTHONPATH"] == str(cwd)


def _chumpy_pickle(path: Path, value: np.ndarray) -> None:
    """A MANO-like pickle whose arrays are chumpy Ch and Select nodes."""
    ch = types.ModuleType("chumpy.ch")
    reordering = types.ModuleType("chumpy.reordering")

    class Ch:
        pass

    class Select:
        pass

    Ch.__module__, Ch.__qualname__ = "chumpy.ch", "Ch"
    Select.__module__, Select.__qualname__ = "chumpy.reordering", "Select"
    ch.Ch, reordering.Select = Ch, Select  # type: ignore[attr-defined]
    sys.modules.update({"chumpy": types.ModuleType("chumpy"), "chumpy.ch": ch, "chumpy.reordering": reordering})
    try:
        full = Ch()
        full.__dict__ = {"x": value}
        selected = Select()
        selected.__dict__ = {"a": full, "idxs": np.arange(value.size // 2), "preferred_shape": (value.size // 2,)}
        template = Ch()
        template.__dict__ = {"x": value * 2}
        with path.open("wb") as handle:
            pickle.dump({"shapedirs": selected, "v_template": template, "f": np.arange(3)}, handle)
    finally:
        for name in ("chumpy", "chumpy.ch", "chumpy.reordering"):
            sys.modules.pop(name, None)


def test_mano_dechumpy_evaluates_ch_and_select_without_chumpy(tmp_path: Path) -> None:
    value = np.arange(12, dtype=np.float64)
    source, target = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    for name in mano_dechumpy.NAMES:
        _chumpy_pickle(source / name, value)
    assert mano_dechumpy.main([str(source), str(target)]) == 0
    model = pickle.loads((target / "MANO_RIGHT.pkl").read_bytes())
    np.testing.assert_array_equal(model["shapedirs"], value[:6])
    np.testing.assert_array_equal(model["v_template"], value * 2)
    manifest = json.loads((target / "MANIFEST.json").read_text())
    assert manifest["files"][0]["converted_from_chumpy"] == {
        "shapedirs": "chumpy.reordering.Select", "v_template": "chumpy.ch.Ch"}
    with pytest.raises(SystemExit):
        mano_dechumpy.main([str(source), str(target)])


class _Entry:
    def __init__(self, root: Path, name: str, data: bytes) -> None:
        self.root, self.name = root, name
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        self.hash = types.SimpleNamespace(mode="sha256", value=digest)

    def __str__(self) -> str:
        return self.name


class _Dist:
    def __init__(self, root: Path, direct: dict, files: dict[str, bytes]) -> None:
        self.root, self.direct = root, direct
        self.files = [_Entry(root, n, d) for n, d in files.items()]
        for name, data in files.items():
            (root / name).write_bytes(data)

    def read_text(self, name: str) -> str:
        return json.dumps(self.direct)

    def locate_file(self, entry: _Entry) -> Path:
        return self.root / entry.name


def test_installed_sam3_revision_requires_vcs_commit_and_intact_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import metadata

    direct = {"url": "https://github.com/facebookresearch/sam3", "vcs_info": {"vcs": "git", "commit_id": "abc"}}
    dist = _Dist(tmp_path, direct, {"model.py": b"code"})
    monkeypatch.setattr(metadata, "distribution", lambda name: dist)
    assert sam31.installed_sam3_revision() == "abc"
    (tmp_path / "model.py").write_bytes(b"edited")
    with pytest.raises(RuntimeError, match="RECORD"):
        sam31.installed_sam3_revision()
    dist.direct = {"url": "file:///x", "dir_info": {"editable": True}}
    assert sam31.installed_sam3_revision() is None


def test_smoke_helpers(tmp_path: Path) -> None:
    assert env_smoke.attention_family(["flash_fwd_kernel", "fmha_cutlassF_f16", "volta_sgemm"]) == ["flash", "mem_efficient"]
    mask = np.zeros((4, 5), bool)
    mask[1:3, 2:4] = True
    assert env_smoke.bbox(mask) == (2, 1, 4, 3)
    assert env_smoke.iou(mask, mask) == 1.0
    luma = np.arange(16, dtype=np.uint8).reshape(4, 4)
    frame = np.concatenate([luma.ravel(), np.zeros(8, np.uint8)]).tobytes()
    (tmp_path / "a.y4m").write_bytes(b"YUV4MPEG2 W4 H4 F25:1 C420jpeg\n" + (b"FRAME\n" + frame) * 2)
    planes = env_smoke.read_y4m(tmp_path / "a.y4m")
    assert len(planes) == 2 and (planes[1] == luma).all()
    assert env_smoke.psnr(luma, luma) == 99.0


def test_validator_requires_every_component_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PS_STAGE_DIR", str(tmp_path))
    monkeypatch.setenv("PS_VALIDATION_PATH", str(tmp_path / "validation.json"))
    components = tmp_path / "components"
    components.mkdir()
    device = {"name": "NVIDIA RTX A6000"}
    (components / "visor.json").write_text(json.dumps({"passed": True, "device": "cpu", "checks": {"a": True}}))
    (components / "sam31.json").write_text(json.dumps({"passed": True, "device": device, "checks": {"b": True}}))
    assert env_smoke.main(["validate", "--group", "sam31", "--gpu-class", "RTX A6000"]) == 0
    assert json.loads((tmp_path / "validation.json").read_text())["passed"] is True
    assert env_smoke.main(["validate", "--group", "sam31", "--gpu-class", "RTX 6000 Ada"]) == 1
    (components / "sam31.json").write_text(json.dumps({"passed": False, "device": device, "checks": {"b": False}}))
    assert env_smoke.main(["validate", "--group", "sam31", "--gpu-class", "RTX A6000"]) == 1
