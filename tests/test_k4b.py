"""K4b selection and review-mask identity checks; no model or archive needed."""

import hashlib
import json
import sys
import tarfile

import numpy as np
import pytest

from experiments.tennis.k4b import numbered, page
from experiments.tennis import k4b
from experiments.visor.b1 import file_sha256
from src.segmentation.masks import ClipMasks
from tools.datasets.k4_inputs import MATCHES
from tools.datasets.k4b_inputs import frame_hash, select_frames


def test_selection_is_global_prefix_and_excludes_entire_k4_clips():
    inventory = [{"match": m, "clip": f"{c:03d}", "frames": 20} for m in MATCHES for c in range(6)]
    excluded = {f"broadcast/{m}/000" for m in MATCHES}
    rows = select_frames(inventory, excluded)
    all_rows = sorted((frame_hash(c["match"], c["clip"], t), c["match"], c["clip"], t)
                      for c in inventory if c["clip"] != "000" for t in range(c["frames"]))
    assert [r["rank"] for r in rows] == [r[0] for r in all_rows[:len(rows)]]
    assert len(rows) >= 48
    assert {r["match"] for r in rows} == set(MATCHES)
    assert len({(r["match"], r["clip"]) for r in rows}) >= 8
    assert all(r["clip"] != "000" for r in rows)
    assert select_frames(list(reversed(inventory)), excluded) == rows
    assert frame_hash("m", "001", 0) == hashlib.sha256(b"pointstream-k4b:m/001/0").hexdigest()


def test_selection_never_relaxes_diversity():
    with pytest.raises(ValueError, match="cannot meet"):
        select_frames([{"match": "alcaraz_hurkacz", "clip": "001", "frames": 100}], set())


def test_duplicate_prompts_share_one_number_but_raw_masks_survive():
    clip = ClipMasks(("person", "player"), 8, 8, 25)
    a = np.zeros((8, 8), bool)
    a[1:4, 1:4] = True
    b = np.zeros((8, 8), bool)
    b[5:7, 5:7] = True
    clip.add(0, "person", 4, a, .9)
    clip.add(0, "player", 2, a, .8)
    clip.add(0, "person", 8, b, .7)
    groups = numbered(clip)
    assert [r["id"] for r in groups] == [1, 2]
    assert {r["prompt"] for r in groups[0]["sources"]} == {"person", "tennis player"}
    assert len(clip.frames[0]) == 3
    assert numbered(ClipMasks(("person", "player"), 8, 8, 25, frames=[[]])) == []


def test_page_is_offline_and_binds_marks_to_both_identities():
    text = page([{"id": "</script>"}], "selection", "result")
    payload = text.split('<script id="data" type="application/json">')[1].split('</script>')[0]
    assert json.loads(payload)["frames"][0]["id"] == "</script>"
    assert "fetch(" not in text
    assert 'result_sha256:data.result_sha256' in text
    assert 'selection_sha256:data.selection_sha256' in text
    assert 'confirmed:false' in text


def test_validator_reads_published_archive_after_scratch_removal(tmp_path, monkeypatch):
    selection = {"inventory": [{"match": m, "clip": f"{c:03d}", "frames": 20} for m in MATCHES for c in range(6)], "excluded": []}
    selection["frames"] = [{**r, "member": "test.yuv", "offset": i, "fps": 25,
                             "context_frames": list(range(max(0, min(r["frame"] - 1, 17)), max(0, min(r["frame"] - 1, 17)) + 3))}
                            for i, r in enumerate(select_frames(selection["inventory"], set()))]
    publish = tmp_path / "publish"
    publish.mkdir()
    clip = ClipMasks(("person", "player"), 540, 960, 25, frames=[[]])
    mask = np.zeros((540, 960), bool)
    mask[100:200, 100:150] = True
    clip.add(0, "person", 1, mask, .9)
    clip.save(publish / "masks.json.zst")
    row = {**selection["frames"][0], "id": "frame0", "masks": "masks.json.zst", "masks_sha256": file_sha256(publish / "masks.json.zst"),
           **k4b.review_frame(np.zeros((1080, 1920, 3), np.uint8), numbered(clip))}
    sam = {"runtime": {"capability": [8, 6], "cuda": "test"}, "sdpa_backend_policy": "native_flash",
           "checkpoint_sha256": "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6", "prompts": k4b.PROMPTS}
    record = {"frames": [{k: v for k, v in row.items() if k not in ("image", "original")}], "sam": sam,
              "count": 1, "selection_sha256": "selection", "converted_inputs_verified": True}
    (publish / "instances.json").write_text(json.dumps(record))
    sha = file_sha256(publish / "instances.json")
    (publish / "index.html").write_text(page([row], "selection", sha))
    (tmp_path / "k4b.json").write_text(json.dumps({"selection": selection, "count": 1, "record_sha256": sha}))
    with tarfile.open(tmp_path / "published.tar", "w") as archive:
        archive.add(publish, arcname="publish")
    monkeypatch.setenv("PS_STAGE_DIR", str(tmp_path))
    monkeypatch.setenv("PS_SCRATCH_DIR", str(tmp_path / "removed-scratch"))
    monkeypatch.setenv("PS_VALIDATION_PATH", str(tmp_path / "validation.json"))
    monkeypatch.setattr(sys, "argv", ["k4b", "validate"])
    assert k4b.main() == 0
    assert json.loads((tmp_path / "validation.json").read_text())["passed"]
    # A valid archive cannot hide a changed raw mask file.
    with (publish / "masks.json.zst").open("ab") as handle:
        handle.write(b"tamper")
    result = k4b.validate_directory(publish, tmp_path / "k4b.json")
    assert result["passed"] is False
    assert result["checks"]["mask_integrity"] is False
    record["frames"][0]["instances"] = []
    (publish / "instances.json").write_text(json.dumps(record))
    assert k4b.validate_directory(publish, tmp_path / "k4b.json")["checks"]["nonempty_model_output"] is False
