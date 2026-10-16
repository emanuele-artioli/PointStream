"""K4b selection and review-mask identity checks; no model or archive needed."""

import hashlib
import json

import numpy as np
import pytest

from experiments.tennis.k4b import numbered, page
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
