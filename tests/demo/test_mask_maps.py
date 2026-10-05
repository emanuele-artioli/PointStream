"""Demo gallery mask maps built on src.segmentation."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from demo.models.matte import read_hand_alphas
from demo.pipeline.maps.av1_crf import CLASS_COLORS_BGR, av1_output_args
from demo.pipeline.maps.masks import paint
from src.segmentation import ClipMasks


def _clip() -> ClipMasks:
    masks = ClipMasks(("arm", "hand"), 6, 8, 30.0)
    arm = np.zeros((6, 8), dtype=bool)
    arm[1:5, 1:7] = True
    hand = np.zeros((6, 8), dtype=bool)
    hand[2:4, 5:7] = True
    masks.add(0, "arm", 1, arm)
    masks.add(0, "hand", 2, hand)
    masks.ensure_frames(2)
    return masks


def test_paint_keys_on_black_with_later_classes_on_top() -> None:
    frame = paint(_clip(), 0)
    assert tuple(frame[2, 5]) == CLASS_COLORS_BGR["hand"]
    assert tuple(frame[1, 1]) == CLASS_COLORS_BGR["arm"]
    assert not frame[0].any() and not paint(_clip(), 1).any()


def test_matte_reads_lossless_foreground_and_resizes(tmp_path: Path) -> None:
    _clip().save(tmp_path / "run")
    alphas = read_hand_alphas(tmp_path / "run", 16, 12, 5)
    assert len(alphas) == 2
    assert alphas[0].shape == (12, 16) and set(np.unique(alphas[0])) == {0, 255}
    assert int(alphas[0][2, 2]) == 255  # arm counts: the foreground is arm and hand
    assert not alphas[1].any()


def test_oracle_reader_accepts_the_shared_mask_stream(tmp_path: Path) -> None:
    from demo.experiments.compare_mask_oracles import rle_supports

    path = _clip().save(tmp_path)
    supports = rle_supports(path.read_bytes())
    assert len(supports) == 2 and int(supports[0].sum()) == 24
    with pytest.raises(AssertionError):
        rle_supports(zlib.compress(json.dumps({"schema": "other"}).encode()))


def test_shared_av1_recipe_is_crf63() -> None:
    args = av1_output_args()
    assert args[args.index("-vf") + 1] == "scale=426:240"
    assert args[args.index("-c:v") + 1] == "libsvtav1"
    assert args[args.index("-preset") + 1] == "7"
    assert args[args.index("-crf") + 1] == "63"
    assert "-b:v" not in args
