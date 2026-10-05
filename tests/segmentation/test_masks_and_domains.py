from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from src.segmentation import BACKENDS, build, domain_names, load_domain
from src.segmentation.masks import SCHEMA, ClipMasks, _string_to_runs, decode_rle, encode_rle


def _blob(height: int = 9, width: int = 11) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[2:6, 3:9] = True
    mask[7, 0] = True
    return mask


@pytest.mark.parametrize("mask", [_blob(), np.zeros((5, 4), bool), np.ones((3, 3), bool)])
def test_rle_round_trips_exactly(mask: np.ndarray) -> None:
    rle = encode_rle(mask)
    assert rle["size"] == list(mask.shape)
    assert np.array_equal(decode_rle(rle), mask)


def test_compressed_coco_counts_decode_without_pycocotools() -> None:
    runs = [3, 4, 2]
    text = "".join(
        chr(r + 48) for r in runs
    )  # runs below 16 (and the first three) are one character each
    assert _string_to_runs(text) == runs
    mask = decode_rle({"size": [3, 3], "counts": text})
    assert mask.ravel(order="F").tolist() == [False] * 3 + [True] * 4 + [False] * 2


def test_clip_masks_label_and_foreground_maps(tmp_path: Path) -> None:
    clip = ClipMasks(("player", "racket"), 9, 11, 25.0)
    player = _blob()
    racket = np.zeros_like(player)
    racket[4:8, 7:10] = True
    clip.add(0, "player", 1, player, 0.9)
    clip.add(0, "racket", 2, racket, 0.8)
    clip.add(2, "player", 1, np.zeros_like(player))  # empty masks are not stored
    assert len(clip) == 3 and clip.frames[1] == [] and clip.frames[2] == []
    assert np.array_equal(clip.foreground(0), player | racket)
    labels = clip.labels(0)
    assert labels[racket].tolist() == [2] * int(
        racket.sum()
    )  # later classes paint over earlier ones
    assert set(np.unique(labels[player & ~racket])) == {1}
    assert not clip.foreground(1).any()
    with pytest.raises(ValueError, match="not one of"):
        clip.add(0, "ball", 3, player)
    with pytest.raises(ValueError, match="shape"):
        clip.add(0, "player", 3, np.ones((2, 2)))

    clip.meta["backend"] = "test"
    path = clip.save(tmp_path / "run")
    assert path.name == "masks.rle"
    loaded = ClipMasks.load(tmp_path / "run")
    assert (
        loaded.classes == clip.classes and len(loaded) == 3 and loaded.meta == {"backend": "test"}
    )
    assert loaded.frames[0][1].track_id == 2 and loaded.frames[0][1].bbox == (7.0, 4.0, 10.0, 8.0)
    assert np.array_equal(loaded.labels(0), labels)


def test_legacy_maps_gallery_payload_loads(tmp_path: Path) -> None:
    mask = _blob()
    doc = {
        "schema": SCHEMA,
        "height": 9,
        "width": 11,
        "fps": 30.0,
        "classes": ["hand"],
        "frames": [
            {
                "index": 1,
                "instances": [
                    {
                        "class_id": 0,
                        "class_name": "hand",
                        "score": 0.5,
                        "bbox": [3, 2, 9, 8],
                        "rle": encode_rle(mask),
                    }
                ],
            }
        ],
    }
    (tmp_path / "payload.bin").write_bytes(zlib.compress(json.dumps(doc).encode()))
    clip = ClipMasks.load(tmp_path)
    assert len(clip) == 2 and clip.frames[0] == []
    assert np.array_equal(clip.foreground(1), mask)


def test_domains_define_foreground_classes_and_backend_overrides(tmp_path: Path) -> None:
    assert {"tennis", "egocentric"} <= set(domain_names())
    tennis = load_domain("tennis")
    assert tennis.classes == ("player", "racket")
    assert tennis.prompts_for("sam") == {"player": "tennis player", "racket": "tennis racket"}
    ego = load_domain("egocentric")
    assert set(ego.classes) == {"hand", "arm"}
    assert ego.options_for("yoloe")["min_hits"] == 5
    assert ego.options_for("sam") == {}

    custom = tmp_path / "domains.yaml"
    custom.write_text(
        "d:\n  classes: {hand: hand}\n  backends:\n    yoloe: {prompts: {hand: person}, conf: 0.1}\n"
    )
    domain = load_domain("d", custom)
    assert domain.prompts_for("yoloe") == {"hand": "person"}
    assert domain.prompts_for("sam") == {"hand": "hand"}
    assert domain.options_for("yoloe") == {"conf": 0.1}
    with pytest.raises(KeyError, match="unknown segmentation domain"):
        load_domain("soccer")


def test_missing_domain_clips_fail_loudly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PS_DATASETS_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not on disk"):
        load_domain("tennis").clip_paths()


def test_backend_registry_names() -> None:
    assert BACKENDS == ("sam31", "yoloe-26n", "yoloe-26s", "yoloe-26m", "yoloe-26l", "yoloe-26x")
    assert build("yoloe-26m", model=object()).name == "yoloe-26m"
    with pytest.raises(KeyError, match="unknown segmentation backend"):
        build("yolo-seg")
