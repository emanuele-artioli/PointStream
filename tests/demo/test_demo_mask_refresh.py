import hashlib
from copy import deepcopy

import pytest
from demo.experiments import demo_mask_refresh as masks


def ready(tmp_path, monkeypatch):
    payload = tmp_path / "data"
    payload.write_bytes(b"immutable")
    identity = {"path": str(payload), "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
    streams = {
        rung: {"relative_path": "data", "sha256": identity["sha256"]} for rung in masks.RUNGS
    }
    monkeypatch.setattr(
        masks, "probe_video", lambda *args: {"codec_name": "av1", "nb_read_frames": "2"}
    )
    return {
        "status": "complete",
        "family": "rtmpose",
        "weights": [identity],
        "clips": {
            clip: {"source_identity": identity, "n_frames": 2, "streams": deepcopy(streams)}
            for clip in ("clip_01", "clip_02", "clip_03")
        },
    }


def test_complete_mask_ladders(tmp_path, monkeypatch):
    masks.validate(ready(tmp_path, monkeypatch), tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_clip",
        "missing_rung",
        "changed_source",
        "missing_weights",
        "wrong_codec",
        "short_decode",
        "changed_media",
    ],
)
def test_bad_masks_block_promotion(tmp_path, monkeypatch, fault):
    report = ready(tmp_path, monkeypatch)
    if fault == "missing_clip":
        report["clips"].pop("clip_02")
    elif fault == "missing_rung":
        report["clips"]["clip_01"]["streams"].pop("1080p")
    elif fault == "changed_source":
        (tmp_path / "data").write_bytes(b"changed")
    elif fault == "missing_weights":
        report["weights"] = []
    elif fault == "changed_media":
        report["clips"]["clip_01"]["streams"]["1080p"]["sha256"] = "bad"
    else:
        monkeypatch.setattr(
            masks,
            "probe_video",
            lambda *args: {
                "codec_name": "h264" if fault == "wrong_codec" else "av1",
                "nb_read_frames": "1" if fault == "short_decode" else "2",
            },
        )
    with pytest.raises(ValueError):
        masks.validate(report, tmp_path)
