import hashlib
import json
from copy import deepcopy

import pytest

from demo.experiments import validate_demo_refresh as validator


def ready_report(tmp_path, monkeypatch):
    source = tmp_path / "input.bin"
    source.write_bytes(b"frozen input")
    identity = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    pitch = tmp_path / "pitch"
    pitch.mkdir()
    clips = {}
    for clip in validator.CLIP_IDS:
        (pitch / f"keypoints_{clip}.json").write_text(json.dumps([{}, {}]))
        clips[clip] = {
            "n_frames": 2,
            "checkpoint_identity": identity,
            "anchor_identity": identity,
            "source_identity": identity,
            "streams": {
                key: {"kbps": 1.0, "det": 0.0, "kp": 0.0} for key in validator.EXPECTED_STREAMS
            },
        }

    def probe(path, ffprobe):
        return {
            "codec_name": "av1" if "_av1_" in path.name else "h264",
            "nb_read_frames": "2",
            "width": 1920,
            "height": 1080,
        }

    monkeypatch.setattr(validator, "probe_video", probe)
    return {"status": "complete", "clips": clips}


def test_complete_decode_receipts_allow_promotion(tmp_path, monkeypatch):
    checks = validator.validate(ready_report(tmp_path, monkeypatch), tmp_path, ffprobe="unused")
    assert checks["native_media_decodes"] and checks["matching_actual_frame_counts"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_rung",
        "stale_input",
        "nonfinite_rate",
        "wrong_codec",
        "short_decode",
        "short_keypoints",
    ],
)
def test_bad_outputs_block_promotion(tmp_path, monkeypatch, fault):
    report = deepcopy(ready_report(tmp_path, monkeypatch))
    row = report["clips"]["clip_01"]
    if fault == "missing_rung":
        row["streams"].pop("ps_1080")
    elif fault == "stale_input":
        (tmp_path / "input.bin").write_bytes(b"changed")
    elif fault == "nonfinite_rate":
        row["streams"]["av1_1080"]["kbps"] = float("nan")
    elif fault in ("wrong_codec", "short_decode"):
        monkeypatch.setattr(
            validator,
            "probe_video",
            lambda *a: {
                "codec_name": "h264",
                "nb_read_frames": "1" if fault == "short_decode" else "2",
                "width": 1920,
                "height": 1080,
            },
        )
    else:
        (tmp_path / "pitch/keypoints_clip_01.json").write_text("[]")
    with pytest.raises(ValueError):
        validator.validate(report, tmp_path, ffprobe="unused")
