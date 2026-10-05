"""Publication must reject mixed, partial or changed demo releases."""

import hashlib
from copy import deepcopy

import pytest

from demo.pitch.release import verify_release
from demo.experiments.clip_identity import CLIP_IDS
from demo.experiments.demo_mask_refresh import FAMILIES, RUNGS
from demo.experiments.validate_demo_refresh import EXPECTED_STREAMS


def complete(tmp_path):
    release = {
        "schema": "pointstream.demo.release.v1",
        "clips": {},
        "maps": {"entries": []},
        "files": {},
    }
    paths = []
    for clip in CLIP_IDS:
        release["clips"][clip] = {
            "n_frames": 299 if clip == "clip_02" else 300,
            "streams": {key: {} for key in EXPECTED_STREAMS},
        }
        paths.extend(f"web_{clip}_{key}.mp4" for key in EXPECTED_STREAMS)
        paths.append(f"keypoints_{clip}.json")
        for family in FAMILIES:
            for rung in RUNGS:
                relative = f"{clip}/{family}_{rung}.mp4"
                release["maps"]["entries"].append(
                    {
                        "clip": clip,
                        "family": family,
                        "resolution": rung,
                        "n_frames": release["clips"][clip]["n_frames"],
                        "overlay_url": relative,
                    }
                )
                paths.append("maps/" + relative)
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        release["files"][relative] = {
            "sha256": hashlib.sha256(relative.encode()).hexdigest(),
            "bytes": len(relative),
        }
    return release


def test_one_complete_release(tmp_path):
    verify_release(tmp_path, complete(tmp_path))


@pytest.mark.parametrize(
    "fault",
    [
        "missing_mask",
        "duplicate_mask",
        "smoke_frames",
        "wrong_interval",
        "missing_camera",
        "changed_file",
        "extra_file",
        "unsafe_path",
    ],
)
def test_bad_release_cannot_publish(tmp_path, fault):
    release = deepcopy(complete(tmp_path))
    if fault == "missing_mask":
        release["maps"]["entries"].pop()
    elif fault == "duplicate_mask":
        release["maps"]["entries"][-1] = release["maps"]["entries"][0]
    elif fault == "smoke_frames":
        release["clips"]["clip_01"]["n_frames"] = 8
    elif fault == "wrong_interval":
        release["maps"]["entries"][0]["n_frames"] = 3000
    elif fault == "missing_camera":
        release["clips"]["clip_02"]["streams"].pop("av1_1080")
    elif fault == "changed_file":
        (tmp_path / "web_clip_02_ref.mp4").write_bytes(b"old clip")
    elif fault == "extra_file":
        release["files"]["historical.mp4"] = {}
    else:
        release["maps"]["entries"][0]["overlay_url"] = "../../historical.mp4"
        old = release["files"].pop("maps/clip_01/yoloe_180p.mp4")
        release["files"]["maps/../../historical.mp4"] = old
    with pytest.raises(ValueError):
        verify_release(tmp_path, release)
