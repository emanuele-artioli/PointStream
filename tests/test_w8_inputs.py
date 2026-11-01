import io
import tarfile

import numpy as np
import pytest

from experiments.tennis import k6b
from experiments.tennis.k6b_rate_streams import encode_keypoints_kps2, encode_reference
from experiments.tennis.w8_inputs import archive_members, decode_inputs, validate_golden


def packets():
    times = np.array([0, 1, 3], dtype=np.int64)
    kp = np.zeros((3, 17, 3), dtype=np.float64)
    kp[:, :, 0] = np.arange(17) * 4
    kp[:, :, 1] = np.arange(17) * 8
    kp[:, :, 2] = 1
    names = ["win__one", "win__two"]
    units = [{"name": name, "t": times, "kp": kp} for name in names]
    bitmap = np.zeros(240, dtype=np.uint8)
    bitmap[[0, 3]] = 1
    meta = {"window": "win", "native_frames": 240, "start_tick": 0, "fps": [50, 1], "time_base": [1, 50],
            "model_profile": "C-q4-guidance2-DDIM30", "common_base": "tennis-base@fit-100",
            "generation_size": [k6b.GEN_W, k6b.GEN_H],
            "crop_rule": [k6b.CROP_MARGIN, k6b.SIZE_PERCENTILE, k6b.CENTRE_SMOOTH], "context": [24, 4],
            "seed_rule": "sha256(unit_name)[:7] then seed*1009+run", "registered_frame_bitmap": np.packbits(bitmap).tobytes().hex(),
            "compositing_order": ["one", "two"]}
    refs = {name: encode_reference(np.zeros((k6b.GEN_H, k6b.GEN_W, 3), dtype=np.uint8),
                                   {"unit": name, "reference_frame": 0, "reference_box": [0, 0, 10, 10],
                                    "pixels": "full_prepared_conditioning_crop"}) for name in names}
    return encode_keypoints_kps2(meta, units), refs, np.packbits(bitmap).tobytes()


def test_decoded_arrays_and_separate_golden(tmp_path):
    blob, refs, judged = packets()
    decoded, binding = decode_inputs(blob, refs, "win", judged)
    assert binding["profile"] == "C-q4-guidance2-DDIM30"
    paths = {}
    for name, fields in decoded.items():
        assert fields["run"].tolist() == [0, 0, 1]
        assert fields["judged"].tolist() == [True, False, True]
        paths[name] = tmp_path / f"{name}.npz"
        np.savez_compressed(paths[name], **fields)
    validate_golden(decoded, paths, blob)
    decoded["win__one"]["judged"][0] = False
    with pytest.raises(ValueError, match="Golden mismatch"):
        validate_golden(decoded, paths, blob)


def test_reject_wrong_binding():
    blob, refs, judged = packets()
    with pytest.raises(ValueError, match="contract"):
        decode_inputs(blob, refs, "other", judged)
    refs["win__one"] = refs["win__two"]
    with pytest.raises(ValueError, match="Reference identity"):
        decode_inputs(blob, refs, "win", judged)


def test_archive_bounds_and_duplicate(tmp_path):
    path = tmp_path / "x.tar"
    with tarfile.open(path, "w") as tar:
        for name in ("a", "a"):
            info = tarfile.TarInfo(name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Duplicate"):
        archive_members(path, ["a"])
    with pytest.raises(ValueError, match="Unsafe"):
        archive_members(path, ["../x"])
