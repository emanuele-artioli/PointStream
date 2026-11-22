"""Bind each selected PNG to the fully verified receiver and its actual joint packet."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from experiments.tennis.player_requalification import array_sha

HERE = Path(__file__).resolve().parent


def bind(directory, out, sender_path, source_path):
    path = directory / "manifest.json"
    record = json.loads(path.read_text())
    sender = json.loads(sender_path.read_text())
    native_source = json.loads(source_path.read_text())
    assert record["source_sha256"] == sender["source_sha256"] == native_source["source_sha256"]
    fps = Fraction(sender["native_fps"])
    assert Fraction(record["native_fps"]) == fps == Fraction(native_source["native_fps"])
    payment = next(
        r for r in sender["rates"] if r["joint_sha256"] == record["wire"]["joint_sha256"]
    )
    assert (
        payment["joint_bytes"] == record["joint_rally_bytes"]
        and payment["init_bytes"] == record["foreground_cache_bytes"]
    )
    assert len(record["all_240_output_sha256"]) == len(record["receipt_sha256"]) == 64
    rows = []
    for r in record["selected"]:
        tick = r["tick"]
        source = native_source["ticks"][str(tick)]
        assert (
            r["source_raw_rgb_sha256"] == source["raw_rgb_sha256"]
            and r["source_typed_rgb_sha256"] == source["typed_rgb_sha256"]
        )
        file = directory / r["file"]
        assert (
            file.resolve().parent == directory.resolve()
            and hashlib.sha256(file.read_bytes()).hexdigest() == r["png_sha256"]
        )
        rgb = np.ascontiguousarray(cv2.imread(str(file))[:, :, ::-1])
        raw = hashlib.sha256(rgb).hexdigest()
        assert rgb.shape == (1080, 1920, 3) and array_sha(rgb) == r["receipt_typed_rgb_sha256"]
        rows.append(
            dict(tick=tick, path=str(file.resolve()), png_sha256=r["png_sha256"], rgb_sha256=raw)
        )
    assert sorted(r["tick"] for r in rows) == sorted(map(int, native_source["ticks"]))
    arm = dict(
        mode="PointStream two players and ball, base CRF " + str(payment["crf"]),
        bytes=payment["joint_bytes"],
        bitstream_sha256=payment["joint_sha256"],
        frames=rows,
        source_ticks=240,
        source_sha256=sender["source_sha256"],
        fps=[fps.numerator, fps.denominator],
        full_receiver_sha256=record["all_240_output_sha256"],
        finish_job=record["finish_job"],
        native_receipt_sha256=record["receipt_sha256"],
    )
    with out.open("x") as f:
        json.dump(arm, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--sender", type=Path, default=HERE.parent / "test-crop-joint-20261122/match102-sender.json"
    )
    parser.add_argument(
        "--source-witness",
        type=Path,
        default=HERE.parent / "test-crop-joint-20261122/match102-source-review-frames.json",
    )
    a = parser.parse_args()
    bind(a.directory, a.out, a.sender, a.source_witness)
