"""Verify the native admission tracks and price a complete near-player signal."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import io
import itertools
import json
from pathlib import Path
import tarfile
import zlib

import cv2
import numpy as np
from experiments.tennis import (
    common_player_packet as packet,
    k6b,
    k6b_rate_streams as rates,
    w2_pose_rate,
)
from src.segmentation.masks import decompress


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def portable(blob):
    magic, raw_size, _, checksum = rates.HEADER.unpack_from(blob)
    raw = decompress(blob[rates.HEADER.size :])
    assert len(raw) == raw_size and hashlib.sha256(raw).digest() == checksum
    encoded = zlib.compress(raw, 6)
    return rates.HEADER.pack(magic, raw_size, len(encoded), checksum) + encoded


def price(t, kp, source, name, fps):
    coder = w2_pose_rate.Setting(q=4, key=4, predictor="previous", interp="linear", vis="all")
    unit = dict(name=name, t=t, kp=kp)
    quant = w2_pose_rate.encode_window([unit], coder).expected[0]
    coded = portable(
        rates.encode_keypoints_kps2({"native_frames": 240}, [dict(name=name, **quant)])
    )
    _, (paid,) = packet._bounded_keypoints(coded, [name])
    assert np.array_equal(t, paid["t"]) and np.array_equal(quant["kp"], paid["kp"])
    boxes = k6b.run_boxes(list(paid["kp"]))
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    try:
        ok, frame = cap.read()
        assert ok and abs(cap.get(cv2.CAP_PROP_FPS) - float(fps)) < 0.01
        assert frame.shape == (1080, 1920, 3)
        image = k6b.crop_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), boxes[0])
    finally:
        cap.release()
    metadata = dict(
        unit=name,
        reference_frame=0,
        reference_box=boxes[0].tolist(),
        pixels="full_prepared_conditioning_crop",
    )
    reference = portable(rates.encode_reference(image, metadata))
    _, decoded = packet._bounded_reference(reference)
    assert np.array_equal(image, decoded)
    return dict(
        pose_bytes=len(coded),
        reference_png_bytes=len(reference),
        pose_sha256=sha(coded),
        reference_section_sha256=sha(reference),
        decoded_reference_rgb_sha256=sha(decoded.tobytes()),
        pose_only_kbps=len(coded) * 8 * float(fps) / 240000,
        cold_pose_reference_only_kbps=(len(coded) + len(reference)) * 8 * float(fps) / 240000,
        note="Two independently decoded field payloads. Excludes base, ball, model, manifest and packet framing.",
    )


def measure(args):
    doc = json.loads((args.job / "result.json").read_text())
    role = json.loads(args.roles.read_text())
    assert doc["selected_ids"] == role["selected"] and doc["source_sha256"] == role["source_sha256"]
    assert sha(args.source.read_bytes()) == doc["source_sha256"] and doc["frames"] == 240
    artifact = args.job / "published.tar"
    assert sha(artifact.read_bytes()) == doc["output_archive_sha256"]
    support = np.ones(240, dtype=bool)
    rows = []
    with tarfile.open(artifact) as tf:
        for index, track in enumerate(doc["selected_ids"]):
            receipt = doc["tracks"][track]
            raw = tf.extractfile("publish/" + track.replace(":", "_") + ".npz").read()
            assert sha(raw) == receipt["npz_sha256"]
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                t, kp, fit, boxes = (z[k] for k in ["t", "kp", "fitted", "mask_bounds_xywh"])
                assert np.array_equal(t, np.arange(240))
                assert (
                    sha(kp.tobytes()) == receipt["kp_sha256"]
                    and sha(boxes.tobytes()) == receipt["mask_box_sha256"]
                )
                assert (
                    np.array_equal(np.flatnonzero(~fit), receipt["missing"])
                    and fit.sum() == receipt["fitted"]
                )
                assert np.array_equal(np.any(kp[:, :, 2] >= 0.5, axis=1), fit)
                support &= fit
                name = f"{doc['event']}_near" if index == 0 else f"{doc['event']}_far"
                row = dict(
                    track=track,
                    measured_role="near" if index == 0 else "far",
                    pose_ticks=int(fit.sum()),
                    eligible=bool(fit.all()),
                )
                if not index and fit.all():
                    row["signal"] = price(t, kp, args.source, name, args.fps)
                rows.append(row)
    groups = [
        dict(start=ticks[0], end=ticks[-1], count=len(ticks))
        for fitted, g in itertools.groupby(range(240), key=lambda i: bool(support[i]))
        if fitted and (ticks := list(g))
    ]
    return dict(
        event=doc["event"],
        job=args.job_id,
        archive_sha256=doc["output_archive_sha256"],
        source_sha256=doc["source_sha256"],
        fps=str(args.fps),
        total_ticks=240,
        available_both=int(support.sum()),
        longest_both=max((r["count"] for r in groups), default=0),
        eligible_240=bool(support.all()),
        eligible_120=any(r["count"] >= 120 for r in groups),
        runs=groups,
        tracks=rows,
        cpu_codec=f"OpenCV {cv2.__version__}, portable zlib REF1 and KPS2; dimensions 512×784.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "roles", "job", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--fps", type=Fraction, required=True)
    p.add_argument("--job-id", required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(measure(a), indent=2) + "\n")
