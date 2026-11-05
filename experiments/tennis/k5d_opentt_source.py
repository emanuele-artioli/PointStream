"""OpenTTGames source-only WASB qualification, gated by an independent join witness.

No coding experiment. This worker is preparation: admission requires a reviewed
frame/coordinate witness and an independently approved budget. Unknown frames are
never counted as absent. All 240 full images reach the temporal judge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_source_oracle as identity
from experiments.tennis.k5d_analysis import score_frames

WASB_SHA = "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
CONFIG_SHA = "6dbcad60e207bf684d0094f59aa922b107be1c1ce83998dc5b167ed59b0d1e0d"


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def metadata(manifest: Path, manifest_sha: str, clip: str, archive: Path) -> dict[str, Any]:
    if identity.sha(manifest) != manifest_sha:
        raise ValueError("selection manifest hash mismatch")
    selection = json.loads(manifest.read_text(), object_pairs_hook=unique_pairs)
    matches = [x for x in selection["items"] if x["id"] == clip]
    if len(matches) != 1 or clip not in {f"test_{i}" for i in range(1, 8)}:
        raise ValueError("unregistered source context")
    expected = matches[0]
    if identity.sha(archive) != expected["label_archive_sha256"]:
        raise ValueError("label archive hash mismatch")
    with zipfile.ZipFile(archive) as z:
        if z.namelist().count("ball_markup.json") != 1:
            raise ValueError("missing or duplicate ball markup")
        raw = z.read("ball_markup.json")
    if hashlib.sha256(raw).hexdigest() != expected["ball_member_sha256"]:
        raise ValueError("ball markup hash mismatch")
    table = json.loads(raw, object_pairs_hook=unique_pairs)
    start = expected["start_zero_based_provisional"]
    selected = []
    for key, value in table.items():
        frame = int(key)
        if str(frame) != key:
            raise ValueError("noncanonical frame key")
        if not start <= frame < start + 240:
            continue
        x, y = value["x"], value["y"]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError("invalid source coordinates")
        if (x, y) == (-1, -1):
            kind = 0
        elif (x == -1 and 0 <= y < 1080) or (y == -1 and 0 <= x < 1920):
            kind = -1
        elif 0 <= x < 1920 and 0 <= y < 1080:
            kind = 1
        else:
            raise ValueError("invalid source coordinate geometry")
        selected.append({"frame": frame, "x": x, "y": y, "kind": kind})
    selected.sort(key=lambda x: x["frame"])
    if selected != expected["labels"] or len({x["frame"] for x in selected}) != len(selected):
        raise ValueError("selected rows differ from frozen manifest")
    if (sum(x["kind"] == 1 for x in selected) != expected["visible"]
            or sum(x["kind"] == 0 for x in selected) != expected["absent"]
            or 240 - sum(x["kind"] != -1 for x in selected) != expected["unknown"]):
        raise ValueError("selected label counts differ")
    return expected


def join_witness(path: Path, expected_sha: str, clip: dict[str, Any], evidence: Path) -> dict[str, Any]:
    if identity.sha(path) != expected_sha:
        raise ValueError("join witness SHA256 mismatch")
    witness = json.loads(path.read_text())
    digest = witness.get("evidence_sha256")
    if (witness.get("schema_version") != 1 or witness.get("passed") is not True
            or witness.get("clip") != clip["id"] or witness.get("video_sha256") != clip["video_sha256"]
            or witness.get("ball_markup_sha256") != clip["ball_member_sha256"]
            or witness.get("annotation_to_zero_based_offset") != 0 or not witness.get("reviewer")
            or not witness.get("method") or not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("missing reviewed independent zero-offset frame/label witness")
    if identity.sha(evidence) != digest:
        raise ValueError("reviewed join evidence SHA256 mismatch")
    return witness


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch

    from experiments.tennis import k5b_trackers as trk

    started = time.monotonic()
    if len(os.sched_getaffinity(0)) != 1 or os.environ.get("PS_CPU_ALLOWANCE") != "1" or torch.cuda.device_count() != 1:
        raise ValueError("requires one CPU and one visible CUDA device")
    clip = metadata(args.selection, args.selection_sha256, args.clip, args.labels)
    # Witness is reviewed before pixels or model calls; no score-selected offset.
    witness = join_witness(args.join_witness, args.join_witness_sha256, clip, args.join_evidence)
    if identity.sha(args.video) != clip["video_sha256"]:
        raise ValueError("video SHA256 mismatch")
    if args.wasb_weights_sha256 != WASB_SHA or args.wasb_config_sha256 != CONFIG_SHA:
        raise ValueError("WASB checkpoint or config is not the frozen tennis judge")
    if identity.tree_sha(args.wasb_code) != args.wasb_code_sha256:
        raise ValueError("WASB tree hash mismatch")
    identity.bind_config(args.wasb_code, args.wasb_config_sha256)
    if identity.sha(args.wasb_config) != args.wasb_config_sha256:
        raise ValueError("WASB config hash mismatch")
    if identity.sha(args.wasb_weights / "wasb_tennis_best.pth.tar") != args.wasb_weights_sha256:
        raise ValueError("WASB checkpoint hash mismatch")
    start = clip["start_zero_based_provisional"]
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError("video cannot be opened")
    try:
        if (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != 1920 or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != 1080
                or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != clip["video_frames"]
                or abs(capture.get(cv2.CAP_PROP_FPS) - 120.) > .001):
            raise ValueError("video metadata differs from frozen selection")
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, start) or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != start:
            raise ValueError("video seek did not establish the requested frame")
        frames = []
        hashes = []
        for frame_index in range(start, start + 240):
            if round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame_index:
                raise ValueError("noncontiguous frame index")
            ok, bgr = capture.read()
            if not ok or bgr.shape != (1080, 1920, 3):
                raise ValueError("truncated or wrong-size context")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (1280, 720), interpolation=cv2.INTER_AREA)
            frames.append(resized)
            hashes.append(hashlib.sha256(resized.tobytes()).hexdigest())
    finally:
        capture.release()
    pixels = np.stack(frames)
    zoo = trk.Zoo(args.wasb_code, args.wasb_weights, names=("wasb-step1",))
    if next(zoo.settings["wasb-step1"]["net"].parameters()).device.type != "cuda":
        raise ValueError("WASB is not on the claimed CUDA device")
    predictions = zoo.run("wasb-step1", pixels)
    lookup = {r["frame"]: r for r in clip["labels"]}
    labels = []
    for idx in range(start, start + 240):
        row = lookup.get(idx)
        labels.append((-1, 0., 0.) if row is None else (row["kind"], row["x"] * 2 / 3, row["y"] * 2 / 3))
    scores = score_frames(labels, predictions)
    rows = [{"frame": start + i, "rgb_sha256": hashes[i], "label_kind": label[0],
             "label_x": label[1], "label_y": label[2], "detected": bool(pred[0]),
             "x": float(pred[1]), "y": float(pred[2])}
            for i, (label, pred) in enumerate(zip(labels, predictions))]
    recall = scores["recall10"]
    output = {"schema_version": 1, "clip": args.clip, "start_zero_based": start, "frames": 240,
              "source_fps": "120/1", "source_size": [1920, 1080], "evaluation_size": [1280, 720],
              "score_radius_evaluation_pixels": 10, "source_pixel_radius": 15,
              "source_scope": "retrospective test footage; G1e motion-use exposure; tennis WASB cross-sport transfer",
              "label_scope": "model-aided labels; unknown excludes missing and partially missing; test_7 absence reliability disputed",
              "frame_join": "zero offset, independently reviewed witness; OpenCV presentation order",
              "identities": {"selection_sha256": args.selection_sha256, "video_sha256": clip["video_sha256"],
                             "label_archive_sha256": clip["label_archive_sha256"], "join_witness_sha256": args.join_witness_sha256,
                             "wasb_code_tree_sha256": args.wasb_code_sha256, "wasb_config_sha256": args.wasb_config_sha256,
                             "wasb_checkpoint_sha256": args.wasb_weights_sha256},
              "join_witness": witness, "scores": scores, "qualified": recall is not None and recall >= .8,
              "device": torch.cuda.get_device_name(0), "elapsed_seconds": time.monotonic() - started, "rows": rows}
    validate_report(output, clip)
    return output


def validate_report(report: dict[str, Any], clip: dict[str, Any]) -> dict[str, Any]:
    if report["clip"] != clip["id"] or report["frames"] != 240 or report["start_zero_based"] != clip["start_zero_based_provisional"]:
        raise ValueError("context mismatch")
    rows = report["rows"]
    if len(rows) != 240 or [r["frame"] for r in rows] != list(range(report["start_zero_based"], report["start_zero_based"] + 240)):
        raise ValueError("incomplete frame rows")
    labels = [(r["label_kind"], r["label_x"], r["label_y"]) for r in rows]
    predictions = [(r["detected"], r["x"], r["y"]) for r in rows]
    if any(len(r["rgb_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in r["rgb_sha256"]) for r in rows):
        raise ValueError("missing image hash")
    lookup = {r["frame"]: r for r in clip["labels"]}
    expected = []
    for idx in range(report["start_zero_based"], report["start_zero_based"] + 240):
        row = lookup.get(idx)
        expected.append((-1, 0., 0.) if row is None else (row["kind"], row["x"] * 2 / 3, row["y"] * 2 / 3))
    if labels != expected:
        raise ValueError("labels differ from frozen frame join")
    scores = score_frames(labels, predictions)
    if scores != report["scores"] or report["qualified"] != (scores["recall10"] is not None and scores["recall10"] >= .8):
        raise ValueError("incorrect source qualification")
    if (scores["visible"], scores["absent"], scores["unknown"]) != (clip["visible"], clip["absent"], clip["unknown"]):
        raise ValueError("label denominator differs from registered context")
    return {"qualified": report["qualified"], "scores": scores}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "video", "labels", "join-witness", "join-evidence", "wasb-code", "wasb-weights", "wasb-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("selection-sha256", "join-witness-sha256", "wasb-code-sha256", "wasb-weights-sha256", "wasb-config-sha256", "clip"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if not os.environ.get("PS_SCRATCH_DIR"):
        parser.error("PS_SCRATCH_DIR is required")
    identity.validate_output(args.output, Path(os.environ["PS_SCRATCH_DIR"]),
                             [args.selection, args.video, args.labels, args.join_witness, args.join_evidence, args.wasb_code, args.wasb_weights, args.wasb_config])
    try:
        report = run(args)
    except Exception as exc:
        report = {"schema_version": 1, "clip": args.clip, "qualified": False, "error": f"{type(exc).__name__}: {exc}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(report, out, allow_nan=False, indent=2)
        out.write("\n")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
