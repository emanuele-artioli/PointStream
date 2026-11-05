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
import math
import tarfile
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

from experiments.tennis import k5d_source_oracle as identity
from experiments.tennis.k5d_analysis import score_frames

WASB_SHA = "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
CONFIG_SHA = "6dbcad60e207bf684d0094f59aa922b107be1c1ce83998dc5b167ed59b0d1e0d"
WASB_TREE_SHA = "d456047c70e9dbd3a2f7196e2d1991a069393ab68d7466d0ee7ecfd23360cab0"
SELECTION_SHA = "b104b87aefae92e3a3ea7766bc51e3b34f6c3c7a1f071bbbbca7eeb218984877"


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



def source_rgb(capture: Any, cv2: Any, start: int, count: int) -> Iterator[Any]:
    """Use the identical seek, read and BGR-to-RGB path for witness and source."""
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, start) or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != start:
        raise ValueError("video seek did not establish the requested frame")
    for index in range(start, start + count):
        if round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index:
            raise ValueError("noncontiguous frame index")
        ok, bgr = capture.read()
        if not ok or bgr.shape != (1080, 1920, 3):
            raise ValueError("truncated or wrong-size context")
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def validate_decode_witness(record: dict[str, Any], evidence_text: str, witness: dict[str, Any]) -> None:
    if hashlib.sha256(evidence_text.encode()).hexdigest() != witness["evidence_sha256"]:
        raise ValueError("decode witness evidence hash mismatch")
    evidence = json.loads(evidence_text)
    expected = evidence["frames"]
    if (evidence.get("clip") != "test_2" or evidence.get("fast_seek_raw_parity") is not True
            or [r["frame_index"] for r in expected] != [1085, 1086, 1087]
            or record.get("frames") != [{"frame_index": r["frame_index"], "raw_rgb_sha256": r["raw_rgb_sha256"]} for r in expected]
            or record.get("passed") is not True):
        raise ValueError("OpenCV/FFmpeg witness RGB mismatch; decode/join instrumentation gap")


def check_decode_witness(capture: Any, cv2: Any, evidence_text: str, witness: dict[str, Any]) -> dict[str, Any]:
    frames = source_rgb(capture, cv2, 1085, 3)
    record = {"passed": True, "frames": [{"frame_index": 1085 + i, "raw_rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest()}
                                          for i, frame in enumerate(frames)]}
    validate_decode_witness(record, evidence_text, witness)
    return record

def run(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch

    from experiments.tennis import k5b_trackers as trk

    started = time.monotonic()
    if args.clip != "test_2" or args.frames not in (16, 240) or {"smoke": 16, "full": 240}.get(os.environ.get("PS_STAGE", "")) != args.frames:
        raise ValueError("only the frozen test_2 16/240 stages are allowed")
    if len(os.sched_getaffinity(0)) != 1 or os.environ.get("PS_CPU_ALLOWANCE") != "1" or torch.cuda.device_count() != 1:
        raise ValueError("requires one CPU and one visible CUDA device")
    if args.selection_sha256 != SELECTION_SHA or args.wasb_code_sha256 != WASB_TREE_SHA:
        raise ValueError("selection or WASB code identity differs from frozen contract")
    clip = metadata(args.selection, args.selection_sha256, args.clip, args.labels)
    if (clip["start_zero_based_provisional"], clip["visible"], clip["absent"], clip["unknown"]) != (1066, 85, 6, 149):
        raise ValueError("frozen test_2 context differs")
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
        evidence_text = args.join_evidence.read_text()
        decode_witness = check_decode_witness(capture, cv2, evidence_text, witness)
        fixed_preflight_seconds = time.monotonic() - started
        io_started = time.monotonic()
        frames = [cv2.resize(rgb, (1280, 720), interpolation=cv2.INTER_AREA)
                  for rgb in source_rgb(capture, cv2, start, args.frames)]
        hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
    finally:
        capture.release()
    pixels = np.stack(frames)
    source_io_seconds = time.monotonic() - io_started
    model_started = time.monotonic()
    zoo = trk.Zoo(args.wasb_code, args.wasb_weights, names=("wasb-step1",))
    if next(zoo.settings["wasb-step1"]["net"].parameters()).device.type != "cuda":
        raise ValueError("WASB is not on the claimed CUDA device")
    model_init_seconds = time.monotonic() - model_started
    judge_started = time.monotonic()
    predictions = zoo.run("wasb-step1", pixels)
    judge_seconds = time.monotonic() - judge_started
    if len(predictions) != args.frames:
        raise ValueError("incomplete judge predictions")
    lookup = {r["frame"]: r for r in clip["labels"]}
    labels = []
    for idx in range(start, start + args.frames):
        row = lookup.get(idx)
        labels.append((-1, 0., 0.) if row is None else (row["kind"], row["x"] * 2 / 3, row["y"] * 2 / 3))
    scores = score_frames(labels, predictions)
    rows = [{"frame": start + i, "rgb_sha256": hashes[i], "label_kind": label[0],
             "label_x": label[1], "label_y": label[2], "detected": bool(pred[0]),
             "x": float(pred[1]), "y": float(pred[2])}
            for i, (label, pred) in enumerate(zip(labels, predictions))]
    recall = scores["recall10"]
    output = {"schema_version": 1, "clip": args.clip, "start_zero_based": start, "frames": args.frames,
              "source_fps": "120/1", "source_size": [1920, 1080], "evaluation_size": [1280, 720],
              "score_radius_evaluation_pixels": 10, "source_pixel_radius": 15,
              "source_scope": "retrospective test footage; G1e motion-use exposure; tennis WASB cross-sport transfer",
              "label_scope": "model-aided labels; unknown excludes missing and partially missing; test_7 absence reliability disputed",
              "frame_join": "zero offset, independently reviewed witness; OpenCV presentation order",
              "identities": {"selection_sha256": args.selection_sha256, "video_sha256": clip["video_sha256"],
                             "label_archive_sha256": clip["label_archive_sha256"], "join_witness_sha256": args.join_witness_sha256,
                             "wasb_code_tree_sha256": args.wasb_code_sha256, "wasb_config_sha256": args.wasb_config_sha256,
                             "wasb_checkpoint_sha256": args.wasb_weights_sha256},
              "join_witness": witness, "decode_witness": decode_witness, "join_evidence_text": evidence_text, "scores": scores, "qualified": (args.frames == 240 and recall is not None and recall >= .8 and scores["false_per_negative"] is not None and scores["false_per_negative"] <= .05),
              "timing": {"preflight_fixed_seconds": fixed_preflight_seconds, "source_io_seconds": source_io_seconds, "model_init_seconds": model_init_seconds, "judge_seconds": judge_seconds},
              "device": torch.cuda.get_device_name(0), "elapsed_seconds": time.monotonic() - started, "rows": rows}
    validate_report(output, clip)
    return output


def validate_report(report: dict[str, Any], clip: dict[str, Any]) -> dict[str, Any]:
    validate_decode_witness(report["decode_witness"], report["join_evidence_text"], report["join_witness"])
    frames = report["frames"]
    if frames not in (16, 240) or report["clip"] != clip["id"] or report["start_zero_based"] != clip["start_zero_based_provisional"]:
        raise ValueError("context mismatch")
    rows = report["rows"]
    if len(rows) != frames or [r["frame"] for r in rows] != list(range(report["start_zero_based"], report["start_zero_based"] + frames)):
        raise ValueError("incomplete frame rows")
    labels = [(r["label_kind"], r["label_x"], r["label_y"]) for r in rows]
    predictions = [(r["detected"], r["x"], r["y"]) for r in rows]
    if any(type(r["detected"]) is not bool or any(isinstance(r[k], bool) or not isinstance(r[k], (int, float)) or not math.isfinite(r[k]) for k in ("x", "y")) for r in rows):
        raise ValueError("invalid judge prediction")
    if any(len(r["rgb_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in r["rgb_sha256"]) for r in rows):
        raise ValueError("missing image hash")
    lookup = {r["frame"]: r for r in clip["labels"]}
    expected = []
    for idx in range(report["start_zero_based"], report["start_zero_based"] + frames):
        row = lookup.get(idx)
        expected.append((-1, 0., 0.) if row is None else (row["kind"], row["x"] * 2 / 3, row["y"] * 2 / 3))
    if labels != expected:
        raise ValueError("labels differ from frozen frame join")
    scores = score_frames(labels, predictions)
    if scores != report["scores"] or report["qualified"] != (frames == 240 and scores["recall10"] is not None and scores["recall10"] >= .8 and scores["false_per_negative"] is not None and scores["false_per_negative"] <= .05):
        raise ValueError("incorrect source qualification")
    if frames == 240 and (scores["visible"], scores["absent"], scores["unknown"]) != (clip["visible"], clip["absent"], clip["unknown"]):
        raise ValueError("label denominator differs from registered context")
    return {"qualified": report["qualified"], "scores": scores}



def forecast_smoke(report: dict[str, Any], execution: dict[str, Any], clock: dict[str, Any], now: float) -> dict[str, float]:
    """Conservative forecast from measured fixed and per-frame work, not 15x startup."""
    timings = report["timing"]
    keys = ("preflight_fixed_seconds", "source_io_seconds", "model_init_seconds", "judge_seconds")
    values = [timings[k] for k in keys]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("invalid timing")
    elapsed = execution["seconds"]
    if execution["exit_code"] != 0 or not math.isfinite(elapsed) or elapsed <= 0 or not clock["started"] <= clock["phase_started"] <= now < clock["budget_end"]:
        raise ValueError("invalid execution or budget clock")
    variable = timings["source_io_seconds"] + timings["judge_seconds"]
    fixed = max(elapsed - variable, timings["preflight_fixed_seconds"] + timings["model_init_seconds"])
    estimate = 1.25 * (fixed + variable * 240 / 16)
    remaining = clock["budget_end"] - now
    # 120 seconds is the complete full-stage ceiling; 10 seconds is retained for publication.
    if estimate + 10 > 120 or estimate + 10 > remaining:
        raise ValueError("full forecast exceeds 120-second stage or remaining campaign budget")
    return {"fixed_seconds": fixed, "io_seconds_per_frame": timings["source_io_seconds"] / 16,
            "judge_seconds_per_frame": timings["judge_seconds"] / 16, "margin": 1.25,
            "full_estimate_seconds": estimate, "publication_reserve_seconds": 10, "remaining_seconds": remaining}


def validate_smoke() -> None:
    target = Path(os.environ["PS_VALIDATION_PATH"])
    try:
        stage = Path(os.environ["PS_STAGE_DIR"])
        with tarfile.open(stage / "published.tar", "r:") as bundle:
            members = [m for m in bundle.getmembers() if m.isfile()]
            if len(members) != 1 or members[0].name != "publish/report.json" or members[0].size > 2_000_000:
                raise ValueError("unexpected smoke archive")
            stream = bundle.extractfile(members[0])
            if stream is None:
                raise ValueError("missing report")
            report = json.load(stream)
        selection = json.loads((Path(__file__).resolve().parents[2] / "docs/experiments/k5d-inputs/opentt-source-selection-20261104.json").read_text())
        clip = next(x for x in selection["items"] if x["id"] == "test_2")
        if report["frames"] != 16 or report["clip"] != "test_2" or report["start_zero_based"] != 1066:
            raise ValueError("wrong smoke context")
        validate_report(report, clip)
        ids = report["identities"]
        if (ids["selection_sha256"] != SELECTION_SHA
                or ids["video_sha256"] != clip["video_sha256"]
                or ids["label_archive_sha256"] != clip["label_archive_sha256"]
                or ids["wasb_code_tree_sha256"] != WASB_TREE_SHA
                or ids["wasb_checkpoint_sha256"] != WASB_SHA or ids["wasb_config_sha256"] != CONFIG_SHA
                or report["source_fps"] != "120/1" or report["evaluation_size"] != [1280, 720]):
            raise ValueError("source or tennis judge identity differs")
        witness = report["join_witness"]
        validate_decode_witness(report["decode_witness"], report["join_evidence_text"], witness)
        if witness.get("passed") is not True or witness.get("annotation_to_zero_based_offset") != 0 or not witness.get("reviewer"):
            raise ValueError("missing reviewed join")
        execution = json.loads((stage / "execution.json").read_text())
        clock = json.loads((stage / "dispatch.json").read_text())["campaign_clock"]
        estimate = forecast_smoke(report, execution, clock, time.monotonic())
        answer = {"passed": True, "checks": {"source_join": True, "all_16_full_images": True,
                  "score_recomputed": True, "measured_full_forecast": True}, "forecast": estimate,
                  "scope": "test_2 source qualification only; smoke is not a qualification result"}
    except Exception as exc:
        target.write_text(json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        raise
    target.write_text(json.dumps(answer) + "\n")

def main() -> int:
    if sys.argv[1:] == ["--validate-smoke"]:
        validate_smoke()
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "video", "labels", "join-witness", "join-evidence", "wasb-code", "wasb-weights", "wasb-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("selection-sha256", "join-witness-sha256", "wasb-code-sha256", "wasb-weights-sha256", "wasb-config-sha256", "clip"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("PS_SCRATCH_DIR", "/nonexistent")) / "publish/report.json")
    parser.add_argument("--frames", type=int, required=True, choices=(16, 240))
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
    return 1 if "error" in report else 0


if __name__ == "__main__":
    sys.exit(main())
