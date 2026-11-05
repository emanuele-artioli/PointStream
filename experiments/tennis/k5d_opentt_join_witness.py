"""CPU-only OpenTT source frame/label join witness; NEVER a judge or auto-approval.

A sequential first-frame reference and fast-seek decode must match at exact
container PTS. Rendered contact sheets remain for independent human review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_opentt_source, k5d_source_oracle as identity

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/experiments/k5d-inputs/opentt-source-selection-20261104.json"
SELECT = ROOT / "docs/experiments/k5d-inputs/opentt-join-witness-selection-20261104.json"
SOURCE_SHA = "b104b87aefae92e3a3ea7766bc51e3b34f6c3c7a1f071bbbbca7eeb218984877"
SELECT_SHA = "bcd51345c39965f70f0d4c051238de54b87670720002c5ea488fa8d93c7661d7"
FFMPEG_SHA = "7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2"
FFPROBE_SHA = "f4e72e0094397518602881be95f16a1d60ec41b03ecfb0aaa6f8e45ffa14583a"


def selection(clip_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if identity.sha(SOURCE) != SOURCE_SHA or identity.sha(SELECT) != SELECT_SHA:
        raise ValueError("source or witness selection differs from frozen hash")
    context = [x for x in json.loads(SOURCE.read_text())["items"] if x["id"] == clip_id]
    witness = [x for x in json.loads(SELECT.read_text())["items"] if x["id"] == clip_id]
    if len(context) != 1 or len(witness) != 1 or witness[0].get("eligible_triplets", 0) < 1:
        raise ValueError("unregistered or ambiguous witness")
    c, w = context[0], witness[0]
    if (w["video_sha256"] != c["video_sha256"] or w["ball_markup_sha256"] != c["ball_member_sha256"]
            or w["frame_indices"] != list(range(w["center_frame_label_index"] - 1, w["center_frame_label_index"] + 2))):
        raise ValueError("witness is not bound to source selection")
    return c, w


def cadence(info: dict[str, Any], expected_frames: int) -> tuple[Fraction, list[int]]:
    streams = info.get("streams", [])
    if len(streams) != 1:
        raise ValueError("expected one video stream")
    s = streams[0]
    if ((s.get("width"), s.get("height")) != (1920, 1080) or int(s.get("nb_frames", 0)) != expected_frames
            or Fraction(s["avg_frame_rate"]) != 120 or Fraction(s["r_frame_rate"]) != 120):
        raise ValueError("video metadata differs from selection or 120fps")
    base = Fraction(s["time_base"])
    step = Fraction(1, 120) / base
    if step.denominator != 1 or step <= 0:
        raise ValueError("time base cannot represent 120fps exactly")
    frames = info.get("frames", [])
    if len(frames) != expected_frames:
        raise ValueError("actual decoded frame count differs")
    pts = [int(f["best_effort_timestamp"]) for f in frames]
    if any(p != pts[0] + i * step.numerator for i, p in enumerate(pts)):
        raise ValueError("presentation timestamps are not exact constant 120fps")
    return base, pts


def extract(ffmpeg: Path, video: Path, first: int, seek_second: int | None, base: Fraction,
            expected_pts: list[int]) -> tuple[list[bytes], list[str], list[int]]:
    offset = 0 if seek_second is None else seek_second * 120
    local = first - offset
    if local < 0:
        raise ValueError("invalid reference seek")
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "info", "-nostdin", "-threads", "1", "-filter_threads", "1", "-copyts"]
    if seek_second is not None:
        cmd += ["-ss", str(seek_second)]
    cmd += ["-i", str(video), "-an", "-vf", f"select=between(n\\,{local}\\,{local + 2}),showinfo", "-frames:v", "3",
            "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    done = subprocess.run(cmd, check=True, capture_output=True, timeout=250)
    frame_bytes = 1920 * 1080 * 3
    if len(done.stdout) != 3 * frame_bytes:
        raise ValueError("decoder did not return exactly three full source frames")
    log = done.stderr.decode("utf-8", "replace")
    base_matches = re.findall(r"config in time_base:\s*(\d+)/(\d+)", log)
    if not base_matches or Fraction(int(base_matches[0][0]), int(base_matches[0][1])) != base:
        raise ValueError("decoder filter time base differs from probe")
    pts = [int(x) for x in re.findall(r"\bn:\s*\d+\s+pts:\s*(-?\d+)", log)]
    if pts != expected_pts:
        raise ValueError("decoded PTS do not match source presentation indices")
    images = [done.stdout[i * frame_bytes:(i + 1) * frame_bytes] for i in range(3)]
    return images, cmd, pts


def write_artifacts(out: Path, clip: dict[str, Any], witness: dict[str, Any], images: list[bytes],
                    pts: list[int], base: Fraction) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    out.mkdir(parents=True, exist_ok=False)
    panels = []
    records = []
    for idx, raw, position, timestamp in zip(witness["frame_indices"], images, witness["coordinates"], pts):
        image = Image.frombytes("RGB", (1920, 1080), raw)
        target = out / f"frame-{idx}.png"
        image.save(target)
        rec = {"frame_index": idx, "pts": timestamp, "time_base": str(base), "raw_rgb_sha256": hashlib.sha256(raw).hexdigest(),
               "png_sha256": identity.sha(target), "png_bytes": target.stat().st_size, "label_xy_source": position}
        records.append(rec)
        panel = image.copy()
        draw = ImageDraw.Draw(panel)
        x, y = position
        draw.ellipse((x - 16, y - 16, x + 16, y + 16), outline="red", width=4)
        draw.line((x - 24, y, x + 24, y), fill="red", width=2)
        draw.line((x, y - 24, x, y + 24), fill="red", width=2)
        draw.rectangle((0, 0, 1200, 55), fill="black")
        draw.text((10, 12), f"{clip['id']} index={idx} pts={timestamp} label=({x},{y})", fill="white")
        panels.append(panel.resize((960, 540)))
    sheet = Image.new("RGB", (960 * 3, 540))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (960 * i, 0))
    contact = out / "contact-sheet.png"
    sheet.save(contact)
    return {"frames": records, "contact_sheet": {"file": contact.name, "sha256": identity.sha(contact), "bytes": contact.stat().st_size}}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(os.sched_getaffinity(0)) != 1 or os.environ.get("PS_CPU_ALLOWANCE") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("requires fleet-enforced one CPU and CUDA_VISIBLE_DEVICES empty")
    clip, witness = selection(args.clip)
    k5d_opentt_source.metadata(SOURCE, SOURCE_SHA, args.clip, args.labels)
    labels = {r["frame"]: r for r in clip["labels"]}
    for frame, position in zip(witness["frame_indices"], witness["coordinates"]):
        row = labels.get(frame)
        if row is None or row["kind"] != 1 or [row["x"], row["y"]] != position:
            raise ValueError("witness triplet labels differ from source selection")
    if identity.sha(args.video) != clip["video_sha256"] or args.video.stat().st_size != clip["video_size"]:
        raise ValueError("video identity differs")
    ffmpeg, ffprobe = Path(sys.prefix) / "bin/ffmpeg", Path(sys.prefix) / "bin/ffprobe"
    if identity.sha(ffmpeg) != FFMPEG_SHA or identity.sha(ffprobe) != FFPROBE_SHA:
        raise ValueError("packed native executable identity differs")
    probe = [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-threads", "1", "-show_streams", "-show_frames",
             "-show_entries", "stream=width,height,nb_frames,time_base,avg_frame_rate,r_frame_rate:frame=best_effort_timestamp",
             "-of", "json", str(args.video)]
    info = json.loads(subprocess.run(probe, check=True, capture_output=True, timeout=250).stdout)
    base, all_pts = cadence(info, clip["video_frames"])
    first = witness["frame_indices"][0]
    expected = all_pts[first:first + 3]
    sequential, slow_cmd, slow_pts = extract(ffmpeg, args.video, first, None, base, expected)
    seek_second = max(0, first // 120 - 1)
    fast, fast_cmd, fast_pts = extract(ffmpeg, args.video, first, seek_second, base, expected)
    if [hashlib.sha256(x).digest() for x in sequential] != [hashlib.sha256(x).digest() for x in fast]:
        raise ValueError("fast seek differs from first-frame sequential reference")
    if len({hashlib.sha256(x).digest() for x in sequential}) != 3:
        raise ValueError("adjacent decoded frames are byte-identical; witness ambiguous")
    result = write_artifacts(args.output, clip, witness, sequential, expected, base)
    result.update({"schema_version": 1, "clip": args.clip, "status": "REVIEW_REQUIRED; no label-origin conclusion",
                   "video_sha256": clip["video_sha256"], "video_bytes": clip["video_size"],
                   "ball_markup_sha256": clip["ball_member_sha256"], "selection_sha256": SOURCE_SHA,
                   "witness_selection_sha256": SELECT_SHA, "ffmpeg_sha256": FFMPEG_SHA, "ffprobe_sha256": FFPROBE_SHA,
                   "source_pts_first": all_pts[0], "source_frame_count": len(all_pts), "fps": "120/1",
                   "time_base": str(base), "sequential_pts": slow_pts, "seek_pts": fast_pts,
                   "sequential_command": slow_cmd, "seek_command": fast_cmd, "probe_command": probe,
                   "fast_seek_raw_parity": True, "claim": "decoder index consistency only; author annotation origin needs independent review"})
    (args.output / "witness.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def validate_smoke() -> None:
    target = Path(os.environ["PS_VALIDATION_PATH"])
    try:
        stage = Path(os.environ["PS_STAGE_DIR"])
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="opentt-witness-") as tmp:
            with tarfile.open(stage / "published.tar", "r:") as bundle:
                members = bundle.getmembers()
                if len(members) > 20 or sum(m.size for m in members) > 200_000_000 or any(not (m.isfile() or m.isdir()) for m in members):
                    raise ValueError("unexpected witness archive")
                bundle.extractall(tmp, filter="data")
            out = Path(tmp) / "publish"
            record = json.loads((out / "witness.json").read_text())
            if record["status"] != "REVIEW_REQUIRED; no label-origin conclusion" or not record["fast_seek_raw_parity"]:
                raise ValueError("witness is incomplete or claims approval")
            clip, w = selection(record["clip"])
            base = Fraction(record["time_base"])
            step = Fraction(1, 120) / base
            expected_pts = [record["source_pts_first"] + idx * step for idx in w["frame_indices"]]
            if (step.denominator != 1 or record["fps"] != "120/1" or record["source_frame_count"] != clip["video_frames"]
                    or record["video_sha256"] != clip["video_sha256"] or record["ffmpeg_sha256"] != FFMPEG_SHA
                    or record["ffprobe_sha256"] != FFPROBE_SHA or record["selection_sha256"] != SOURCE_SHA
                    or record["witness_selection_sha256"] != SELECT_SHA or record["sequential_pts"] != expected_pts
                    or record["seek_pts"] != expected_pts or [r["frame_index"] for r in record["frames"]] != w["frame_indices"]
                    or [r["pts"] for r in record["frames"]] != expected_pts):
                raise ValueError("witness identity, count or exact PTS mismatch")
            for row in record["frames"]:
                if identity.sha(out / f"frame-{row['frame_index']}.png") != row["png_sha256"]:
                    raise ValueError("frame artifact hash mismatch")
            if len(record["frames"]) != 3 or identity.sha(out / "contact-sheet.png") != record["contact_sheet"]["sha256"]:
                raise ValueError("contact sheet or frame coverage mismatch")
        answer = {"passed": True, "checks": {"three_exact_pts_frames": True, "fast_seek_matches_sequential": True,
                   "hash_bound_review_artifacts": True}, "scope": "machine consistency only; manual join review pending"}
    except Exception as exc:
        target.write_text(json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        raise
    target.write_text(json.dumps(answer) + "\n")


def main() -> None:
    if sys.argv[1:] == ["--validate-smoke"]:
        validate_smoke()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--mode", choices=("inspect", "no-additional-work"), required=True)
    args = parser.parse_args()
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    args.output = scratch / "publish"
    if args.mode == "no-additional-work":
        if os.environ.get("PS_STAGE") != "full":
            raise ValueError("no-work marker only allowed in full stage")
        args.output.mkdir()
        (args.output / "no-additional-work.json").write_text(json.dumps({"scope": "pilot completed in smoke; no second video or decoding"}) + "\n")
        return
    if os.environ.get("PS_STAGE") != "smoke":
        raise ValueError("pilot inspection only allowed in smoke stage")
    try:
        run(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        error = {"schema_version": 1, "clip": args.clip, "status": "STOP; no join conclusion",
                 "error": f"{type(exc).__name__}: {exc}"}
        (args.output / "witness-error.json").write_text(json.dumps(error) + "\n")
        raise


if __name__ == "__main__":
    main()
