"""Try bounded SVT VBR settings to price a saved streak at the target rate."""
import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import time
from fractions import Fraction
from pathlib import Path

from experiments.tennis import k5b_trackers as trackers
from experiments.tennis.k5d_encoder import convert_rgb
from src.codecs import svtav1
from src.pointstream import ball, ball_stream

WIDTH, HEIGHT, FPS, FULL = 1280, 720, 30, 292
ORIGINAL = "publish/svt-62/stream.psbl"
BASELINE = "publish/svt-62/baseline.psbl"
TOOL_HASHES = {
    "SvtAv1EncApp": "4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4",
    "dav1d": "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6",
    "ffmpeg": "7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member(archive, name):
    file = archive.extractfile(name)
    if file is None:
        raise ValueError("Missing original packet: " + name)
    return file.read()


def run(args):
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("CPU isolation or affinity failed")
    tools = {name: svtav1.tool(name) for name in TOOL_HASHES}
    if any(tools[k]["sha256"] != v for k, v in TOOL_HASHES.items()):
        raise ValueError("Native tools differ from historic producer")
    with tarfile.open(args.original) as archive:
        original, baseline = member(archive, ORIGINAL), member(archive, BASELINE)
    if len(original) != 77970 or len(baseline) != 76720:
        raise ValueError("Historical packet sizes differ")
    stream = ball_stream.unpack(original)
    old_baseline = ball_stream.unpack(baseline)
    if stream.native != old_baseline.native or stream.frames != FULL:
        raise ValueError("Saved arms do not share the native stream")
    track, apps = ball.decode_frames(stream.ball_payload, FULL, WIDTH, HEIGHT)
    payload = ball.encode_frames(track[:args.frames], apps[:args.frames], WIDTH, HEIGHT)
    setup_elapsed = time.monotonic() - started
    source_started = time.monotonic()
    clip = next(c for c in trackers.tracknet_clips(args.source, [9])
                if c["id"] == "tracknet/game9/Clip5")
    if len(clip["rows"]) != FULL:
        raise ValueError("Unexpected source length")
    clip["rows"] = clip["rows"][:args.frames]
    source = trackers.tracknet_frames(args.source, clip)
    if source.shape != (args.frames, HEIGHT, WIDTH, 3):
        raise ValueError("Source geometry differs")
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    stage = Path(os.environ["PS_STAGE_DIR"])
    work = scratch / "matched-ball"
    work.mkdir()
    yuv = work / "source.yuv"
    convert_rgb(tools["ffmpeg"]["path"], source, yuv, Fraction(FPS))
    del source
    source_seconds = time.monotonic() - source_started
    records = []
    bounds = [20, 80]
    target = len(baseline) if args.frames == FULL else None
    best = None
    for iteration in range(8 if target else 2):
        crf = bounds[iteration] if iteration < 2 else (bounds[0] + bounds[1]) // 2
        if iteration >= 2 and crf in bounds:
            break
        ivf = work / f"candidate-{iteration}.ivf"
        command = svtav1.encode_command(tools["SvtAv1EncApp"]["path"], yuv, ivf,
            width=WIDTH, height=HEIGHT, fps=Fraction(FPS), frames=args.frames,
            crf=crf, preset=4, full_range=False)
        rc_index = command.index("--rc")
        command[rc_index:rc_index + 4] = ["--rc", "1", "--tbr", str(crf)]
        command[command.index("--keyint") + 1] = "300"
        encode_started = time.monotonic()
        process = subprocess.run(command, check=True, capture_output=True, timeout=200)
        encode_seconds = time.monotonic() - encode_started
        if not ivf.is_file():
            raise ValueError("SVT did not emit a stream: " + process.stderr.decode(errors="replace")[-2000:])
        native = ivf.read_bytes()
        if len(svtav1.ivf_frames(native)) != args.frames:
            raise ValueError("Encoded IVF does not contain every frame")
        packet = ball_stream.Stream(WIDTH, HEIGHT, Fraction(FPS), args.frames,
                                    stream.codec, stream.resource_id, native, payload)
        packed = ball_stream.pack(packet)
        if ball_stream.unpack(packed).ball_payload != payload:
            raise ValueError("Packet changed fitted sender symbols")
        packet_path = work / f"candidate-{iteration}.psbl"
        packet_path.write_bytes(packed)
        rec = {"target_kbps": crf, "native_bytes": len(native), "packet_bytes": len(packed),
               "packet_sha256": sha(packet_path), "command": command, "encode_seconds": encode_seconds}
        records.append(rec)
        if target:
            if best is None or abs(len(packed) - target) < abs(best["packet_bytes"] - target):
                best = rec
            if abs(len(packed) - target) / target <= .005:
                break
            if iteration == 1 and not records[0]["packet_bytes"] <= target <= records[1]["packet_bytes"]:
                break
            if iteration >= 2:
                if len(packed) < target:
                    bounds[0] = crf
                else:
                    bounds[1] = crf
    publish = scratch / "publish"
    publish.mkdir()
    for i in range(len(records)):
        (publish / f"candidate-{i}.psbl").write_bytes((work / f"candidate-{i}.psbl").read_bytes())
    (publish / "historical-baseline.psbl").write_bytes(baseline)
    (publish / "historical-streak.psbl").write_bytes(original)
    hash_started = time.monotonic()
    source_sha, original_sha = sha(args.source), sha(args.original)
    hash_seconds = time.monotonic() - hash_started
    report = {"stage": os.environ["PS_STAGE"], "frames": args.frames, "fps": FPS,
              "source_archive_sha256": source_sha, "original_archive_sha256": original_sha,
              "setup_seconds": setup_elapsed, "source_prepare_seconds": source_seconds,
              "hash_seconds": hash_seconds,
              "original_streak_sha256": hashlib.sha256(original).hexdigest(),
              "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
              "source_payload_sha256": hashlib.sha256(stream.ball_payload).hexdigest(),
              "candidate_payload_sha256": hashlib.sha256(payload).hexdigest(),
              "target_bytes": target, "trials": records, "best": best,
              "within_half_percent": bool(best and abs(best["packet_bytes"] - target) / target <= .005),
              "tools": tools, "elapsed_seconds": time.monotonic() - started}
    data = json.dumps(report, indent=2) + "\n"
    (publish / "report.json").write_text(data)
    (stage / "matched-ball-report.json").write_text(data)


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    report = json.loads((stage / "matched-ball-report.json").read_text())
    n = 24 if os.environ["PS_STAGE"] == "smoke" else FULL
    checks = {"frames": report["frames"] == n, "cadence": report["fps"] == FPS,
              "source": report["source_archive_sha256"] == "bb45372d011ef4092d78173169c8ad6280d7dbdb9998efaf2b1c72a63ca1dd18",
              "original": report["original_archive_sha256"] == "e939b3e79e5fedb534ae135148045eb490d9d7e9aca3c92c5bb0edbf1de53e0d",
              "tools": all(report["tools"][k]["sha256"] == v for k, v in TOOL_HASHES.items()),
              "trials": len(report["trials"]) >= 2}
    if n == 24:
        checks["monotonic"] = report["trials"][0]["packet_bytes"] <= report["trials"][1]["packet_bytes"]
        projected = (report["setup_seconds"] + report["hash_seconds"] + 120
                     + 1.5 * (FULL / 24) * (report["source_prepare_seconds"]
                     + 8 * max(row["encode_seconds"] for row in report["trials"])))
        checks["forecast"] = projected < 800
    else:
        checks["same_payload"] = report["source_payload_sha256"] == report["candidate_payload_sha256"]
    with tarfile.open(stage / "published.tar") as archive:
        packets = []
        for index, trial in enumerate(report["trials"]):
            data = member(archive, f"publish/candidate-{index}.psbl")
            packets.append(ball_stream.unpack(data))
            if hashlib.sha256(data).hexdigest() != trial["packet_sha256"]:
                raise ValueError("Published packet hash mismatch")
        if any(packet.frames != n for packet in packets):
            raise ValueError("Published packet frame count differs")
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            ivf, yuv = Path(tmp) / "test.ivf", Path(tmp) / "test.yuv"
            ivf.write_bytes(packets[-1].native)
            command = svtav1.decode_command(report["tools"]["dav1d"]["path"], ivf, yuv, threads=1)
            subprocess.run(command, check=True, capture_output=True, timeout=100)
            checks["native_decode"] = yuv.stat().st_size == n * WIDTH * HEIGHT * 3 // 2
    result = {"passed": all(checks.values()), "checks": checks}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result) + "\n")
    if not result["passed"]:
        raise ValueError("Failed checks: " + str(checks))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--original", type=Path)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
