"""Encode the actual /000-to-/002 intervening broadcast gap for rate feasibility."""
import argparse
import hashlib
import json
import os
import resource
import shutil
import struct
import subprocess
import tarfile
import time
import tempfile
from fractions import Fraction
from pathlib import Path

from src.codecs import svtav1

SOURCE_SHA = "676331e0f4843f81a250b24c8a2f7ac0bb119ead4ffad0f145651ffb91a43e66"
TOOLS = {
    "SvtAv1EncApp": "4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4",
    "dav1d": "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6",
    "ffmpeg": "7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2",
}
START, COUNT, FPS = 76866, 10825, 50
WIDTH, HEIGHT = 240, 136


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ivf(path, frames):
    raw = path.read_bytes()
    if raw[:4] != b"DKIF" or struct.unpack_from("<HHIIII", raw, 12) != (WIDTH, HEIGHT, FPS, 1, frames, 0):
        raise ValueError("Unexpected IVF header")
    pos, payload = 32, 0
    for tick in range(frames):
        size, stamp = struct.unpack_from("<IQ", raw, pos)
        if stamp != tick or pos + 12 + size > len(raw):
            raise ValueError("Missing or misordered IVF frame")
        payload += size
        pos += 12 + size
    if pos != len(raw):
        raise ValueError("Extra or incomplete IVF data")
    return len(raw), payload


def run(args):
    if len(os.sched_getaffinity(0)) != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("One-core CPU-only isolation required")
    if sha(args.source) != SOURCE_SHA:
        raise ValueError("Broadcast source identity differs")
    tools = {name: svtav1.tool(name) for name in (*TOOLS, "ffprobe")}
    if any(tools[name]["sha256"] != expected for name, expected in TOOLS.items()):
        raise ValueError("Pinned native tools differ")
    probe = subprocess.run([tools["ffprobe"]["path"], "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(args.source)],
                           text=True, capture_output=True, check=True, timeout=60)
    stream, = json.loads(probe.stdout)["streams"]
    if (stream["width"], stream["height"], stream["avg_frame_rate"], int(stream["nb_frames"])) != (1920, 1080, "50/1", 115505):
        raise ValueError("Broadcast geometry, cadence or frame count differs")
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    stage = Path(os.environ["PS_STAGE_DIR"])
    work = scratch / "gap"
    work.mkdir()
    started = time.monotonic()
    raw = work / "input.yuv"
    conversion = [tools["ffmpeg"]["path"], "-v", "error", "-nostdin", "-ss", str(START / FPS),
                  "-i", str(args.source), "-frames:v", str(args.frames),
                  "-vf", f"scale={WIDTH}:{HEIGHT}:flags=area:out_range=tv,format=yuv420p",
                  "-filter_threads", "1", "-fps_mode", "passthrough", "-threads", "1",
                  "-f", "rawvideo", "-pix_fmt", "yuv420p", str(raw)]
    subprocess.run(conversion, check=True, timeout=1600)
    conversion_seconds = time.monotonic() - started
    frame_size = WIDTH * HEIGHT * 3 // 2
    if raw.stat().st_size != frame_size * args.frames:
        raise ValueError("Gap conversion has incomplete frames")
    if args.frames == 240:
        selected = work / "boundary.yuv"
        exact = [tools["ffmpeg"]["path"], "-v", "error", "-nostdin", "-i", str(args.source),
                 "-vf", f"select=eq(n\\,{START}),scale={WIDTH}:{HEIGHT}:flags=area:out_range=tv,format=yuv420p",
                 "-frames:v", "1", "-filter_threads", "1", "-fps_mode", "passthrough", "-threads", "1",
                 "-f", "rawvideo", "-pix_fmt", "yuv420p", str(selected)]
        subprocess.run(exact, check=True, timeout=180)
        with raw.open("rb") as handle:
            if handle.read(frame_size) != selected.read_bytes():
                raise ValueError("Fast seek differs from exact frame index")
    codec = svtav1.code(raw, work, width=WIDTH, height=HEIGHT, fps=Fraction(FPS), frames=args.frames,
                        crf=70, preset=8, threads=1, full_range=False,
                        encoder=tools["SvtAv1EncApp"]["path"], decoder=tools["dav1d"]["path"], timeout=1500)
    stream_path, decoded_path = Path(codec["stream"]), Path(codec["decoded"])
    total, payload = validate_ivf(stream_path, args.frames)
    if codec["decoded_frames"] != args.frames or payload != codec["payload_bytes"]:
        raise ValueError("Decoder or IVF frame count differs")
    publish = scratch / "publish"
    publish.mkdir()
    shutil.copyfile(stream_path, publish / "fallback.ivf")
    budget = 565250
    reference = 243000
    report = {"source_sha256": SOURCE_SHA, "source_probe": stream, "source_start": START,
              "frames": args.frames, "fps": FPS, "geometry": [WIDTH, HEIGHT], "crf": 70, "preset": 8,
              "stage": os.environ["PS_STAGE"], "cpu_affinity": list(os.sched_getaffinity(0)),
              "tools": tools, "conversion": conversion, "bytes": total, "payload_bytes": payload,
              "stream_sha256": sha(stream_path), "decoded_sha256": sha(decoded_path),
              "decoded_frames": args.frames, "kbps": total * 8 * FPS / args.frames / 1000,
              "conversion_seconds": conversion_seconds,
              "encode_seconds": codec["encode_seconds"], "decode_seconds": codec["decode_seconds"],
              "native_cpu_seconds": codec["encode_cpu_seconds"] + codec["decode_cpu_seconds"],
              "elapsed_seconds": time.monotonic() - started,
              "usage_cpu_seconds": sum(getattr(r, "ru_utime") + getattr(r, "ru_stime") for r in
                                       (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN))),
              "total_budget_bytes_20kbps": budget, "reference_channel_bytes": reference,
              "remaining_after_references_and_gap_bytes": budget - reference - total if args.frames == COUNT else None}
    encoded = json.dumps(report, indent=2) + "\n"
    (publish / "report.json").write_text(encoded)
    (stage / "gap-report.json").write_text(encoded)


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    report = json.loads((stage / "gap-report.json").read_text())
    spec = json.loads((stage.parent / "spec.json").read_text())
    smoke = os.environ["PS_STAGE"] == "smoke"
    frames = 240 if smoke else COUNT
    checks = {"source": report["source_sha256"] == SOURCE_SHA, "start": report["source_start"] == START,
              "frames": report["frames"] == frames and report["decoded_frames"] == frames,
              "geometry": report["geometry"] == [WIDTH, HEIGHT] and report["fps"] == FPS,
              "settings": report["crf"] == 70 and report["preset"] == 8,
              "cpu": len(report["cpu_affinity"]) == 1,
              "tools": all(report["tools"][name]["sha256"] == expected for name, expected in TOOLS.items())}
    with tarfile.open(stage / "published.tar") as tar:
        data = tar.extractfile("publish/fallback.ivf").read()
    checks["stream"] = hashlib.sha256(data).hexdigest() == report["stream_sha256"] and len(data) == report["bytes"]
    checks["ivf"] = len(svtav1.ivf_frames(data)) == frames
    if smoke:
        with tempfile.TemporaryDirectory(prefix="gap-validator-", dir=os.environ["PS_SCRATCH_DIR"]) as temp:
            path = Path(temp) / "stream.ivf"
            decoded = Path(temp) / "decoded.yuv"
            path.write_bytes(data)
            subprocess.run(svtav1.decode_command(report["tools"]["dav1d"]["path"], path, decoded, threads=1),
                           check=True, timeout=90)
            checks["independent_decode"] = sha(decoded) == report["decoded_sha256"]
    projection = None
    if smoke:
        native = report["conversion_seconds"] + report["encode_seconds"] + report["decode_seconds"]
        projection = 1.5 * native * COUNT / 240 + 180
        checks["forecast"] = projection <= spec["full"]["seconds"]
    result = {"passed": all(checks.values()), "checks": checks, "projected_full_seconds": projection}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--frames", type=int, choices=(240, COUNT))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate()
    elif args.source is not None and args.frames is not None:
        run(args)
    else:
        parser.error("source and frames are required")


if __name__ == "__main__":
    main()
