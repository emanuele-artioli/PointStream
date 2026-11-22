"""Refit a previously computed sender track to a new native AV1 base."""
import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path


from experiments.tennis import k5b1b, k5b_trackers as trackers, k5c_independent_ball as old
from experiments.tennis.windows import rgb_frames
from src.codecs import svtav1
from src.pointstream import ball, ball_stream

WIDTH, HEIGHT, TOTAL = 1280, 720, 292
DECODER = "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6"


def get(tar, name):
    file = tar.extractfile(name)
    if file is None:
        raise ValueError("Missing member " + name)
    return file.read()


def decode(stream, root, frames, decoder):
    ivf, yuv = root / "native.ivf", root / "native.yuv"
    ivf.write_bytes(stream.native)
    subprocess.run(svtav1.decode_command(decoder, ivf, yuv, threads=1),
                   check=True, capture_output=True, timeout=120)
    if yuv.stat().st_size != TOTAL * WIDTH * HEIGHT * 3 // 2:
        raise ValueError("Native decoder did not produce every frame")
    # Decode the full stream to preserve codec context, then select the prefix.
    all_frames = rgb_frames(yuv, WIDTH, HEIGHT, TOTAL, root / "rgb.npy")
    return all_frames[:frames]


def run(args):
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Refit requires one CPU and no GPU")
    tool = svtav1.tool("dav1d")
    if tool["sha256"] != DECODER:
        raise ValueError("Decoder differs from original")
    with tarfile.open(args.original) as source_archive:
        history = json.loads(get(source_archive, "publish/source-qualification.json"))
        previous = ball_stream.unpack(get(source_archive, "publish/svt-62/stream.psbl"))
    with tarfile.open(args.matched) as matched:
        receipt = json.loads(get(matched, "publish/report.json"))
        if receipt["frames"] != TOTAL or not receipt["within_half_percent"]:
            raise ValueError("The rate-matched pilot did not pass")
        index = next(i for i, row in enumerate(receipt["trials"])
                     if row["packet_sha256"] == receipt["best"]["packet_sha256"])
        packed = get(matched, f"publish/candidate-{index}.psbl")
    if hashlib.sha256(packed).hexdigest() != receipt["best"]["packet_sha256"]:
        raise ValueError("Candidate packet differs from accepted receipt")
    candidate = ball_stream.unpack(packed)
    if history["frames"] != TOTAL or len(history["rows"]) != TOTAL:
        raise ValueError("Original model-track record incomplete")
    # Sender sees the saved model predictions only. The labels in the archival
    # source-qualification file are not used for fitting.
    predictions = [(row["balltrack"]["detected"], row["balltrack"]["x"],
                    row["balltrack"]["y"]) for row in history["rows"][:args.frames]]
    start_track = k5b1b.track_of(predictions, (HEIGHT, WIDTH))
    clip = next(c for c in trackers.tracknet_clips(args.source, [9])
                if c["id"] == "tracknet/game9/Clip5")
    clip["rows"] = clip["rows"][:args.frames]
    source = trackers.tracknet_frames(args.source, clip)
    root = Path(os.environ["PS_SCRATCH_DIR"])
    decoded = []
    for name, stream in (("previous", previous), ("candidate", candidate)):
        directory = root / name
        directory.mkdir()
        decoded.append(decode(stream, directory, args.frames, tool["path"]))
    fit_started = time.monotonic()
    fits = old.fit_streak(start_track, source, decoded, (HEIGHT, WIDTH))
    fit_seconds = time.monotonic() - fit_started
    payloads = [ball.encode_frames(item["track"], item["appearances"], WIDTH, HEIGHT)
                for item in fits]
    if args.frames == TOTAL and payloads[0] != previous.ball_payload:
        raise ValueError("Refit from saved source-track predictions failed original payload reproduction")
    output = ball_stream.Stream(WIDTH, HEIGHT, candidate.fps, args.frames,
                                candidate.codec, candidate.resource_id, candidate.native, payloads[1])
    # A prefix test uses a full native stream and therefore only evaluates fitting.
    # Only serialize a full output because packet/native frame counts must agree.
    result = {"stage": os.environ["PS_STAGE"], "frames": args.frames,
              "fit_seconds": fit_seconds, "elapsed_seconds": time.monotonic() - started,
              "source_track_positions": sum(p is not None for p in start_track),
              "historic_payload_reproduced": args.frames == TOTAL and payloads[0] == previous.ball_payload,
              "fixed_payload_sha256": hashlib.sha256(candidate.ball_payload).hexdigest(),
              "refitted_payload_sha256": hashlib.sha256(payloads[1]).hexdigest()}
    publish = root / "publish"
    publish.mkdir()
    if args.frames == TOTAL:
        packet = ball_stream.pack(output)
        (publish / "refitted.psbl").write_bytes(packet)
        result["refitted_bytes"] = len(packet)
        result["fixed_bytes"] = len(packed)
        result["target_bytes"] = receipt["target_bytes"]
        result["refitted_sha256"] = hashlib.sha256(packet).hexdigest()
        result["within_half_percent"] = abs(len(packet) - receipt["target_bytes"]) / receipt["target_bytes"] <= .005
    data = json.dumps(result, indent=2) + "\n"
    (publish / "report.json").write_text(data)
    (Path(os.environ["PS_STAGE_DIR"]) / "refit-report.json").write_text(data)


def validate():
    report = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "refit-report.json").read_text())
    n = 24 if os.environ["PS_STAGE"] == "smoke" else TOTAL
    checks = {"frames": report["frames"] == n,
              "sender_has_detections": report["source_track_positions"] > 0}
    if n == 24:
        checks["forecast"] = report["elapsed_seconds"] + 1.5 * report["fit_seconds"] * TOTAL / 24 + 180 < 720
    else:
        checks["reproduces_original_payload"] = report["historic_payload_reproduced"]
        with tarfile.open(Path(os.environ["PS_STAGE_DIR"]) / "published.tar") as archive:
            packet = get(archive, "publish/refitted.psbl")
        checks["packet_checksum"] = hashlib.sha256(packet).hexdigest() == report["refitted_sha256"]
        checks["packet_decode"] = ball_stream.unpack(packet).frames == TOTAL
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps({"passed": all(checks.values()), "checks": checks}) + "\n")
    if not all(checks.values()):
        raise ValueError("Refit validation failed: " + str(checks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--matched", type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate() if args.validate else run(args)
