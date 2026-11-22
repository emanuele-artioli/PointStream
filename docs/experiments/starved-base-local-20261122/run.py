"""Quarter-scale source and cleaned rally base, using native codec and packet paths."""
import argparse
import hashlib
import json
import resource
import struct
import subprocess
import time
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
import numpy as np

from experiments.tennis import common_player_ball as ball
from experiments.tennis import common_player_packet as player
from experiments.tennis import common_receiver_manifest as manifests
from src.codecs import svtav1

ORIGINAL = Path("/Users/manu/Datasets/pointstream-submission-20261118/ps-native-ctl-20261118-full/source.yuv")
CLEAN = Path("/private/tmp/ps-clean-court-grabcut-20261119/clean.yuv")
DELIVERY = Path("/Users/manu/.codex/visualizations/2026/11/18/01a1edc1-dd4e-7ab3-b819-fbd88da8db05/clean-court-deliveries-20261119/grabcut")
EXPECTED = {
    "original": "6cae4bd7a46bfa4bd2a7a7ec5c926647ea43d9e7a5366eb96e464eaffe1e603c",
    "clean": "4f27937bff116d51aab914a1c2e7262b51ef14483a2e2455551c2b786a684dfc",
}
CARRIER_SHA = "25242f4db9c081a75472762db8b409c827c7f84626d073e7838d40bef724910c"
SIDE_SHA = "bdf8280e62e2285898868e9a04101abe9c5af891e66a9364220f4403a207de40"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu():
    return sum(getattr(resource.getrusage(kind), "ru_utime") + getattr(resource.getrusage(kind), "ru_stime")
               for kind in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, choices=(24, 240), required=True)
    parser.add_argument("--preset", type=int, choices=(4,), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start, cpu_start = time.monotonic(), cpu()
    assert sha(ORIGINAL) == EXPECTED["original"] and sha(CLEAN) == EXPECTED["clean"]
    ff = svtav1.tool("ffmpeg")
    encoder = svtav1.tool("SvtAv1EncApp")
    decoder = svtav1.tool("dav1d")
    source_packet = (DELIVERY / "players.pscp").read_bytes()
    side = (DELIVERY / "ball.sidecar").read_bytes()
    assert hashlib.sha256(source_packet).hexdigest() == CARRIER_SHA
    assert hashlib.sha256(side).hexdigest() == SIDE_SHA
    source_parts, _, source_decoded = player.unpack(source_packet)
    positions, tile, canvas = ball._open(source_packet, side, decoder["sha256"])
    assert len(positions) == 240 and canvas == (1920, 1080)
    outputs = []
    for name, path in (("original", ORIGINAL), ("clean", CLEAN)):
        raw = args.output / (name + ".yuv")
        command = [ff["path"], "-v", "error", "-threads", "2", "-f", "rawvideo", "-pixel_format", "yuv420p",
                   "-video_size", "960x544", "-framerate", "50", "-i", str(path), "-vf",
                   "scale=240:136:flags=area,format=yuv420p", "-frames:v", str(args.frames),
                   "-f", "rawvideo", str(raw)]
        subprocess.run(command, check=True, timeout=180)
        assert raw.stat().st_size == args.frames * 240 * 136 * 3 // 2
        for crf in (36, 48, 60):
            folder = args.output / f"{name}-{crf}"
            folder.mkdir()
            result = svtav1.code(raw, folder, width=240, height=136, fps=Fraction(50),
                                 frames=args.frames, crf=crf, preset=args.preset, threads=2,
                                 full_range=False, encoder=encoder["path"], decoder=decoder["path"], timeout=180)
            assert result["decoded_frames"] == args.frames and result["temporal_units"] == args.frames
            data = Path(result["stream"]).read_bytes()
            assert struct.unpack_from("<HHIIII", data, 12) == (240, 136, 50, 1, args.frames, 0)
            assert len(data) == result["payload_bytes"] + 32 + 12 * args.frames
            result.update(arm=name, crf=crf, decoded_sha256=sha(Path(result["decoded"])),
                          stream_kbps=len(data) * 8 * 50 / args.frames / 1000)
            if args.frames == 240 and name == "clean":
                profile = player._embedded_profile(source_packet)
                profile["coded_size"] = [240, 136]
                references = {key: value for key, value in source_parts.items() if key.startswith("ref")}
                carrier = player.build(profile, data, source_parts["keypoints"], references)
                decoded_parts, _, decoded_cond = player.unpack(carrier)
                assert decoded_parts == {**source_parts, "base": data}
                for unit in source_decoded:
                    assert len(source_decoded[unit]) == len(decoded_cond[unit])
                    for src, dst in zip(source_decoded[unit], decoded_cond[unit]):
                        assert src["seed"] == dst["seed"]
                        for key in ("reference", "boxes", "kp", "t"):
                            assert np.array_equal(src[key], dst[key])
                rebound = ball.pack(carrier, positions, tile, decoder_sha256=decoder["sha256"])
                assert ball._open(carrier, rebound, decoder["sha256"]) == (positions, tile, canvas)
                companion = json.loads(manifests.encode(hashlib.sha256(carrier).hexdigest()))
                companion.update(profile="common-tennis-macos-cpu-v1", native_ffmpeg=ff["sha256"],
                                 numpy=np.__version__, opencv_build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest())
                manifest = json.dumps(companion, sort_keys=True, separators=(",", ":")).encode("ascii")
                files = {"players.pscp": carrier, "ball.sidecar": rebound, "receiver.manifest": manifest}
                for filename, blob in files.items():
                    (folder / filename).write_bytes(blob)
                total = sum(map(len, files.values()))
                reference_bytes = sum(map(len, references.values()))
                result.update(files={key: len(blob) for key, blob in files.items()}, reference_bytes=reference_bytes,
                              total_bytes=total, cold_kbps=total / 600,
                              optimistic_no_reference_kbps=(total-reference_bytes)/600)
            outputs.append(result)
    report = {"inputs": EXPECTED, "frames": args.frames, "tools": [ff, encoder, decoder],
              "outputs": outputs, "cpu_seconds": cpu()-cpu_start, "wall_seconds": time.monotonic()-start}
    if args.frames == 24:
        assert report["cpu_seconds"] * 10 * 1.5 + 120 < 900, "forecast exceeds budget"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "metrics.json").write_text(json.dumps({"valid": True, "metrics": {"frames": args.frames},
                                                          "cost": max(report["cpu_seconds"], .001)}) + "\n")
    print(json.dumps({"frames": args.frames, "cpu_seconds": report["cpu_seconds"],
                      "rows": [{k: row[k] for k in ("arm", "crf", "stream_kbps")} for row in outputs]}))


if __name__ == "__main__":
    main()
