"""Recover three lossless stroke frames from the verified Q24 native decoder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

import cv2

from experiments.tennis import starved_dcvc_regions as dcvc
from src.pointstream.tracked_patch import yuv_to_rgb


def run(args):
    import torch

    if torch.cuda.device_count() != 1 or dcvc.sha(args.old) != dcvc.ARCHIVE:
        raise ValueError("DCVC source or CUDA device mismatch")
    if any(dcvc.sha(getattr(args, k)) != v for k, v in dcvc.WEIGHTS.items() if k != "vgg"):
        raise ValueError("DCVC model identity differs")
    root = Path(os.environ["PS_SCRATCH_DIR"])
    attempts = []
    with tarfile.open(args.old) as archive:
        for attempt in range(2):
            work = root / f"decode-{attempt}"
            work.mkdir()
            try:
                native, proof = dcvc.replay(args, archive, 24, work)
                attempts.append({"attempt": attempt, "passed": True})
                break
            except ArithmeticError as exc:
                attempts.append({"attempt": attempt, "passed": False, "error": str(exc)})
        else:
            raise ValueError("Historical decoder parity failed twice: " + json.dumps(attempts))
        prior = json.load(archive.extractfile("publish/nad000/dcvc-s2-q24-reports.json"))
    expected = prior["decode"]["decode-a"]["passes"][0]["frame_sha256"]
    publish = root / "publish"
    publish.mkdir()
    frames = {}
    for tick in (78, 86, 94):
        raw_hash = hashlib.sha256(native[tick].tobytes()).hexdigest()
        if raw_hash != expected[tick]:
            raise ValueError("Stroke source frame differs")
        rgb = yuv_to_rgb(native[tick], 960, 544, False)
        ok, png = cv2.imencode(".png", rgb[:, :, ::-1])
        if not ok:
            raise ValueError("Failed to serialize lossless frame")
        blob = png.tobytes()
        name = f"q24-stroke-{tick:03d}.png"
        (publish / name).write_bytes(blob)
        frames[str(tick)] = {"filename": name, "native_yuv_sha256": raw_hash,
                             "png_sha256": hashlib.sha256(blob).hexdigest(),
                             "size": [960, 544]}
    report = {"stage": os.environ["PS_STAGE"], "dcvc": proof, "attempts": attempts,
              "frames": frames, "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability(0))}
    content = json.dumps(report, indent=2) + "\n"
    (publish / "report.json").write_text(content)
    (Path(os.environ["PS_STAGE_DIR"]) / "report.json").write_text(content)


def validate():
    report = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "report.json").read_text())
    checks = {"packet": report["dcvc"]["packet_sha256"] == dcvc.PACKETS[24],
              "all_native_frames": report["dcvc"]["historical_frame_hashes_matched"] == 240,
              "gpu": report["capability"] == [8, 9],
              "three_stroke_frames": set(report["frames"]) == {"78", "86", "94"}}
    with tarfile.open(Path(os.environ["PS_STAGE_DIR"]) / "published.tar") as tar:
        for row in report["frames"].values():
            blob = tar.extractfile("publish/" + row["filename"]).read()
            checks[row["filename"]] = hashlib.sha256(blob).hexdigest() == row["png_sha256"]
    if os.environ["PS_STAGE"] == "smoke":
        checks["forecast"] = report["dcvc"]["decode_seconds"] * 1.5 + 20 < 105
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps({"passed": all(checks.values()), "checks": checks}) + "\n")
    if not all(checks.values()):
        raise ValueError("Q24 stroke validation failed: " + str(checks))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--video", type=Path)
    p.add_argument("--count", type=int)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
