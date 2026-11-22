"""Replay an archived DCVC packet and use the exact reviewed-frame metric worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import zipfile

import cv2

from experiments.tennis import starved_dcvc_regions as dcvc
from experiments.tennis import starved_joint_metrics as metric
from src.pointstream.tracked_patch import yuv_to_rgb

REFERENCE = Path("docs/experiments/starved-joint-metric-20261122/result.json")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args):
    import torch

    started = time.monotonic()
    if torch.cuda.device_count() != 1 or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Expected one CUDA GPU and one CPU")
    if any(sha(getattr(args, k)) != v for k, v in dcvc.WEIGHTS.items() if k != "vgg"):
        raise ValueError("DCVC model identity mismatch")
    if sha(args.old) != dcvc.ARCHIVE or sha(args.frames) != dcvc.FRAMES:
        raise ValueError("Source archive identity mismatch")
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
            raise ValueError("Both historical decode comparisons failed")
    if proof["historical_frame_hashes_matched"] != 240:
        raise ValueError("Historical 240-frame equality was not established")
    prepare = root / "source-inputs"
    prepare.mkdir()
    shutil.copyfile(args.supports / "supports.npz", prepare / "supports.npz")
    input_map = json.loads((args.supports / "key.json").read_text())
    index = {(arm, tick): digest for digest, (arm, tick) in input_map.items()}
    ticks = sorted(dcvc.SOURCES)
    frames_zip = prepare / "frames.zip"
    updated = {}
    with zipfile.ZipFile(args.frames) as existing, zipfile.ZipFile(frames_zip, "x", compression=zipfile.ZIP_STORED) as output:
        for tick in ticks:
            for arm in ("source", "plain", "roi"):
                digest = index[arm, tick]
                if arm == "source" and digest != dcvc.SOURCES[tick]:
                    raise ValueError("Original source frame identity differs")
                content = existing.read(digest + ".png")
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError("Source/control frame hash mismatch")
                output.writestr(digest + ".png", content)
                updated[digest] = [arm, tick]
            canvas = cv2.resize(yuv_to_rgb(native[tick], 960, 544, False),
                                (1920, 1080), interpolation=cv2.INTER_CUBIC)
            ok, png = cv2.imencode(".png", canvas[:, :, ::-1])
            if not ok:
                raise ValueError("Failed to encode lossless decoded frame")
            content = png.tobytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest in updated:
                raise ValueError("Decoded frame ID collision")
            output.writestr(digest + ".png", content)
            updated[digest] = ["pointstream", tick]
    key_path = prepare / "key.json"
    key_path.write_text(json.dumps(updated))
    metric.SOURCE_SHA = sha(frames_zip)
    metric.KEY_SHA = sha(key_path)
    metric.run(argparse.Namespace(frames=frames_zip, supports=prepare, alex=args.alex,
                                  vgg=args.vgg, count=args.count))
    stage = Path(os.environ["PS_STAGE_DIR"])
    report = json.loads((stage / "scores.json").read_text())
    if sha(REFERENCE) != "5367187da394bdac36f202ac9e2b1e65137ec54ca6dd3f9d3bf7035c38575b7e":
        raise ValueError("Frozen AV1/ROI reference score changed")
    expected = json.loads(REFERENCE.read_text())
    expected_rows = {(r["arm"], r["tick"]): r for r in expected["rows"]}
    max_delta = 0.0
    for row in report["rows"]:
        if row["arm"] not in ("plain", "roi"):
            continue
        prior = expected_rows[row["arm"], row["tick"]]
        if row["frame_sha256"] != prior["frame_sha256"] or row["source_sha256"] != prior["source_sha256"]:
            raise ValueError("Control frame differs")
        for kind in ("dists", "lpips"):
            for key, value in row[kind].items():
                max_delta = max(max_delta, abs(value - prior[kind][key]))
    if max_delta > 1e-4:
        raise ValueError(f"Control scoring differs by {max_delta}")
    report["dcvc"] = proof
    report["dcvc_attempts"] = attempts
    report["control_max_absolute_delta"] = max_delta
    report["original_frames_sha256"] = dcvc.FRAMES
    report["reference_sha256"] = sha(REFERENCE)
    report["wrapper_elapsed_seconds"] = time.monotonic() - started
    data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (stage / "scores.json").write_text(data)
    (root / "publish" / "scores.json").write_text(data)


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    report = json.loads((stage / "scores.json").read_text())
    count = 1 if os.environ["PS_STAGE"] == "smoke" else 12
    checks = {"count": report["count"] == count, "rows": len(report["rows"]) == 3 * count,
              "dcvc": report["dcvc"]["qp"] == 24 and report["dcvc"]["historical_frame_hashes_matched"] == 240,
              "packet": report["dcvc"]["packet_sha256"] == dcvc.PACKETS[24],
              "controls": report["control_max_absolute_delta"] <= 1e-4,
              "gpu": report["capability"] == [8, 9],
              "source": report["original_frames_sha256"] == dcvc.FRAMES,
              "self": all(abs(v) <= 1e-5 for v in report["self_check"].values())}
    if count == 1:
        fixed = report["wrapper_elapsed_seconds"] - report["score_seconds"]
        checks["forecast"] = fixed + 1.5 * report["score_seconds"] * 12 + 30 < 250
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps({"passed": all(checks.values()), "checks": checks}) + "\n")
    if not all(checks.values()):
        raise ValueError("DCVC scoring validation failed: " + str(checks))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("frames", "supports", "alex", "vgg", "old", "image", "video"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--count", type=int)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
