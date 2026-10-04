"""Time the five SAM-crop pose heads on one GPU.

Exits before timing if the ONNX session did not actually take the CUDA provider.
Eight crops, three warmup calls, each model timed alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import onnxruntime as ort

from demo.experiments.compare_pose_on_sam_crops import HAND_ONNX
from demo.experiments.pose_crop_gallery import load_models, read_crop

REPORT = Path("/home/itec/emanuele/pointstream-data/jobs/sam-crop-pose/report.json")


def cuda_ready() -> list[str]:
    session = ort.InferenceSession(str(HAND_ONNX), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    providers = session.get_providers()
    print("providers", providers, flush=True)
    if "CUDAExecutionProvider" not in providers:
        raise SystemExit("CUDA execution provider did not load")
    return providers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crops", type=int, default=8)
    args = parser.parse_args()
    providers = cuda_ready()
    report = json.loads(REPORT.read_text())
    samples = []
    for row in report["crops"]:
        if row["look"] or row["area"] < 8000:
            continue
        image, mask = read_crop(row)
        if image is None:
            continue
        samples.append((image, row["box"]))
        if len(samples) >= args.crops:
            break
    models = load_models()
    timings = {}
    for name, model in models.items():
        for image, box in samples[:3]:
            model(image, [box])
        started = time.perf_counter()
        for image, box in samples:
            model(image, [box])
        elapsed = time.perf_counter() - started
        timings[name] = {"ms_per_crop": 1000.0 * elapsed / len(samples), "crops": len(samples)}
        print("time", name, round(timings[name]["ms_per_crop"], 2), flush=True)
    order = sorted(timings, key=lambda name: timings[name]["ms_per_crop"])
    payload = {
        "schema": "pointstream.sam_crop_pose_gpu_timing.v1",
        "providers": providers,
        "crops": len(samples),
        "warmup": 3,
        "order_fastest_first": order,
        "timing_ms_per_crop": timings,
        "cpu_ms_per_crop": {
            "rtmpose-m": 31.7,
            "dwpose-l": 62.1,
            "rtmpose-l": 62.8,
            "rtmw-l": 94.8,
            "vitpose-l": 224.2,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
