"""Produce predeclared automatic (unreviewed) ball centers on the player window."""
import argparse
import hashlib
import json
import math
import platform
import resource
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.tennis import k5b_trackers

SOURCE = "b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d"
CODE = "5bec7ca535d2562ce0252a45f3dded0cfd5897571a6e29a10247a87884d961ff"
WEIGHTS = "b052479e6229d840a1a7f42dbc83a6b752a53ef8bc5841a4b957acd519563fec"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "archive", "weights", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--frames", choices=(8, 240), type=int, required=True)
    args = p.parse_args()
    started = time.monotonic()
    cpu = time.process_time()
    if [sha(path) for path in (args.source, args.archive, args.weights)] != [SOURCE, CODE, WEIGHTS]:
        raise ValueError("Authentication failed")
    args.output.mkdir(exist_ok=False, parents=True)
    with tarfile.open(args.archive) as tar:
        tar.extractall(args.output / "code", filter="data")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    tracker = k5b_trackers.BallTrack(args.output / "code", args.weights, device="cpu")
    if next(tracker.net.parameters()).device.type != "cpu" or tracker.net.training:
        raise ValueError("CPU evaluation mode required")
    cap = cv2.VideoCapture(str(args.source), cv2.CAP_FFMPEG)
    frames = np.empty((args.frames, 720, 1280, 3), np.uint8)
    try:
        if [cap.get(k) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                                  cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT)] != [1920, 1080, 50, 240]:
            raise ValueError("Unexpected native input")
        for t in range(args.frames):
            ok, image = cap.read()
            if not ok:
                raise ValueError("Source frame missing")
            frames[t] = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (1280, 720), interpolation=cv2.INTER_AREA)
    finally:
        cap.release()
    # A nonfinite heatmap must fail the trial even if thresholding it yields no detections.
    def valid_heatmap(module, inputs, outputs):
        if not torch.isfinite(outputs[0]).all():
            raise ValueError("Nonfinite BallTrack heatmap")
    tracker.net.register_forward_hook(valid_heatmap)
    predictions = tracker.run(frames)
    if args.frames == 8 and tracker.run(frames) != predictions:
        raise ValueError("Nonreproducible smoke")
    if len(predictions) != args.frames:
        raise ValueError("Incomplete track")
    centers = []
    for visible, x, y in predictions:
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < 1280 and 0 <= y < 720):
            raise ValueError("Invalid source center")
        centers.append([min(1919, round(1.5*x)), min(1079, round(1.5*y))] if visible else None)
    record = dict(window="nadal_kyrgios/000@120", source_sha256=SOURCE, canvas_size=[1920,1080], fps=[50,1],
                  centers=centers, prediction_scope="unreviewed automatic sender positions",
                  code_archive_sha256=CODE, weights_sha256=WEIGHTS, torch=str(torch.__version__),
                  device="cpu", cv2=cv2.__version__, numpy=np.__version__, checkpoint=tracker.meta,
                  runner_sha256=sha(Path(__file__)), tracker_module_sha256=sha(Path(k5b_trackers.__file__)),
                  rgb_input_sha256=hashlib.sha256(frames).hexdigest(), platform=platform.platform(),
                  smoke_repeat_checked=args.frames == 8, cpu_seconds=time.process_time()-cpu,
                  elapsed_seconds=time.monotonic()-started, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (args.output/"track.json").write_text(json.dumps(record, indent=2)+"\n")
    print(f"{len(centers)} timestamps, {sum(x is not None for x in centers)} placements; "
          f"{record['elapsed_seconds']:.1f}s wall, {record['cpu_seconds']:.1f}s CPU")


if __name__ == "__main__":
    main()
