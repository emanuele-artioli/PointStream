"""Source-only London person detection and seeded tracking diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO, __version__

SOURCE_SHA = "8ab59163aa86f3cb0f26b1981e6e546f148cca916e40731c0cdee5c6621e4b78"
WEIGHTS_SHA = "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(first, second):
    a, b = np.asarray(first), np.asarray(second)
    center_a, center_b = (a[:2] + a[2:]) / 2, (b[:2] + b[2:]) / 2
    return float(np.linalg.norm(center_a - center_b) / np.linalg.norm(a[2:] - a[:2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=[24, 106], required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    assert sha(a.source) == SOURCE_SHA and sha(a.weights) == WEIGHTS_SHA
    review = json.loads(a.proposals.read_text().split("=", 1)[1].strip().removesuffix(";"))
    reviewed = {int(k.split(":")[1]): v for k, v in review.items() if k.startswith("v006_p0014:")}
    assert len(reviewed) == 8 and len(reviewed[6]) == 2
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    model = YOLO(str(a.weights))
    capture = cv2.VideoCapture(str(a.source))
    assert abs(capture.get(cv2.CAP_PROP_FPS) - 25) < 0.001
    rows, started = [], time.monotonic()
    for tick in range(a.frames):
        ok, frame = capture.read()
        assert ok and frame.shape == (720, 1280, 3)
        result, = model.predict(frame, device="cpu", classes=[0], imgsz=960, conf=0.18, verbose=False)
        boxes = result.boxes.xyxy.cpu().numpy()
        assert np.isfinite(boxes).all()
        rows.append({"tick": tick, "boxes": boxes.tolist()})
    capture.release()
    elapsed = time.monotonic() - started
    forecast = elapsed * 106 / a.frames * 1.5 + 30
    if a.frames == 24:
        assert forecast < 600
    tracks = {6: [p["box"] for p in reviewed[6]]}
    for indices in (range(7, a.frames), range(5, -1, -1)):
        prior = tracks[6]
        for tick in indices:
            candidates = rows[tick]["boxes"]
            selected = []
            for old in prior:
                valid = sorted((distance(old, box), i, box) for i, box in enumerate(candidates)
                               if i not in selected)
                if not valid or valid[0][0] > 1:
                    selected = []
                    break
                selected.append(valid[0][1])
            if len(selected) != 2:
                tracks[tick] = None
                break
            prior = [candidates[index] for index in selected]
            tracks[tick] = prior
    comparisons = []
    for tick, expected in reviewed.items():
        if tick >= a.frames:
            continue
        actual = tracks.get(tick)
        target = [p["box"] for p in expected]
        if actual is None:
            comparisons.append({"tick": tick, "matched": False, "reason": "missing track"})
            continue
        direct = [distance(target[i], actual[i]) for i in range(2)]
        flipped = [distance(target[i], actual[1-i]) for i in range(2)]
        comparisons.append({"tick": tick, "matched": max(direct) <= .5,
                            "distances": direct, "flipped_distances": flipped})
    a.out.write_text(json.dumps({"scope": "exploratory source-only boxes; reviewed seed, no pose or reconstruction",
        "source_sha256": SOURCE_SHA, "weights_sha256": WEIGHTS_SHA, "review_sha256": sha(a.proposals),
        "script_sha256": sha(Path(__file__)), "torch": torch.__version__, "ultralytics": __version__,
        "seconds": elapsed, "forecast_106_seconds": forecast, "frames": a.frames,
        "linked_frames": sum(x is not None for x in tracks.values()), "review_comparisons": comparisons,
        "detections": rows}, indent=2, allow_nan=False) + "\n")
    print(f"frames={a.frames} seconds={elapsed:.1f} forecast={forecast:.1f} linked={sum(x is not None for x in tracks.values())} reviewed_pass={sum(x['matched'] for x in comparisons)}/{len(comparisons)}")


if __name__ == "__main__":
    main()
