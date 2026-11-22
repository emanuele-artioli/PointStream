"""Independent person-mask inference on blinded full-frame source and decoder outputs."""

import argparse
import base64
import hashlib
import json
import os
import tarfile
import time
import zipfile
from pathlib import Path

WEIGHT = "92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3"
FRAMES = "4c891777b33b10ba92fd94fa1a59ba360aacf175e7bfd67c382206058d59550a"
SMOKE = {
    "0360ccc4ec8eea097f872285551080d36de520b8ec84ab4f0ccf30b5e9284a7a",
    "ddcc3a5181f1c4737dcfe75dd50269c05d3fb41c9840d2fd198b811bb132cb3c",
    "9c7d70f367d0f814e2c6fc5aeffc7d903d256b186d194a7d6bd6270d0fad593e",
    "d21b2720b957c8dfbfc7954a20038a54af5affa04f021c66a6c63627b2117bfc",
}


# Both profiles use the first reviewed tick for the same source-image check.
SOURCE_SMOKE = "0360ccc4ec8eea097f872285551080d36de520b8ec84ab4f0ccf30b5e9284a7a"
PROFILES = {
    "initial": (FRAMES, SMOKE),
    "joint21": (
        "f6936efec14c18624bccf16dcb3c0189f386fbb5a769f92d4366fe9f7e57a2d0",
        {
            "0360ccc4ec8eea097f872285551080d36de520b8ec84ab4f0ccf30b5e9284a7a",
            "a0bfc3dd0f139b75e48e01f598fd7a146d4438d929f0fe212b69f3d850b17d8c",
            "0b98bc0ef17519f354e2f219b99ac971a76607e8f764ef0bc7e146a6edbc4cc9",
            "88beeebd4a276964760a4cb0fee4f57c5892d04465c57bd21c288b33e2dfa52c",
        },
    ),
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    frame_hash, smoke_ids = PROFILES[args.profile]
    start = time.monotonic()
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    os.environ["YOLO_CONFIG_DIR"] = str(scratch / "yolo-config")
    os.environ["YOLO_OFFLINE"] = "true"
    if len(os.sched_getaffinity(0)) != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU isolation check failed")
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    if sha(args.weight) != WEIGHT or sha(args.frames) != frame_hash:
        raise ValueError("Staged input changed")
    path = scratch / "yolo26x-seg.pt"
    if path.resolve() != args.weight.resolve():
        path.symlink_to(args.weight.resolve())
    model = YOLO(str(path), task="segment")
    assert model.names[0] == "person"
    rows = {}
    with zipfile.ZipFile(args.frames) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) == 48
        names = sorted(archive.namelist())
        if args.count == 4:
            names = [name for name in names if Path(name).stem in smoke_ids]
        started = time.monotonic()
        for name in names:
            data = archive.read(name)
            identity = hashlib.sha256(data).hexdigest()
            assert identity + ".png" == name
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            assert frame.shape == (1080, 1920, 3)
            (result,) = model.predict(
                frame,
                imgsz=1280,
                conf=0.25,
                iou=0.7,
                classes=[0],
                retina_masks=True,
                device="cpu",
                verbose=False,
            )
            persons = []
            if len(result.boxes):
                assert result.masks is not None and result.masks.data.shape == (
                    len(result.boxes),
                    1080,
                    1920,
                )
                for box, mask in zip(result.boxes, result.masks.data):
                    assert int(box.cls.item()) == 0 and mask.device.type == "cpu"
                    arr = (mask.numpy() > 0.5).astype(np.uint8)
                    x, y, w, h = cv2.boundingRect(arr)
                    if not w or not h:
                        continue
                    ok, png = cv2.imencode(".png", arr[y : y + h, x : x + w] * 255)
                    assert ok
                    persons.append(
                        dict(
                            rect=[x, y, w, h],
                            score=float(box.conf.item()),
                            pixels=int(arr.sum()),
                            png=base64.b64encode(png).decode(),
                        )
                    )
            rows[identity] = persons
    end = time.monotonic()
    report = dict(
        rows=rows,
        weight=WEIGHT,
        frames=frame_hash,
        ultralytics=ultralytics.__version__,
        torch=str(torch.__version__),
        setup_seconds=started - start,
        scoring_seconds=end - started,
    )
    publish = scratch / "publish"
    publish.mkdir()
    (publish / "predictions.json").write_text(json.dumps(report))


def validate(profile):
    frame_hash, smoke_ids = PROFILES[profile]
    import cv2
    import numpy as np

    stage = Path(os.environ["PS_STAGE_DIR"])
    with tarfile.open(stage / "published.tar") as archive:
        r = json.load(archive.extractfile("publish/predictions.json"))
    smoke = os.environ["PS_STAGE"] == "smoke"
    assert (
        r["weight"] == WEIGHT
        and r["frames"] == frame_hash
        and len(r["rows"]) == (4 if smoke else 48)
    )
    assert smoke_ids <= r["rows"].keys()
    for preds in r["rows"].values():
        for p in preds:
            x, y, w, h = p["rect"]
            im = cv2.imdecode(np.frombuffer(base64.b64decode(p["png"], validate=True), np.uint8), 0)
            assert 0 <= x < x + w <= 1920 and 0 <= y < y + h <= 1080 and 0.25 <= p["score"] <= 1
            assert im.shape == (h, w) and np.count_nonzero(im) == p["pixels"] > 0
            assert np.all((im == 0) | (im == 255))
    assert r["rows"][SOURCE_SMOKE]
    forecast = r["setup_seconds"] + 18 * r["scoring_seconds"] + 65 if smoke else None
    checks = dict(
        authenticated_inputs=r["weight"] == WEIGHT and r["frames"] == frame_hash,
        complete_records=len(r["rows"]) == (4 if smoke else 48),
        source_persons=len(r["rows"][SOURCE_SMOKE]) > 0,
        geometry_and_masks=True,
        time_budget=forecast is None or forecast < 1150,
    )
    passed = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=passed, checks=checks, smoke_forecast_s=forecast))
    )
    if not passed:
        raise ValueError("Smoke forecast exceeds the full scoring cap")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path)
    p.add_argument("--weight", type=Path)
    p.add_argument("--count", type=int, choices=(4, 48))
    p.add_argument("--validate", action="store_true")
    p.add_argument("--profile", choices=tuple(PROFILES), default="initial")
    args = p.parse_args()
    validate(args.profile) if args.validate else run(args)
