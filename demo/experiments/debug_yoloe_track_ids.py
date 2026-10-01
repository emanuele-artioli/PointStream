"""Print YOLOE track ids for clip_03. Diagnostic only."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from demo.pipeline.maps.yoloe_masks import (
    YOLOE_IMGSZ,
    YOLOE_TRACK_CONF,
    YOLOE_TRACKER,
    HandTrackFilter,
    _as_numpy,
    load_yoloe_model,
)

CLIP = Path("/home/itec/emanuele/tmp/maps-clips/clip_03.mp4")


def main() -> None:
    model, _weights, _extra = load_yoloe_model(["hand"])
    filt = HandTrackFilter()
    names = {0: "hand"}
    for index, result in enumerate(model.track(
        source=str(CLIP),
        stream=True,
        persist=True,
        retina_masks=True,
        verbose=False,
        device="cuda:0",
        half=True,
        imgsz=YOLOE_IMGSZ,
        conf=YOLOE_TRACK_CONF,
        tracker=str(YOLOE_TRACKER),
    )):
        boxes = getattr(result, "boxes", None)
        n = 0 if boxes is None else len(boxes)
        is_track = bool(getattr(boxes, "is_track", False)) if boxes is not None else False
        ids = _as_numpy(getattr(boxes, "id", None)) if boxes is not None else None
        conf = _as_numpy(getattr(boxes, "conf", None)) if boxes is not None else None
        masks = getattr(result, "masks", None)
        n_masks = 0 if masks is None or masks.data is None else int(len(masks.data))
        painted = filt.class_masks(result, 1080, 1920, names)["hand"]
        id_list = [] if ids is None else [int(v) for v in np.asarray(ids).reshape(-1)[:6]]
        conf_list = [] if conf is None else [round(float(v), 3) for v in np.asarray(conf).reshape(-1)[:6]]
        if index < 15 or index % 30 == 0 or int(painted.sum()) > 0:
            print(
                f"f={index} boxes={n} masks={n_masks} track={is_track} ids={id_list} conf={conf_list} paint={int(painted.sum())}",
                flush=True,
            )
        if index >= 299:
            break


if __name__ == "__main__":
    main()
