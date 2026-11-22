"""Measure receiver-readable person localization against the same independent body labels."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
from statistics import mean
import tarfile

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.tennis import common_player_packet, k6b, w8_payload

HERE = Path(__file__).resolve().parent
NADAL_POSE = "26b35756f5b038dbf509115d38eda8606913726109aaba3dcf20b6e79372e6d0"
ALCARAZ_POSE = "ab12294fec2bba5bb5b21bf373a3a31c620b80cf9463cd92ab0383647d1a8aa9"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def mask_box(mask):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return [x, y, x + w, y + h]


def overlap(a, b):
    common = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return common / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - common)


def score(a):
    wire = a.packet.read_bytes()
    assert sha(wire) == (NADAL_POSE if a.event == "nadal" else ALCARAZ_POSE)
    profile = common_player_packet._embedded_profile(wire)
    sections, _ = w8_payload.unpack(wire, profile)
    _, tracks = common_player_packet._bounded_keypoints(
        sections["poses"], profile["player"]["order"]
    )
    assert len(tracks) == 2 and all(np.array_equal(t["t"], np.arange(240)) for t in tracks)
    label_bytes = a.labels.read_bytes()
    if a.event == "nadal":
        receipt = json.loads(
            (HERE.parent / "joint-player-machine-20261122/result.json").read_text()
        )
        assert (
            sha(label_bytes) == receipt["label_sha256"]
            and sha(a.predictions.read_bytes()) == receipt["predictions_sha256"]
        )
        assert sha(a.key.read_bytes()) == receipt["key_sha256"]
        binding = json.loads(a.key.read_text())
        finish = json.loads((a.masks / "finish.json").read_text())
        assert (
            sha((a.masks / "finish.json").read_bytes())
            == "645d719da006b9df133849753ed45b2da00a821958b716c1b4debf5e625ce131"
        )
        projected = {}
        for t in tracks:
            name = t["name"]
            f = a.masks / (name + "-masks.npz")
            assert sha(f.read_bytes()) == finish["mask_files"][name]
            with np.load(f, allow_pickle=False) as z:
                assert np.array_equal(z["t"], t["t"])
                projected[name] = (z["masks"], k6b.run_boxes(list(t["kp"])))
    else:
        contract = json.loads(a.key.read_text())
        assert sha(label_bytes) == contract["normalized_labels_sha256"]
        binding = contract["keys"]
    with tarfile.open(a.predictions) as archive:
        pred = json.load(archive.extractfile("publish/predictions.json"))
    assert set(binding) == set(pred["rows"]) and len(binding) == 48
    lookup = {(arm, tick): identity for identity, (arm, tick) in binding.items()}
    rows = []
    for labeled in json.loads(label_bytes)["frames"]:
        tick = labeled["tick"]
        truth = [
            mask_box(
                cv2.imdecode(
                    np.frombuffer(
                        base64.b64decode(p["mask"]["png_base64"], validate=True), np.uint8
                    ),
                    0,
                )
                > 0
            )
            for p in labeled["players"]
        ]
        boxes = {"pose": [], "silhouette": []}
        for idx, t in enumerate(tracks):
            kp = t["kp"][tick]
            xy = kp[kp[:, 2] >= 0.5, :2]
            if len(xy) >= 2 and np.all(np.ptp(xy, axis=0) > 0):
                x, y = xy.min(axis=0)
                u, v = xy.max(axis=0)
                boxes["pose"].append(
                    [max(float(x), 0), max(float(y), 0), min(float(u), 1920), min(float(v), 1080)]
                )
            if a.event == "nadal":
                masks, geom = projected[t["name"]]
                x, y, w, h = (round(float(v)) for v in geom[tick])
                keep = cv2.resize(
                    masks[tick].astype(np.uint8),
                    (max(w, 1), max(h, 1)),
                    interpolation=cv2.INTER_NEAREST,
                )
                x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, 1920), min(y + h, 1080)
                piece = keep[y0 - y : y1 - y, x0 - x : x1 - x]
                assert int(piece.sum()) == finish["rows"][tick]["projected_pixels"][t["name"]]
                if piece.any():
                    bx, by, bu, bv = mask_box(piece)
                    boxes["silhouette"].append([x0 + bx, y0 + by, x0 + bu, y0 + bv])
            else:
                raw = (
                    a.masks / "private-review" / f"{tick:03d}/receiver-mask-{idx + 1}.png"
                ).read_bytes()
                assert sha(raw) == contract["receiver_masks"][str(tick)][idx]
                boxes["silhouette"].append(
                    mask_box(cv2.imdecode(np.frombuffer(raw, np.uint8), 0) > 0)
                )
        for (arm, k), identity in lookup.items():
            if k == tick:
                boxes[arm] = [
                    [x, y, x + w, y + h]
                    for p in pred["rows"][identity]
                    for x, y, w, h in [p["rect"]]
                ]
        for arm, group in boxes.items():
            cost = np.array([[overlap(ref, b) for b in group] for ref in truth])
            ii, jj = linear_sum_assignment(-cost)
            found = {i: float(cost[i, j]) for i, j in zip(ii, jj, strict=True)}
            rows.append(dict(tick=tick, arm=arm, iou=[found.get(i, 0.0) for i in (0, 1)]))
    summary = {}
    for arm in boxes:
        vals = [v for r in rows if r["arm"] == arm for v in r["iou"]]
        assert len(vals) == 24
        summary[arm] = dict(
            mean_iou=mean(vals), at_least_half=sum(v >= 0.5 for v in vals), n=len(vals)
        )
    return dict(
        event=a.event,
        wire_sha256=sha(wire),
        pose_bytes=len(sections["poses"]),
        source_labels_sha256=sha(label_bytes),
        predictions_sha256=sha(a.predictions.read_bytes()),
        summary=summary,
        rows=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("event", choices=["nadal", "al028"])
    for k in ("packet", "labels", "masks", "key", "predictions", "output"):
        p.add_argument(k, type=Path)
    args = p.parse_args()
    result = score(args)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))
