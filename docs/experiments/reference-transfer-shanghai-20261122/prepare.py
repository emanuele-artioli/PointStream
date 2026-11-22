"""Restore the exact shipped AVIF RGB and rebind the two original learned-conditioning records."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from experiments.tennis import common_player_packet as packet, common_receiver_manifest as manifest
from experiments.tennis import w8_payload as wire

HERE = Path(__file__).resolve().parent
CARD = HERE.parent / "test-crop-joint-20261122"
sys.path.insert(0, str(HERE.parent / "reference-transfer-al028-20261122"))
from prepare import refblob  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main(a):
    a.out.mkdir(exist_ok=False, parents=True)
    old = a.paid / "players.pscp"
    old_wire = old.read_bytes()
    cam = packet._embedded_profile(old_wire)
    sender = json.loads((CARD / "match102-sender.json").read_text())
    actual = next(r for r in sender["rates"] if r["crf"] == 50)
    assert sha(old_wire) == actual["carrier_sha256"] and cam["recipe"] == packet.RECIPE_2997
    bank = json.loads((CARD / "match102-reference-cost.json").read_text())
    assert bank["source"] == sender["source_sha256"]
    levels = {r["setting"]: r for r in bank["rows"]}
    q = levels["avif90"]
    lossless = levels["webp"]
    png = levels["png"]
    init = (a.images / "avif90.psr").read_bytes()
    assert sha(init) == q["init_sha256"] and len(init) == q["init_bytes"]
    cp = packet._embedded_profile(init)
    enc, _ = wire.unpack(init, cp)
    s, _, runs = packet.unpack(old_wire)
    unchanged = {}
    results = {}
    for key, r in q["references"].items():
        unit = r["unit"]
        rr = cp["reference_metadata"][key]
        assert (
            unit == rr["unit"]
            and sha(enc[key]) == r["image_sha256"]
            and enc[key] == (a.images / f"avif90-{key}.avif").read_bytes()
        )
        rgb = cv2.imread(str(a.images / f"avif90-{key}.png"), cv2.IMREAD_COLOR)[:, :, ::-1]
        assert sha(rgb.tobytes()) == r["rgb_sha256"] and rgb.shape == (784, 512, 3)
        assert sha(runs[unit][0]["reference"].tobytes()) == r["source_rgb_sha256"]
        ok, n = cv2.imencode(".png", rgb[:, :, ::-1])
        assert ok
        unchanged[key] = refblob(rr, n.tobytes())
        results[unit] = dict(
            seed=runs[unit][0]["seed"],
            image_sha=r["image_sha256"],
            decoded_rgb_sha=r["rgb_sha256"],
            source_rgb_sha=r["source_rgb_sha256"],
        )
    new = packet.build(cam, s["base"], s["keypoints"], unchanged)
    n, _, new_runs = packet.unpack(new)
    assert (
        n["base"] == s["base"]
        and n["keypoints"] == s["keypoints"]
        and packet._embedded_profile(new) == cam
    )
    for unit in results:
        x = runs[unit][0]
        y = new_runs[unit][0]
        assert (
            x["seed"] == y["seed"]
            and all(np.array_equal(x[k], y[k]) for k in ("t", "kp", "boxes", "reference_box"))
            and all(
                np.array_equal(packet.pose_for_tick(x, t), packet.pose_for_tick(y, t))
                for t in range(240)
            )
        )
        assert sha(y["reference"].tobytes()) == results[unit]["decoded_rgb_sha"]
    (a.out / "players.pscp").write_bytes(new)
    mf = manifest.encode(sha(new))
    (a.out / "players.manifest").write_bytes(mf)
    manifest.verify(a.out / "players.manifest", sha(new))

    def img(x):
        return sum(v["image_bytes"] for v in x["references"].values())

    saving = 1 - img(q) / img(lossless)
    assert saving >= 0.30
    report = dict(
        source=bank["source"],
        source_control_carrier=sha(old_wire),
        avif_local_carrier=sha(new),
        avif_local_manifest=sha(mf),
        source_grid=240,
        test_ticks=list(range(24)),
        native_fps="30000/1001",
        source_webp_bytes=img(lossless),
        reference_q90_bytes=img(q),
        image_saving=saving,
        complete_initialization_bytes={
            "webp": lossless["init_bytes"],
            "q90": q["init_bytes"],
            "original_png": png["init_bytes"],
        },
        q90_init_sha=q["init_sha256"],
        source_only_oracle=True,
        source_conditioning_hash=sha(n["keypoints"]),
        resources=cam["resources"],
        units=results,
    )
    (a.out / "oracle.json").write_text(json.dumps(report, indent=2) + "\n")
    print(saving, report["source_webp_bytes"], report["reference_q90_bytes"], sha(new))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("paid", "images", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    main(p.parse_args())
