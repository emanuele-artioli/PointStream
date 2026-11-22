"""Authenticate the earlier JPEG renders before comparing their downstream scores."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np
import torch
from experiments.tennis import common_player_packet as packet, k6b
from experiments.tennis.player_requalification import array_sha
from src.codecs import quality
import score


def member_sha(data):
    return hashlib.sha256(data).hexdigest()


def main(a):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    start = time.process_time()
    people, frames, human, masks, boxes, renders, controls, raw_sha, _ = score.read_inputs(a)
    assert max(abs(v["observed"] - v["previous"]) for v in controls.values()) < 1e-5
    with tarfile.open(a.scratch / "lossless.tar") as archive:
        common = json.load(archive.extractfile("publish/receipt.json"))
    approved = json.loads(
        (score.HERE.parent / "player-reference-pilot-20261117/prepared-manifest.json").read_text()
    )
    info = {}
    for q in [40, 70]:
        mode = f"jpeg{q}"
        expected = approved[str(q)]
        corpus = a.scratch / (mode + ".tar")
        audit = json.loads(
            (score.HERE.parent / f"appearance-resume-20261118/q{q}-audit.json").read_text()
        )
        original = {u["unit"]: u for u in expected["units"]}
        info[mode] = {
            "producer_job": audit["job_id"],
            "producer_revision": audit["source_revision"],
            "archive_sha256": score.digest(corpus),
            "units": {},
        }
        with tarfile.open(corpus) as z:
            data = z.extractfile("publish/receipt.json").read()
            assert member_sha(data) == audit["checks"]["full_receipt_sha256"]
            receipt = json.loads(data)
            assert all(
                receipt[k] == common[k]
                for k in ("profile", "fit_sha256", "base_sha256", "model_record", "base_role")
            )
            assert (
                receipt["plan"]["settings"] == common["plan"]["settings"]
                and receipt["plan"]["source"] == common["plan"]["source"]
            )
            for person in people:
                unit = "nadal_kyrgios__000_at120__" + person.replace(":", "_")
                out = next(r for r in receipt["outputs"] if r["unit"] == unit)
                track = next(r for r in receipt["plan"]["units"] if r["unit"] == unit)
                base = next(r for r in common["plan"]["units"] if r["unit"] == unit)
                assert all(
                    track[k] == base[k] for k in ("t", "pose_sha256", "box_sha256", "seed", "run")
                )
                encoded = a.jpeg_inputs / f"q{q}" / (unit + ".ref1")
                meta, image = packet._bounded_transport(
                    encoded.read_bytes(), b"REF1", packet.MAX_REF
                )
                assert (
                    score.digest(encoded) == original[unit]["encoded_ref1_sha256"]
                    and len(encoded.read_bytes()) == original[unit]["encoded_ref1_bytes"]
                )
                assert meta["codec"] == ".jpg" and meta["quality"] == q
                stored = a.jpeg_inputs / f"q{q}/publish" / (unit + ".npz")
                assert (
                    score.digest(stored) == track["npz_sha256"] == original[unit]["new_npz_sha256"]
                )
                with np.load(stored, allow_pickle=False) as prep:
                    assert (
                        array_sha(prep["reference"])
                        == out["reference_sha256"]
                        == original[unit]["decoded_rgb_sha256"]
                    )
                    assert original[unit]["source_rgb_sha256"] == base["reference_sha256"]
                blob = z.extractfile("publish/" + unit + "@tennis_only.npz").read()
                assert member_sha(blob) == out["output_sha256"]
                with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
                    assert np.array_equal(npz["t"], np.arange(24)) and np.array_equal(
                        npz["box"], boxes[person]
                    )
                    crops = npz["crops"]
                    assert crops.shape == (24, 784, 512, 3)
                    renders[(mode, person)] = crops
                    sse = sum(
                        int(
                            np.square(
                                frame.astype(np.int32)
                                - k6b.crop_image(frames[t], boxes[person][t]).astype(np.int32)
                            ).sum(dtype=np.int64)
                        )
                        for t, frame in enumerate(crops)
                    )
                info[mode]["units"][person] = dict(
                    unit=unit,
                    seed=track["seed"],
                    pose_sha256=track["pose_sha256"],
                    image_bytes=len(image),
                    image_sha256=member_sha(image),
                    wrapped_ref1_bytes=encoded.stat().st_size,
                    output_sha256=out["output_sha256"],
                    decoded_reference_sha256=out["reference_sha256"],
                    full_crop_rgb_psnr=10 * math.log10(255**2 * crops.size / sse),
                )
        info[mode]["total_image_bytes"] = sum(
            x["image_bytes"] for x in info[mode]["units"].values()
        )
        info[mode]["total_ref1_bytes"] = sum(
            x["wrapped_ref1_bytes"] for x in info[mode]["units"].values()
        )
    prior = json.loads((score.HERE / "body-result.json").read_text())
    lookup = {(r["tick"], r["person"], r["support"]): r for r in prior["rows"]}
    net = quality.load_dists(a.weights, "cpu")
    rows = []
    with torch.inference_mode():
        for t, frame in enumerate(frames):
            for p in people:
                b = boxes[p][t]
                _, _, w, h = k6b.int_box(b)
                outputs = {
                    name: cv2.resize(renders[name, p][t], (w, h), interpolation=cv2.INTER_LANCZOS4)
                    for name in ("jpeg40", "jpeg70", "lossless")
                }
                for support, maskdict in [("source_model", masks), ("human", human)]:
                    if (t, p) not in maskdict:
                        continue
                    values = score.compare(net, frame, outputs, maskdict[t, p], b)
                    check = lookup[t, p, support]["scores"]["lossless"]
                    other = values["scores"]["lossless"]
                    assert (
                        abs(check["dists"] - other["dists"]) < 3e-6
                        and check["sum_squared_rgb"] == other["sum_squared_rgb"]
                    )
                    rows.append(dict(tick=t, person=p, support=support, **values))
            assert time.process_time() - start < 900
    summary = {}
    for support in ("source_model", "human"):
        summary[support] = {}
        for mode, meta in info.items():
            persons = {}
            for p in people:
                vals = [
                    r["scores"][mode] for r in rows if r["support"] == support and r["person"] == p
                ]
                persons[p] = dict(
                    dists=float(np.mean([x["dists"] for x in vals])),
                    rgb_psnr=10
                    * math.log10(
                        255**2
                        * sum(x["samples"] for x in vals)
                        / sum(x["sum_squared_rgb"] for x in vals)
                    ),
                    ticks=len(vals),
                )
            summary[support][mode] = dict(
                per_player=persons,
                dists=float(np.mean([x["dists"] for x in persons.values()])),
                rgb_psnr=float(np.mean([x["rgb_psnr"] for x in persons.values()])),
            )
    with a.output.open("x") as f:
        json.dump(
            dict(
                source_yuv_sha256=raw_sha,
                source_controls_exact=True,
                init=info,
                summary=summary,
                rows=rows,
                cpu_seconds=time.process_time() - start,
                script_sha256=score.digest(Path(__file__)),
            ),
            f,
            indent=2,
            allow_nan=False,
        )
        f.write("\n")
    print(
        {k: (v["total_image_bytes"], summary["source_model"][k]["dists"]) for k, v in info.items()}
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("scratch", "weights", "labels", "jpeg-inputs", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--yuv", type=Path)
    main(p.parse_args())
