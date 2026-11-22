"""One-window W7 player/base recomposition with original pixel controls."""

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path
import subprocess
import tarfile
import time

import cv2
import numpy as np

from experiments.tennis import k6, k6b, k6b_matte_judge as judge
from experiments.tennis.k4 import safe
from experiments.tennis.k4b_features import convert
from experiments.tennis.k6b_matte_cpu import box_maps
from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_masks
from experiments.tennis.k6b_rate_transport import array_sha, sha
from experiments.tennis.k6b_real_base_compose import compose
from experiments.tennis.k6b_real_base_judge import extract, unit_mask
from experiments.tennis.k6b_receiver_sam import project, transport_root
from experiments.visor.h3 import rgb_memmap
from experiments.visor.h3b import dists_crops, load_dists
from src.codecs import svtav1
from src.segmentation.masks import decompress

WID = "alcaraz_hurkacz/034@120"
SOURCE = "7db0c143cd416ad17b4bda4eaa191d51da197eccc1143e9eb402f8bb2a7f0ee6"
WINDOW_SOURCES = {
    "al028": "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934",
    "al034": SOURCE,
    "al043": "847cb8272134bd31a5447ff3cce02b28a906fae44559b3e0a58661d8171b6c16",
}


def tar_member(archive, member, output):
    with tarfile.open(archive) as tar:
        data = tar.extractfile(member)
        if data is None:
            raise ValueError("missing " + member)
        with output.open("wb") as f:
            while chunk := data.read(1024 * 1024):
                f.write(chunk)


def run(a):
    import torch

    began = time.monotonic()
    if not torch.cuda.is_available() or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("GPU or CPU isolation")
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, 20000 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    )
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    root = Path(os.environ["PS_SCRATCH_DIR"])
    publish = root / "publish"
    publish.mkdir()
    anchors = judge.extract([a.anchor], root / "anchors")
    extract(a.transport, root / "wire")
    wire = transport_root(root / "wire")
    extract(a.masks, root / "masks")
    generated = k6b.load_generated([str(a.generated)], root / "generated")
    meta, units = decode_keypoints((wire / "packets" / safe(WID) / "keypoints.kps").read_bytes())
    boxes = box_maps(units)
    previous = json.loads(a.prior.read_text())
    prior_row = next(r for r in previous["rows"] if r["id"] == WID)
    masks = {}
    for i, (unit, box) in enumerate(zip(units, boxes)):
        file = root / "masks" / "publish" / (unit["name"] + "-masks.npz")
        with np.load(file) as data:
            original = next(
                x for x in previous["prediction"]["runs"][0]["rows"] if x["name"] == unit["name"]
            )
            if (
                data["t"].tolist() != original["ticks"]
                or array_sha(data["masks"]) != original["output_sha256"]
            ):
                raise ValueError("receiver mask identity")
            masks[i] = {
                int(t): project(mask, box[int(t)]) for t, mask in zip(data["t"], data["masks"])
            }
    native = k6b.k6_window(anchors[WID])
    row = native["row"]
    source_masks = decode_masks((wire / "packets" / safe(WID) / "exact-rle.matte").read_bytes())[1]
    if sha(a.source.read_bytes()) != SOURCE:
        raise ValueError("source MP4 identity")
    source_yuv = root / "source.yuv"
    convert("ffmpeg", {"id": WID, "start": row["start"], "frames": 240}, a.source, source_yuv)
    truth = rgb_memmap(source_yuv, root / "truth.npy", 240, 1920, 1080, False, "cuda")
    regions_all = [
        {
            p: cv2.resize(
                source_masks[t, i].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            for i, p in enumerate(row["players"])
            if source_masks[t, i].any()
        }
        for t in range(240)
    ]
    heights = {
        (p, t): float(np.ptp(np.nonzero(r[p].any(1))[0]) + 1) if p in r and r[p].any() else 1.0
        for t, r in enumerate(regions_all)
        for p in row["players"]
    }
    coded = json.loads(decompress((anchors[WID] / "keypoints.json.zst").read_bytes()))["decoded"]
    decoded = {
        key: {
            (name.rsplit("@", 1)[0], int(name.rsplit("@", 1)[1])): np.asarray(v, np.float64)
            for name, v in values.items()
        }
        for key, values in coded.items()
    }
    old_base_file = root / "old-base.npy"
    tar_member(a.gpu_base, f"publish/{a.window}-s2-gpu.npy", old_base_file)
    old_base = np.load(old_base_file, mmap_mode="r")
    old_window_file = root / "old-window.json"
    old_scores_file = root / "old-scores.npz"
    tar_member(a.old_judge, "publish/" + safe(WID) + "/window.json", old_window_file)
    tar_member(a.old_judge, "publish/" + safe(WID) + "/scores.npz", old_scores_file)
    old_window = json.loads(old_window_file.read_text())
    old_scores = np.load(old_scores_file)
    candidate = root / "q62.ivf"
    tar_member(a.candidate, "publish/q62.ivf", candidate)
    source_record = json.loads((root / "candidate-result.json").read_text()) if False else None
    with tarfile.open(a.candidate) as archive:
        source_record = json.load(archive.extractfile("publish/result.json"))
    qrow = next(x for x in source_record["rows"] if x["crf"] == 62)
    if sha(candidate.read_bytes()) != qrow["stream_sha256"]:
        raise ValueError("new IVF hash")
    decoder = svtav1.tool("dav1d")
    new_yuv = root / "new.yuv"
    subprocess.run(
        svtav1.decode_command(decoder["path"], candidate, new_yuv, threads=1),
        check=True,
        timeout=600,
    )
    if sha(new_yuv.read_bytes()) != qrow["decoded_sha256"]:
        raise ValueError("new decoder hash")
    new_small = rgb_memmap(new_yuv, root / "new-small.npy", 240, 960, 544, False, "cuda")
    net = load_dists(a.dists, "cuda")
    if not next(net.parameters()).is_cuda:
        raise ValueError("DISTS not on CUDA")
    setup = time.monotonic() - began
    records = []
    for t in row["judged"][: a.ticks]:
        true = np.asarray(truth[t])
        if array_sha(true) != prior_row["source_rgb_frame_sha256"][str(t)]:
            raise ValueError("source RGB parity")
        regions = regions_all[t]
        r = np.logical_or.reduce([*regions.values(), np.zeros((1080, 1920), bool)])
        fill = k6.telea_fill(true, r)
        _, drawn = k6b.k6_oracle(
            t,
            native,
            k6.variant_names()[-9:],
            truth,
            regions_all,
            fill,
            meta["compositing_order"],
            row["scored"],
            heights,
            decoded,
            max(1, round(k6.REF_GAP_S * float(Fraction(row["fps"])))),
        )
        support = r.copy()
        for value in drawn.values():
            support |= value
        if sha(np.packbits(support).tobytes()) != row["scoring_masks"][str(t)]["s"]:
            raise ValueError("original scoring support")
        ys, xs = np.nonzero(support)
        crop = (
            slice(max(0, ys.min() - k6.DISTS_PAD), min(1080, ys.max() + 1 + k6.DISTS_PAD)),
            slice(max(0, xs.min() - k6.DISTS_PAD), min(1920, xs.max() + 1 + k6.DISTS_PAD)),
        )
        crops, player_boxes, player_masks = {}, {}, {}
        for i, unit in enumerate(units):
            player = row["players"][i]
            g = generated[(unit["name"], "match5")]
            idx = np.flatnonzero(g["t"] == t)
            if len(idx) == 1:
                crops[player] = g["crops"][int(idx[0])]
            if t in boxes[i]:
                player_boxes[player] = boxes[i][t]
            player_masks[player] = unit_mask(masks[i], len(idx) == 1, t)
        old_composite, _ = compose(
            np.asarray(old_base[t // 3]),
            crops,
            player_boxes,
            player_masks,
            meta["compositing_order"],
        )
        if array_sha(old_composite) != old_window["composite_hashes"][str(t)][5]:
            raise ValueError("historical composite pixel parity")
        old_score = dists_crops(net, [old_composite[crop]], true[crop], support[crop], "cuda")[0]
        old_expected = float(old_scores["foreground_dists"][5, t // 3])
        if abs(old_score - old_expected) > 1e-4:
            raise ValueError("historical DISTS control")
        new_base = cv2.resize(np.asarray(new_small[t]), (1920, 1080), interpolation=cv2.INTER_CUBIC)
        new_composite, _ = compose(
            new_base, crops, player_boxes, player_masks, meta["compositing_order"]
        )
        base_score, player_score = dists_crops(
            net, [new_base[crop], new_composite[crop]], true[crop], support[crop], "cuda"
        )
        records.append(
            {
                "tick": t,
                "old_score": old_score,
                "old_expected": old_expected,
                "plain_dists": base_score,
                "player_dists": player_score,
                "new_base_sha256": array_sha(new_base),
                "new_composite_sha256": array_sha(new_composite),
            }
        )
    result = {
        "window": WID,
        "stage": os.environ["PS_STAGE"],
        "ticks": a.ticks,
        "candidate": qrow,
        "records": records,
        "setup_seconds": setup,
        "elapsed_seconds": time.monotonic() - began,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    output = json.dumps(result, indent=2) + "\n"
    (publish / "result.json").write_text(output)
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(output)


def validate(a):
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "result.json").read_text())
    n = 3 if r["stage"] == "smoke" else 80
    checks = {
        "window": r["window"] == WID,
        "ticks": r["ticks"] == n and len(r["records"]) == n,
        "cuda": r["capability"] in ([8, 9], [8, 6]),
        "old_controls": all(abs(x["old_score"] - x["old_expected"]) < 1e-4 for x in r["records"]),
        "finite": all(
            np.isfinite([x["plain_dists"], x["player_dists"]]).all() for x in r["records"]
        ),
    }
    if n == 3:
        checks["forecast"] = (
            r["setup_seconds"] + (r["elapsed_seconds"] - r["setup_seconds"]) * 80 / 3 * 1.5 + 120
            < a.full_limit
        )
    result = {"passed": all(checks.values()), "checks": checks}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result) + "\n")
    if not result["passed"]:
        raise SystemExit(str(checks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "anchor",
        "transport",
        "masks",
        "generated",
        "source",
        "gpu-base",
        "old-judge",
        "candidate",
        "prior",
        "dists",
    ):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--ticks", type=int)
    p.add_argument("--window", choices=tuple(WINDOW_SOURCES), default="al034")
    p.add_argument("--full-limit", type=int, required=True)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    global WID, SOURCE
    WID = f"alcaraz_hurkacz/{a.window[2:]}@120"
    SOURCE = WINDOW_SOURCES[a.window]
    validate(a) if a.validate else run(a)


if __name__ == "__main__":
    main()
