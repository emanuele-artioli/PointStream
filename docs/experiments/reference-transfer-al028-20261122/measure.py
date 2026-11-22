"""Score the prospectively chosen Al028 reference arms on identical original player pixels."""

import argparse
import base64
import hashlib
import io
from importlib import import_module
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time

import cv2
import numpy as np
import torch
from experiments.tennis import common_player_packet as packet, k6b, k6b_rate_streams as rates
from experiments.tennis.player_requalification import array_sha
from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
from src.codecs import quality

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference-body-rd-20261122"))
score = import_module("score")

SOURCE = "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934"
MASKS = "8b0871ab633edcf4c77cb810cf2d3617a2a9593f0aa51af451c2a2ec21e30607"
LABELS = "098fc0b387cbc895fb64c543670023f87a26da155a68383a0b80d540ce6ab3f9"


def load(p):
    return json.loads(p.read_text())


def original(a):
    assert (
        score.digest(a.source) == SOURCE
        and score.digest(a.masks) == MASKS
        and score.digest(a.labels) == LABELS
    )
    cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
    assert [
        cap.get(x)
        for x in (
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
            cv2.CAP_PROP_FRAME_COUNT,
        )
    ] == [1920, 1080, 25, 492]
    images = []
    for tick in range(24):
        ok, img = cap.read()
        assert ok
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    r = load(a.labels)["frames"][0]
    assert r["tick"] == 10
    # Match the source raster presented to the reviewer independently of PNG byte serialization.
    frame = a.review / "assets/source-010.png"
    assert score.digest(frame) == r["source_png_sha256"]
    assert np.array_equal(images[10], cv2.imread(str(frame))[:, :, ::-1])
    with tarfile.open(a.masks) as z:
        _, masks = rates.decode_masks(
            z.extractfile("publish/packets/alcaraz_hurkacz__028@120/exact-rle.matte").read()
        )
    assert masks.shape == (240, 2, 540, 960)
    native = {
        (t, i): cv2.resize(
            masks[t, i].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        for t in range(24)
        for i in (0, 1)
    }
    for i, body in enumerate(r["players"]):
        assert body["id"] == f"archived-track-{i}" and body["status"] == "accept"
        png = base64.b64decode(body["mask"]["png_base64"], validate=True)
        assert hashlib.sha256(png).hexdigest() == body["mask"]["sha256"]
        assert np.array_equal(
            cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE) > 0, native[10, i]
        )
    return images, native


def source_tracks(a, masks):
    known = load(HERE / "image-oracle.json")
    assert score.digest(a.carrier) == known["source_carrier_sha256"]
    _, _, cond = packet.unpack(a.carrier.read_bytes())
    tracks = {}
    for name, rs in cond.items():
        p = int(name.rsplit("_", 1)[-1])
        assert p in (4, 6) and len(rs) == 1 and np.array_equal(rs[0]["t"][:24], np.arange(24))
        boxes = rs[0]["boxes"][:24]
        appearances = []
        for t, b in enumerate(boxes):
            x, y, w, h = k6b.int_box(b)
            assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080
            overlap = [masks[t, i][y : y + h, x : x + w].sum() / masks[t, i].sum() for i in (0, 1)]
            assert min(overlap) == 0 and max(overlap) > 0.5
            appearances.append(int(np.argmax(overlap)))
        assert len(set(appearances)) == 1
        tracks[p] = (rs[0], appearances[0], name)
    assert {v[1] for v in tracks.values()} == {0, 1}
    return tracks


def archives(a, tracks):
    jobs = load(a.jobs)
    oracle = load(HERE / "image-oracle.json")
    extra = load(HERE / "jpeg-webp-oracle.json")["rows"]
    expected = {
        "lossless": oracle["source_carrier_sha256"],
        "q90": oracle["avif_carrier_sha256"],
        **{m: extra[m]["carrier_sha256"] for m in ("jpeg40", "jpeg70")},
    }
    checked_refs = {}
    for mode, path in (
        ("q90", a.avif_carrier),
        ("jpeg40", a.jpeg_root / "jpeg40/players.pscp"),
        ("jpeg70", a.jpeg_root / "jpeg70/players.pscp"),
    ):
        assert score.digest(path) == expected[mode]
        _, _, parsed = packet.unpack(path.read_bytes())
        checked_refs[mode] = {unit: record[0]["reference"] for unit, record in parsed.items()}
    dims = {}
    arrays = {}
    payloads = {}
    previous = None
    for j in jobs:
        for stage, mode in j["arms"].items():
            node = a.archives / j["job_id"] / stage
            ar = node / "published.tar"
            assert score.digest(ar) == load(node / "worker_timing.json")["archive_sha256"]
            execfile = node / "execution.json"
            ex = load(execfile)
            assert ex["exit_code"] == 0 and not ex["timing_contaminated"] and not ex["contention"]
            with tarfile.open(ar) as tar:
                g = json.load(tar.extractfile("publish/generation.json"))
                assert (
                    g["carrier_sha256"] == expected[mode]
                    and len(g["outputs"]) == 1
                    and g["frames"] == 24
                    and not g["conditioning_only"]
                )
                assert (
                    g["isolation"]["interfaces"] == ["lo"]
                    and min(g["isolation"]["denials"].values()) >= 1
                )
                assert g["device"] == "NVIDIA RTX A6000" and g["isolation"]["landlock_abi"] >= 3
                identifier = {
                    k: g[k]
                    for k in (
                        "device",
                        "gpu_uuid",
                        "cuda",
                        "torch",
                        "model_record",
                        "weight_hashes",
                        "common_fit_sha256",
                        "producer_sources",
                    )
                }
                if previous is not None:
                    assert previous == identifier
                previous = identifier
                row = g["outputs"][0]
                t = j["track"]
                cond, _, unit = tracks[t]
                assert (
                    row["unit"] == unit
                    and row["seed"] == cond["seed"]
                    and row["distinct_frames"] == 24
                )
                assert (
                    row["kernels"]["attention_family"]
                    and row["kernels"]["tensor_core_like"]
                    and row["kernels"]["kernel_launches"] > 0
                )
                s = row["array_sha256"]
                assert s["t"] == array_sha(cond["t"][:24]) and s["box"] == array_sha(
                    np.asarray(cond["boxes"][:24])
                )
                assert s["pose"] == array_sha(
                    np.stack([packet.pose_for_tick(cond, x) for x in range(24)])
                )
                if mode == "lossless":
                    assert s["reference"] == array_sha(cond["reference"])
                else:
                    rr = oracle["reference_rows"] if mode == "q90" else extra[mode]["rows"]
                    unit_order = sorted(tracks).index(t)
                    native = checked_refs[mode][unit]
                    assert (
                        rr[f"ref{unit_order}"]["decoded_rgb_sha256"]
                        == hashlib.sha256(native.tobytes()).hexdigest()
                    )
                    assert array_sha(native) == s["reference"]
                blob = tar.extractfile("publish/" + row["crops_file"]).read()
                assert hashlib.sha256(blob).hexdigest() == row["crops_file_sha256"]
                with np.load(io.BytesIO(blob), allow_pickle=False) as z:
                    assert np.array_equal(z["t"], cond["t"][:24]) and np.array_equal(
                        z["box"], cond["boxes"][:24]
                    )
                    assert z["crops"].shape == (24, 784, 512, 3)
                    arrays[mode, t] = z["crops"]
                payloads[f"{mode}-{t}"] = dict(
                    job_id=j["job_id"],
                    stage=stage,
                    native_archive_sha256=score.digest(ar),
                    output_sha256=row["crops_file_sha256"],
                    conditioning_arrays=s,
                    seed=row["seed"],
                    kernels=row["kernels"]["kernel_launches"],
                    archive_worker_seconds=load(node / "worker_timing.json")["worker_wall_seconds"],
                )
                dims[mode] = 1
    modes = list(dims)
    if not a.audit_one:
        assert {"lossless", "q90"} <= set(modes) and len(arrays) == 2 * len(modes)
    return modes, arrays, payloads, previous


def summarize(rows, modes):
    answer = {}
    for mode in modes:
        people = {}
        for p in (4, 6):
            records = [row["scores"][mode] for row in rows if row["person"] == p]
            people[str(p)] = dict(
                voxels=sum(r["samples"] for r in records) // 3,
                ticks=len(records),
                dists=float(np.mean([r["dists"] for r in records])),
                rgb_psnr=10
                * math.log10(
                    255**2
                    * sum(r["samples"] for r in records)
                    / sum(r["sum_squared_rgb"] for r in records)
                ),
            )
        answer[mode] = dict(
            per_player=people,
            **{f: float(np.mean([v[f] for v in people.values()])) for f in ("dists", "rgb_psnr")},
        )
    return answer


def assess(a):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    started = time.process_time()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    frames, masks = original(a)
    tracks = source_tracks(a, masks)
    if a.preflight:
        print({t: (r[1], r[2]) for t, r in tracks.items()})
        return
    modes, arrays, inputs, model = archives(a, tracks)
    if a.audit_one:
        print({k: payload["output_sha256"][:12] for k, payload in inputs.items()})
        return
    assert score.digest(a.weights) == score.WEIGHT_SHA
    net = quality.load_dists(a.weights, "cpu")
    rows, temporal = [], []
    full_crop_sse = {(mode, person): 0 for mode in modes for person in tracks}
    prev_native = {}
    yy, xx = np.mgrid[:1080, :1920]
    with torch.inference_mode():
        for t, frame in enumerate(frames):
            if t:
                flow = dis_flow(
                    cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(frames[t - 1], cv2.COLOR_RGB2GRAY),
                )
                source_delta = luma(frame) - luma(warp(frames[t - 1], flow))
                qx, qy = xx + flow[:, :, 0], yy + flow[:, :, 1]
            for p, (cond, i, _) in tracks.items():
                box = cond["boxes"][t]
                x, y, w, h = k6b.int_box(box)
                mask = masks[t, i]
                ref_crop = k6b.crop_image(frame, box).astype(np.int32)
                for mode in modes:
                    diff = arrays[mode, p][t].astype(np.int32) - ref_crop
                    full_crop_sse[mode, p] += int(np.square(diff).sum(dtype=np.int64))
                out = {
                    m: cv2.resize(arrays[m, p][t], (w, h), interpolation=cv2.INTER_LANCZOS4)
                    for m in modes
                }
                if t == 0 and p == 4:
                    smoke_start = time.process_time()
                    control = score.compare(
                        net,
                        frame,
                        {"truth": frame[y : y + h, x : x + w], modes[0]: out[modes[0]]},
                        mask,
                        box,
                    )
                    assert (
                        abs(control["scores"]["truth"]["dists"]) < 1e-5
                        and control["scores"]["truth"]["sum_squared_rgb"] == 0
                    )
                    assert (
                        time.process_time()
                        - started
                        + 1.5 * 50 * len(modes) * (time.process_time() - smoke_start)
                        < 2400
                    )
                row = score.compare(net, frame, out, mask, box)
                row.update(tick=t, person=p, source_proposal=i, human_accepted=t == 10)
                rows.append(row)
                if t:
                    px, py, pw, ph = k6b.int_box(cond["boxes"][t - 1])
                    sy, sx = slice(y, y + h), slice(x, x + w)
                    eligible = (
                        mask[sy, sx]
                        & (qx[sy, sx] >= px)
                        & (qx[sy, sx] < px + pw - 1)
                        & (qy[sy, sx] >= py)
                        & (qy[sy, sx] < py + ph - 1)
                    )
                    eligible &= warp(masks[t - 1, i].astype(np.float32), flow)[sy, sx] > 0.999
                    assert eligible.any()
                    for m in modes:
                        canvas = np.zeros_like(frame)
                        canvas[py : py + ph, px : px + pw] = prev_native[m, p]
                        diff = luma(out[m]) - luma(warp(canvas, flow)[sy, sx])
                        temporal.append(
                            dict(
                                tick=t,
                                person=p,
                                mode=m,
                                pixels=int(eligible.sum()),
                                luma_error=float(
                                    np.abs(source_delta[sy, sx][eligible] - diff[eligible]).mean()
                                ),
                            )
                        )
                for m in modes:
                    prev_native[m, p] = out[m]
            assert time.process_time() - started < 2400
    result = dict(
        source_sha256=SOURCE,
        source_labels_sha256=LABELS,
        matte_sha256=MASKS,
        weights_sha256=score.WEIGHT_SHA,
        scorer_sha256=score.digest(Path(__file__)),
        shared_scorer_sha256=score.digest(Path(score.__file__)),
        cpu_seconds=time.process_time() - started,
        cpu_pytorch=str(torch.__version__),
        opencv=cv2.__version__,
        gpu_model=model,
        producers=inputs,
        full_crop_psnr={
            mode: {
                str(person): 10
                * math.log10(255**2 * arrays[mode, person].size / full_crop_sse[mode, person])
                for person in tracks
            }
            for mode in modes
        },
        body_scores=rows,
        source_model=summarize(rows, modes),
        human_accepted=summarize([r for r in rows if r["tick"] == 10], modes),
        temporal_scores=temporal,
        temporal={
            m: float(np.mean([r["luma_error"] for r in temporal if r["mode"] == m])) for m in modes
        },
        q90_tolerance_pass=summarize(rows, modes)["q90"]["dists"]
        <= summarize(rows, modes)["lossless"]["dists"] + 0.01,
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                m: [
                    result["source_model"][m]["dists"],
                    result["source_model"][m]["rgb_psnr"],
                    result["temporal"][m],
                ]
                for m in modes
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in (
        "source",
        "masks",
        "labels",
        "review",
        "carrier",
        "archives",
        "jobs",
        "weights",
        "output",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--avif-carrier", type=Path, required=True)
    p.add_argument("--jpeg-root", type=Path, required=True)
    p.add_argument("--audit-one", action="store_true")
    p.add_argument("--preflight", action="store_true")
    assess(p.parse_args())
