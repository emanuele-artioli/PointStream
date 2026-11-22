"""Score both context-matched reference settings on the 24 prespecified original Shanghai instants."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time

import cv2
import numpy as np
import torch

from experiments.tennis import common_player_packet as packet, k6b
from experiments.tennis.player_requalification import array_sha
from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
from src.codecs import quality

HERE = Path(__file__).resolve().parent
for folder in ("reference-body-rd-20261122", "test-crop-joint-20261122"):
    sys.path.insert(0, str(HERE.parent / folder))
import score as reference  # noqa: E402
import score_native  # noqa: E402


def read(p):
    return json.loads(p.read_text())


def source(a):
    grid = read(HERE.parent / "test-crop-joint-20261122/match102-source-grid.json")
    assert score_native.sha(a.source) == grid["source_sha256"]
    cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 30000 / 1001) < 0.001
    supports, ms = score_native.get_support(a.features)
    frames = []
    masks = {}
    for t in range(24):
        ok, im = cap.read()
        assert ok
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        assert array_sha(rgb) == grid["native_rgb_typed_sha256"][t]
        frames.append(rgb)
        for role, ((x0, y0, x1, y1), part) in zip(("near", "far"), supports[t], strict=True):
            m = np.zeros((1080, 1920), bool)
            m[y0:y1, x0:x1] = part
            masks[t, role] = m
    cap.release()
    return frames, masks, ms


def load_crops(a, role, stage, control, oracle):
    unit = "match102_" + role
    p = a.archives / role / stage
    zfile = p / "published.tar"
    timing = read(p / "worker_timing.json")
    execution = read(p / "execution.json")
    assert (
        execution["exit_code"] == 0
        and not execution["contention"]
        and not execution["timing_contaminated"]
    )
    assert score_native.sha(zfile) == timing["archive_sha256"]
    with tarfile.open(zfile) as z:
        g = json.load(z.extractfile("publish/generation.json"))
        (r,) = g["outputs"]
        assert (
            g["carrier_sha256"]
            == oracle["source_control_carrier" if stage == "smoke" else "avif_local_carrier"]
        )
        assert (
            g["frames"] == 24
            and not g["conditioning_only"]
            and g["device"] == "NVIDIA RTX A6000"
            and g["isolation"]["landlock_abi"] >= 3
            and g["isolation"]["interfaces"] == ["lo"]
        )
        assert (
            r["kernels"]["attention_family"]
            and r["kernels"]["tensor_core_like"]
            and r["kernels"]["kernel_launches"] > 0
        )
        expected = dict(
            t=array_sha(control["t"][:24]),
            box=array_sha(control["boxes"][:24]),
            pose=array_sha(np.stack([packet.pose_for_tick(control, t) for t in range(24)])),
            reference=array_sha(control["reference"]),
        )
        assert (
            r["unit"] == unit
            and r["shape"] == [24, 784, 512, 3]
            and r["seed"] == control["seed"]
            and r["distinct_frames"] == 24
        )
        assert all(r["array_sha256"][k] == v for k, v in expected.items())
        raw = z.extractfile("publish/" + r["crops_file"]).read()
        assert hashlib.sha256(raw).hexdigest() == r["crops_file_sha256"]
        with np.load(io.BytesIO(raw), allow_pickle=False) as n:
            assert np.array_equal(n["t"], control["t"][:24]) and np.array_equal(
                n["box"], control["boxes"][:24]
            )
            arr = n["crops"]
        assert arr.dtype == np.uint8 and arr.shape == (24, 784, 512, 3)
    return arr, dict(
        archive_sha256=timing["archive_sha256"],
        decoder=g["device"],
        device_uuid=g["gpu_uuid"],
        rendered_sha=r["crops_file_sha256"],
        seconds=timing["worker_wall_seconds"],
        conditioning=expected,
        seed=r["seed"],
    )


def measure(a):
    start = time.process_time()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)
    frames, masks, ms = source(a)
    oracle = read(HERE / "image-oracle.json")
    profiles = {
        k: packet.unpack(pa.read_bytes())[2] for k, pa in (("lossless", a.control), ("q90", a.q90))
    }
    assert (
        score_native.sha(a.control) == oracle["source_control_carrier"]
        and score_native.sha(a.q90) == oracle["avif_local_carrier"]
    )
    if a.preflight:
        return print(
            "Verified 24 native source hashes, 48 original body supports, both independently bound reference settings."
        )
    arrays = {}
    provenance = {}
    seeds = {}
    for mode, stage in (("lossless", "smoke"), ("q90", "full")):
        for role in ("near", "far"):
            unit = "match102_" + role
            cond = profiles[mode][unit][0]
            arrays[mode, role], provenance[mode, role] = load_crops(a, role, stage, cond, oracle)
            if mode == "lossless":
                seeds[role] = cond["seed"]
            assert seeds[role] == cond["seed"]
    assert score_native.sha(a.weights) == reference.WEIGHT_SHA
    net = quality.load_dists(a.weights, "cpu")
    prev = {}
    rows = []
    motion = []
    yy, xx = np.mgrid[:1080, :1920]
    with torch.inference_mode():
        for t, im in enumerate(frames):
            if t:
                flow = dis_flow(
                    cv2.cvtColor(im, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(frames[t - 1], cv2.COLOR_RGB2GRAY),
                )
                source_delta = luma(im) - luma(warp(frames[t - 1], flow))
                qx, qy = xx + flow[:, :, 0], yy + flow[:, :, 1]
            for role in ("near", "far"):
                cond = profiles["lossless"]["match102_" + role][0]
                box = cond["boxes"][t]
                x, y, w, h = k6b.int_box(box)
                sy, sx = slice(y, y + h), slice(x, x + w)
                m = masks[t, role]
                out = {
                    mode: cv2.resize(
                        arrays[mode, role][t], (w, h), interpolation=cv2.INTER_LANCZOS4
                    )
                    for mode in profiles
                }
                if t == 0 and role == "near":
                    mark = time.process_time()
                    smoke = reference.compare(
                        net, im, {"source": im[sy, sx], "lossless": out["lossless"]}, m, box
                    )
                    assert (
                        abs(smoke["scores"]["source"]["dists"]) < 1e-5
                        and smoke["scores"]["source"]["sum_squared_rgb"] == 0
                        and time.process_time() - start + 1.5 * 96 * (time.process_time() - mark)
                        < 1500
                    )
                r = reference.compare(net, im, out, m, box)
                r.update(tick=t, role=role)
                rows.append(r)
                if t:
                    px, py, pw, ph = k6b.int_box(cond["boxes"][t - 1])
                    eligible = (
                        m[sy, sx]
                        & (qx[sy, sx] >= px)
                        & (qx[sy, sx] < px + pw - 1)
                        & (qy[sy, sx] >= py)
                        & (qy[sy, sx] < py + ph - 1)
                    )
                    eligible &= warp(masks[t - 1, role].astype(np.float32), flow)[sy, sx] > 0.999
                    assert eligible.any()
                    for mode in profiles:
                        canvas = np.zeros_like(im)
                        canvas[py : py + ph, px : px + pw] = prev[mode, role]
                        diff = luma(out[mode]) - luma(warp(canvas, flow)[sy, sx])
                        motion.append(
                            dict(
                                tick=t,
                                role=role,
                                mode=mode,
                                pixels=int(eligible.sum()),
                                luma_error=float(
                                    np.abs(source_delta[sy, sx][eligible] - diff[eligible]).mean()
                                ),
                            )
                        )
                for mode in profiles:
                    prev[mode, role] = out[mode]
            assert time.process_time() - start < 1500
    assert len(rows) == 48 and len(motion) == 92
    result = {}
    for m in profiles:
        result[m] = {}
        for role in ("near", "far", "both"):
            sub = [r["scores"][m] for r in rows if role == "both" or role == r["role"]]
            tm = [
                r["luma_error"]
                for r in motion
                if r["mode"] == m and (role == "both" or role == r["role"])
            ]
            result[m][role] = dict(
                body_dists=float(np.mean([r["dists"] for r in sub])),
                luma_temporal_error=float(np.mean(tm)),
                person_ticks=len(sub),
                transitions=len(tm),
            )
    delta = result["q90"]["both"]["body_dists"] - result["lossless"]["both"]["body_dists"]
    summary = dict(
        source=oracle["source"],
        mask_rle=ms,
        decision=bool(oracle["image_saving"] >= 0.3 and delta <= 0.01),
        delta_dists=delta,
        delta_temporal=result["q90"]["both"]["luma_temporal_error"]
        - result["lossless"]["both"]["luma_temporal_error"],
        reference_image_saving=oracle["image_saving"],
        prices=oracle["complete_initialization_bytes"],
        original_image_bytes=oracle["source_webp_bytes"],
        q90_image_bytes=oracle["reference_q90_bytes"],
        metric=result,
        source_masks="original native source-model supports",
        scorer=score_native.sha(Path(__file__)),
        cpu_seconds=time.process_time() - start,
        producers={k[0] + "-" + k[1]: v for k, v in provenance.items()},
        rows=rows,
        temporal=motion,
    )
    with a.out.open("x") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(delta, summary["delta_temporal"], summary["decision"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("source", "features", "control", "q90", "weights", "archives", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--preflight", action="store_true")
    measure(p.parse_args())
