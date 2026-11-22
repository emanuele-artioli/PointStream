"""Score fresh and previously delivered Alcaraz references on the same target bodies."""

import argparse
import io
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np
import torch

from experiments.tennis import common_player_packet as packet, k6b
from experiments.tennis.player_requalification import array_sha, sha256
from src.codecs import quality
from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("body_score", HERE.parent / "reference-body-rd-20261122/score.py")
score = module_from_spec(spec)
spec.loader.exec_module(score)


def read(path):
    return json.loads(path.read_text())


def learn(a, roots):
    arrays, provenance = {}, []
    for index, job in enumerate(a.jobs):
        for tag, stage in [("fresh", "smoke"), ("prior", "full")]:
            base = roots[tag]
            book = read(base / "result.json")
            carrier = base / f"crf{book['selected_crf']}/players.pscp"
            raw = carrier.read_bytes()
            _, _, units = packet.unpack(raw)
            name = book["foreground"][index]["unit"]
            cond = units[name][0]
            where = a.archives / job / stage
            archive = where / "published.tar"
            stamp = read(where / "worker_timing.json")
            execution = read(where / "execution.json")
            assert sha256(archive) == stamp["archive_sha256"]
            assert execution["exit_code"] == 0 and not execution["contention"]
            assert not execution["timing_contaminated"]
            with tarfile.open(archive) as f:
                gen = json.load(f.extractfile("publish/generation.json"))
                (row,) = gen["outputs"]
                checks = gen["isolation"]
                assert gen["carrier_sha256"] == sha256(carrier) and gen["frames"] == 24
                assert not gen["conditioning_only"] and checks["interfaces"] == ["lo"]
                assert min(checks["denials"].values()) >= 1 and checks["landlock_abi"] >= 3
                assert row["unit"] == name and row["seed"] == cond["seed"]
                assert row["distinct_frames"] == 24 and row["kernels"]["kernel_launches"] > 0
                assert row["array_sha256"]["reference"] == array_sha(cond["reference"])
                b = f.extractfile("publish/" + row["crops_file"]).read()
                assert packet.wire.sha(b) == row["crops_file_sha256"]
                with np.load(io.BytesIO(b), allow_pickle=False) as d:
                    assert np.array_equal(d["t"], cond["t"][:24])
                    assert np.array_equal(d["box"], cond["boxes"][:24])
                    assert d["crops"].shape == (24, 784, 512, 3)
                    arrays[index, tag] = (d["crops"], d["box"])
            provenance.append(
                dict(
                    mode=tag,
                    index=index,
                    job=job,
                    archive=sha256(archive),
                    seed=row["seed"],
                    crop=row["crops_file_sha256"],
                    device=gen["device"],
                )
            )
    return arrays, provenance


def run(a):
    torch.set_num_threads(getattr(a, "threads", 2))
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    strong = [v for v in read(a.anchors) if v["kind"] == "native_yuv420p" and v["kbps"] <= a.budget]
    assert strong and a.budget > 0 and len(a.jobs) == 2
    assert (
        abs(
            a.budget
            - next(
                q["kbps_cached"]
                for q in read(a.fresh / "result.json")["rates"]
                if q["crf"] == read(a.fresh / "result.json")["selected_crf"]
            )
        )
        < 1e-8
    )
    arrays, receipts = learn(a, dict(fresh=a.fresh, prior=a.prior))
    native = {}
    for arm in strong:
        w, h, path = arm["width"], arm["height"], Path(arm["path"])
        assert sha256(path) == arm["sha256"] and path.stat().st_size == w * h * 360
        native[arm["name"]] = (np.memmap(path, np.uint8, "r", shape=(240, h * 3 // 2, w)), w, h)
    masks = read(a.supports)
    assert sha256(a.source) == masks["source_sha256"] and sha256(a.weights) == score.WEIGHT_SHA
    rows, clock = [], time.process_time()
    net = quality.load_dists(a.weights, "cpu")
    cap = cv2.VideoCapture(str(getattr(a, "canonical", None) or a.source), cv2.CAP_FFMPEG)
    assert cap.get(cv2.CAP_PROP_FPS) == 25
    for t in range(getattr(a, "frames", 24)):
        ok, bgr = cap.read()
        assert ok
        truth = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert array_sha(truth) == masks["source_rgb_hashes"][t]
        base = {
            k: cv2.resize(
                yuv_to_rgb(d[t], w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
            )
            for k, (d, w, h) in native.items()
        }
        for i in [0, 1]:
            (mask_row,) = [r for r in masks["rows"] if r["tick"] == t and r["object_index"] == i]
            mask = np.zeros((1080, 1920), bool)
            x0, y0, x1, y1 = mask_row["xyxy"]
            mp = a.supports.parent / mask_row["mask"]
            assert sha256(mp) == mask_row["sha256"]
            mask[y0:y1, x0:x1] = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0
            assert int(mask.sum()) == mask_row["area"]
            fb, pb = arrays[i, "fresh"][1][t], arrays[i, "prior"][1][t]
            assert np.array_equal(fb, pb)
            x, y, w, h = k6b.int_box(fb)
            outs = {
                tag: cv2.resize(arrays[i, tag][0][t], (w, h), interpolation=cv2.INTER_LANCZOS4)
                for tag in ["fresh", "prior"]
            }
            outs.update({k: v[y : y + h, x : x + w] for k, v in base.items()})
            if not rows:
                outs["identity"] = truth[y : y + h, x : x + w]
            with torch.inference_mode():
                r = score.compare(net, truth, outs, mask, fb)
            if not rows:
                assert abs(r["scores"].pop("identity")["dists"]) < 1e-5
            rows.append(dict(tick=t, player=i, **r))
        if t == 1:
            assert (time.process_time() - clock) * 12 * 1.5 + 20 < 1200
    cap.release()
    means = [
        {
            tag: float(np.mean([r["scores"][tag]["dists"] for r in rows if r["player"] == i]))
            for tag in ["fresh", "prior", *native]
        }
        for i in [0, 1]
    ]
    decline = [
        m["prior"] - m["fresh"] >= 0.15 and m["prior"] > min(m[n] for n in native) for m in means
    ]
    a.out.write_text(
        json.dumps(
            dict(
                source_sha256=sha256(a.source),
                crop_receipts=receipts,
                budget_kbps=a.budget,
                supports_sha256=sha256(a.supports),
                anchors=strong,
                means=means,
                stop_prior_full=all(decline),
                rows=rows,
                cpu_seconds=time.process_time() - clock,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ["source", "supports", "weights", "anchors", "archives", "fresh", "prior", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--canonical", type=Path)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--frames", type=int, choices=(3, 24), default=24)
    p.add_argument("--jobs", nargs=2, required=True)
    p.add_argument("--budget", type=float, required=True)
    run(p.parse_args())
