"""Return only prespecified review images; every image is bound to its genuine native 240-frame receiver."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np

from experiments.tennis.player_requalification import array_sha, sha256


def add(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def export(a, arm, native, binding, cfg, out):
    assert (
        arm["kind"] in ("rgb_npy", "rgb_zseq") and cfg["source_sha256"] == binding["source_sha256"]
    )
    assert cfg["event"] == a.event and str(cfg["fps"]) == binding["fps"]
    selected = set(cfg["ticks"])
    assert len(selected) == 12 and all(0 <= t < 240 for t in selected)
    p = Path(arm["path"])
    receipt_path = Path(arm["receipt"])
    r = json.loads(receipt_path.read_text())
    assert r["ticks"] == list(range(240)) and len(r["rows"]) == 240
    assert r["wire"]["active_ticks"] == [0, 240] and r["wire"]["joint_bytes"] == arm["bytes"]
    assert (
        sha256(p) == arm["sha256"]
        and r["isolation"]["interfaces"] == ["lo"]
        and r["isolation"]["mutation_denied"]
    )
    if arm["kind"] == "rgb_npy":
        assert r["order"] == [a.event + "_far", a.event + "_near"] and r["kernels"]
        seq = np.load(p, mmap_mode="r", allow_pickle=False)
        assert seq.shape == (240, 1080, 1920, 3) and seq.dtype == np.uint8
    else:
        seq = native.SealedRgbSequence(p, r)
    pack = {
        "source_sha256": binding["source_sha256"],
        "native_fps": binding["fps"],
        "finish_job": a.producer,
        "all_240_output_sha256": arm["sha256"],
        "joint_rally_bytes": r["wire"]["joint_bytes"],
        "foreground_cache_bytes": r["wire"]["cache_bytes"],
        "wire": r["wire"],
        "receipt_sha256": sha256(receipt_path),
        "ticks": {},
    }
    with tarfile.open(out, "x") as z:
        for t in range(240):
            rgb = seq[t]
            typed = array_sha(rgb)
            assert r["rows"][t]["tick"] == t and r["rows"][t]["frame_sha256"] == typed
            if t not in selected:
                continue
            ok, png = cv2.imencode(".png", rgb[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 5])
            assert ok
            assert np.array_equal(cv2.imdecode(png, cv2.IMREAD_COLOR)[:, :, ::-1], rgb)
            raw = png.tobytes()
            fn = f"tick-{t:03}.png"
            add(z, fn, raw)
            row = dict(
                tick=t,
                file=fn,
                png_sha256=hashlib.sha256(raw).hexdigest(),
                raw_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                receipt_typed_rgb_sha256=typed,
            )
            for key, v in cfg.get("source_rgb", {}).get(str(t), {}).items():
                if key in ("raw_rgb_sha256", "typed_rgb_sha256"):
                    row["source_" + key] = v
            pack["ticks"][str(t)] = row
        assert len(pack["ticks"]) == 12
        pack["selected"] = list(pack["ticks"].values())
        add(z, "manifest.json", (json.dumps(pack, indent=2) + "\n").encode())
    assert out.stat().st_size < 70 * 1024**2
    return sha256(out)
