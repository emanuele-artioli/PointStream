"""Validate the paired cross-shot renders and build a blinded private source review."""
import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.tennis.k6b import crop_image
from experiments.tennis.player_reference_review import resolved
from experiments.tennis.player_requalification import make_plan, sha256, validate_output
from experiments.tennis.player_requalification_preview import encode

UNITS = ["nadal_kyrgios__002_at120__person_000000000000",
         "nadal_kyrgios__002_at120__person_000000000002"]
SOURCE_SHA = "f0551443eaff4e4b3539f21cd0c6ac66f55601d480ad099a571e8626db735e4a"


def checked(args, label):
    config_path = getattr(args, label + "_config")
    config_hash = getattr(args, label + "_hash")
    conditioning = getattr(args, label + "_conditioning")
    smoke = getattr(args, label + "_smoke")
    full = getattr(args, label + "_full")
    config = resolved(config_path, config_hash, conditioning, args.base)
    assert [unit["unit"] for unit in config["units"]] == UNITS
    assert config["source"]["source_video_sha256"] == SOURCE_SHA
    s = validate_output(smoke, config, config_hash, "tennis_only", 1)
    f = validate_output(full, config, config_hash, "tennis_only", 2)
    assert s["plan"]["units"][0] == f["plan"]["units"][0]
    with np.load(smoke / f"{UNITS[0]}@tennis_only.npz", allow_pickle=False) as x, \
            np.load(full / f"{UNITS[0]}@tennis_only.npz", allow_pickle=False) as y:
        assert set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
    return config, f


def source_crops(source, units, config):
    import cv2

    if sha256(source) != SOURCE_SHA:
        raise ValueError("Source hash mismatch")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
                            "-of", "json", str(source)], capture_output=True, check=True, timeout=30)
    stream, = json.loads(probe.stdout)["streams"]
    if (stream["width"], stream["height"], stream["r_frame_rate"],
            stream["avg_frame_rate"], stream["nb_frames"]) != (1920, 1080, "50/1", "50/1", "240"):
        raise ValueError("Source geometry or cadence mismatch")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(source),
                          "-frames:v", "24", "-vf",
                          "select=between(n\\,0\\,23),scale=1920:1080:flags=area:out_range=tv,format=yuv420p",
                          "-filter_threads", "1", "-fps_mode", "passthrough", "-threads", "1",
                          "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"],
                         capture_output=True, check=True, timeout=60).stdout
    w, h = 1920, 1080
    size = w * h * 3 // 2
    if len(raw) != size * 24:
        raise ValueError("Incomplete source decode")
    references = []
    for unit in config["units"]:
        with np.load(unit["path"], allow_pickle=False) as arrays:
            references.append(arrays["reference_box"].copy())
    source = [[] for _ in units]
    deltas = []
    for tick in range(24):
        frame = np.frombuffer(raw, dtype=np.uint8, count=size, offset=tick * size)
        y = frame[:w*h].reshape(h, w).astype(np.float32)
        u = frame[w*h:w*h+w*h//4].reshape(h//2, w//2).astype(np.float32)
        v = frame[w*h+w*h//4:].reshape(h//2, w//2).astype(np.float32)
        cb = (cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR) - 128.0) / 224.0
        cr = (cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) - 128.0) / 224.0
        yn = (y - 16.0) / 219.0
        red = yn + 2.0 * (1.0 - 0.2126) * cr
        blue = yn + 2.0 * (1.0 - 0.0722) * cb
        green = (yn - 0.2126 * red - 0.0722 * blue) / (1.0 - 0.2126 - 0.0722)
        rgb = np.clip(np.rint(np.stack((red, green, blue), axis=2) * 255.0), 0, 255).astype(np.uint8)
        for i, data in enumerate(units):
            source[i].append(crop_image(rgb, data["box"][tick]))
            if tick == 0:
                ref = crop_image(rgb, references[i])
                deltas.append(int(np.max(np.abs(ref.astype(np.int16) - data["reference"].astype(np.int16)))))
    if any(deltas):
        raise ValueError(f"Source and reference crop differ: {deltas}")
    return [np.stack(row) for row in source], deltas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arm in ("fresh", "reuse"):
        for field in ("config", "conditioning", "smoke", "full"):
            p.add_argument(f"--{arm}-{field}", type=Path, required=True)
        p.add_argument(f"--{arm}-hash", required=True)
    for field in ("base", "source", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    args = p.parse_args()
    fresh, fresh_receipt = checked(args, "fresh")
    reuse, reuse_receipt = checked(args, "reuse")
    fresh_plan, fresh_arrays = make_plan(fresh, args.fresh_hash, 2)
    reuse_plan, reuse_arrays = make_plan(reuse, args.reuse_hash, 2)
    assert fresh_plan["source"] == reuse_plan["source"] and fresh_plan["settings"] == reuse_plan["settings"]
    for f, r, fa, ra in zip(fresh_plan["units"], reuse_plan["units"], fresh_arrays, reuse_arrays):
        assert all(f[key] == r[key] for key in ("unit", "run", "seed", "t", "pose_sha256", "box_sha256"))
        assert all(np.array_equal(fa[key], ra[key]) for key in ("pose", "t", "box"))
        assert not np.array_equal(fa["reference"], ra["reference"])
    sources, deltas = source_crops(args.source, fresh_arrays, fresh)
    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(202611220202)
    mapping = {}
    sections = []
    for i, unit in enumerate(UNITS):
        order = ["fresh", "reuse"]
        rng.shuffle(order)
        mapping[unit] = dict(zip("AB", order))
        for fps in (10, 50):
            encode(sources[i], args.output / f"{unit}-source-{fps}.mp4", fps)
        panels = []
        for label, arm in mapping[unit].items():
            full = getattr(args, arm + "_full")
            with np.load(full / f"{unit}@tennis_only.npz", allow_pickle=False) as data:
                crops = data["crops"]
                for fps in (10, 50):
                    encode(crops, args.output / f"{unit}-{label}-{fps}.mp4", fps)
                Image.fromarray(crops[12]).save(args.output / f"{unit}-{label}.png")
            panels.append(f'<div><b>{label}</b><video controls loop src="{unit}-{label}-50.mp4"></video>'
                          f'<p>Slowed to 10 fps</p><video controls loop src="{unit}-{label}-10.mp4"></video>'
                          f'<img src="{unit}-{label}.png"></div>')
        sections.append(f'<h2>Track {i + 1}</h2><div class="grid"><div><b>Source</b>'
                        f'<video controls loop src="{unit}-source-50.mp4"></video>'
                        f'<p>Slowed to 10 fps</p><video controls loop src="{unit}-source-10.mp4"></video>'
                        f'</div>{"".join(panels)}</div>')
    page = ('<!doctype html><meta charset="utf-8"><title>Cross-shot player appearance</title>'
            '<style>.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1em}video,img{width:100%}</style>'
            '<h1>Cross-shot player appearance</h1><p>24 native frames. Judge A and B for body, clothing, '
            'identity, pose and racket against the source before opening the reveal file. '
            'The 10 fps clips are slowed.</p>' + "".join(sections))
    (args.output / "review.html").write_text(page)
    files = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(args.output.iterdir())}
    receipt = {"mappings": mapping, "source_sha256": SOURCE_SHA, "source_reference_max_delta": deltas,
               "fresh_full_receipt_sha256": sha256(args.fresh_full / "receipt.json"),
               "reuse_full_receipt_sha256": sha256(args.reuse_full / "receipt.json"),
               "fresh_output_units": len(fresh_receipt["outputs"]),
               "reuse_output_units": len(reuse_receipt["outputs"]), "files": files}
    (args.output / "reveal.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"source_reference_max_delta": deltas, "files": files}))


if __name__ == "__main__":
    main()
