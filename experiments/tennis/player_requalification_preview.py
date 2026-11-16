"""Build a local, blind development preview of authenticated player contexts."""
from __future__ import annotations

import argparse
import copy
import html
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.tennis.player_requalification import (
    PROFILES, SAFE, compare_receipts, digest_json, make_plan,
    sha256,
)


def resolve_units(config: dict[str, Any], bindings: list[list[str]]) -> dict[str, Any]:
    """Resolve only consumed unit files, with exact binding and containment checks."""
    result = copy.deepcopy(config)
    roots: dict[str, Path] = {}
    for name, raw in bindings:
        if not SAFE.fullmatch(name) or name in roots:
            raise ValueError("invalid or duplicate binding")
        roots[name] = Path(raw).resolve(strict=True)
    used = set()
    for unit in result["units"]:
        value = unit.get("path")
        if not isinstance(value, str):
            raise ValueError("unit path required")
        if not value.startswith("staged:"):
            if not Path(value).is_absolute():
                raise ValueError("direct unit paths must be absolute")
            continue
        parts = value.removeprefix("staged:").split("/")
        if not SAFE.fullmatch(parts[0]) or any(not p or p in (".", "..") for p in parts):
            raise ValueError("invalid staged unit path")
        name = parts[0]
        if name not in roots:
            raise ValueError(f"missing staged unit binding: {name}")
        root = roots[name]
        target = root.joinpath(*parts[1:]).resolve(strict=True)
        if not target.is_relative_to(root):
            raise ValueError("staged unit path escapes root")
        unit["path"] = str(target)
        used.add(name)
    if used != set(roots):
        raise ValueError("extra staged unit binding")
    return result


def authenticate(config_path: Path, config_hash: str, bindings: list[list[str]],
                 folders: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if sha256(config_path) != config_hash:
        raise ValueError("config hash mismatch")
    config = resolve_units(json.loads(config_path.read_text()), bindings)
    receipts = [json.loads((p / "receipt.json").read_text()) for p in folders]
    profiles = [r.get("profile") for r in receipts]
    if len(folders) not in (1, 3) or len(set(profiles)) != len(profiles) or any(p not in PROFILES for p in profiles):
        raise ValueError("provide one profile or all three distinct profiles")
    if len(receipts) == 3:
        compare_receipts(receipts)
    units = receipts[0].get("plan", {}).get("units", [])
    if not units or len(units) != len(config.get("units", [])):
        raise ValueError("full ordered unit coverage required")
    plan, arrays = make_plan(config, config_hash, len(units))
    if plan["source"].get("native_fps") != [50, 1]:
        raise ValueError("preview requires the declared native 50 Hz")
    for folder, receipt in zip(folders, receipts):
        profile = receipt["profile"]
        if (receipt.get("schema") != 1 or receipt.get("scope") != "24-native-frame-context-only"
                or receipt.get("plan") != plan or receipt.get("plan_sha256") != digest_json(plan)):
            raise ValueError("receipt plan mismatch")
        records = receipt.get("outputs")
        if not isinstance(records, list) or len(records) != len(units):
            raise ValueError("output record coverage mismatch")
        for entry, data, record in zip(plan["units"], arrays, records):
            name = entry["unit"]
            if not SAFE.fullmatch(name):
                raise ValueError("unsafe unit name")
            path = folder / f"{name}@{profile}.npz"
            sidecar = json.loads(path.with_suffix(".json").read_text())
            if (record != sidecar or record.get("unit") != name or record.get("profile") != profile
                    or record.get("plan_sha256") != digest_json(plan)
                    or record.get("reference_sha256") != entry["reference_sha256"]
                    or record.get("pose_sha256") != entry["pose_sha256"]
                    or record.get("output_sha256") != sha256(path)):
                raise ValueError("output hash or sidecar mismatch")
            with np.load(path, allow_pickle=False) as result:
                if (set(result.files) != {"t", "box", "crops"}
                        or not np.array_equal(result["t"], data["t"])
                        or not np.array_equal(result["box"], data["box"])
                        or result["crops"].shape != (24, *data["reference"].shape)
                        or result["crops"].dtype != np.uint8):
                    raise ValueError("output ticks, boxes or crops mismatch")
    return plan, arrays, receipts


def encode(crops: np.ndarray, path: Path, fps: int) -> None:
    h, w = crops.shape[1:3]
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", "pipe:0",
                    "-an", "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-filter_threads", "1",
                    "-c:v", "libx264", "-threads", "1", "-crf", "10", "-pix_fmt", "yuv420p",
                    "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-movflags", "+faststart", "-n", str(path)], input=crops.tobytes(), check=True)


def source_crops(source: Path, plan: dict[str, Any], arrays: list[dict[str, Any]],
                 config: Path, bindings: list[list[str]]) -> tuple[list[np.ndarray], list[int]]:
    """Reproduce K4 window conversion and K6's limited-range RGB view."""
    from experiments.tennis.k6b import crop_image
    import cv2

    info = plan["source"]
    if sha256(source) != info.get("source_video_sha256"):
        raise ValueError("source hash mismatch")
    if (info.get("window") != "nadal_kyrgios/000@120" or info.get("width") != 1920
            or info.get("height") != 1080 or info.get("native_fps") != [50, 1]
            or any(entry["t"] != list(range(24)) for entry in plan["units"])):
        raise ValueError("unsupported source window, geometry or tick offset")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
                            "-of", "json", str(source)], capture_output=True, check=True, timeout=30)
    streams = json.loads(probe.stdout)["streams"]
    if (len(streams) != 1 or streams[0].get("width") != 1920 or streams[0].get("height") != 1080
            or streams[0].get("r_frame_rate") != "50/1" or streams[0].get("avg_frame_rate") != "50/1"
            or streams[0].get("nb_frames") != "240"):
        raise ValueError("source geometry, cadence or frame count mismatch")
    resolved = resolve_units(json.loads(config.read_text()), bindings)
    references = []
    for unit in resolved["units"]:
        with np.load(unit["path"], allow_pickle=False) as f:
            box = f["reference_box"]
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
                raise ValueError("invalid reference box")
            references.append(box.copy())
    # Same K4 select/area/limited YUV420 conversion; no temporal resampling.
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(source),
                          "-frames:v", "24", "-vf", "select=between(n\\,0\\,23),scale=1920:1080:flags=area:out_range=tv,format=yuv420p",
                          "-filter_threads", "1", "-fps_mode", "passthrough", "-threads", "1", "-f", "rawvideo",
                          "-pix_fmt", "yuv420p", "pipe:1"], capture_output=True, check=True, timeout=30).stdout
    w, h = 1920, 1080
    size = w * h * 3 // 2
    if len(raw) != size * 24:
        raise ValueError("incomplete decoded source")
    source_arrays: list[list[np.ndarray]] = [[] for _ in arrays]
    deltas = []
    for t in range(24):
        frame = np.frombuffer(raw, dtype=np.uint8, count=size, offset=t * size)
        y = frame[:w*h].reshape(h, w).astype(np.float32)
        u = frame[w*h:w*h+w*h//4].reshape(h//2, w//2).astype(np.float32)
        v = frame[w*h+w*h//4:].reshape(h//2, w//2).astype(np.float32)
        # OpenCV's half-pixel bilinear coordinates match interpolate(align_corners=False).
        cb = (cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR) - 128.0) / 224.0
        cr = (cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) - 128.0) / 224.0
        yn = (y - 16.0) / 219.0
        red = yn + 2.0 * (1.0 - 0.2126) * cr
        blue = yn + 2.0 * (1.0 - 0.0722) * cb
        green = (yn - 0.2126 * red - 0.0722 * blue) / (1.0 - 0.2126 - 0.0722)
        rgb = np.clip(np.rint(np.stack((red, green, blue), axis=2) * 255.0), 0, 255).astype(np.uint8)
        for i, data in enumerate(arrays):
            source_arrays[i].append(crop_image(rgb, data["box"][t]))
            if t == 0:
                ref = crop_image(rgb, references[i])
                delta = int(np.max(np.abs(ref.astype(np.int16) - data["reference"].astype(np.int16))))
                deltas.append(delta)
    if any(deltas):
        raise ValueError(f"source/reference alignment unresolved; max RGB deltas {deltas}")
    return [np.stack(crops) for crops in source_arrays], deltas


def build(config: Path, config_hash: str, bindings: list[list[str]], folders: list[Path],
          output: Path, seed: int, slow: int | None = None, source: Path | None = None) -> None:
    plan, arrays, receipts = authenticate(config, config_hash, bindings, folders)
    if slow is not None and not 1 <= slow < 50:
        raise ValueError("slowed cadence must be 1–49 frames per second")
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("output must be outside the checkout")
    source_data, deltas = source_crops(source, plan, arrays, config, bindings) if source else (None, None)
    output.mkdir(parents=False, exist_ok=False)
    rng = random.Random(seed)
    paired = len(receipts) == 3
    mappings: dict[str, dict[str, str]] = {}
    sections = []
    for unit_index, (entry, data) in enumerate(zip(plan["units"], arrays)):
        unit = entry["unit"]
        order = list(range(len(folders)))
        rng.shuffle(order)
        mappings[unit] = {}
        Image.fromarray(data["reference"]).save(output / f"{unit}-reference.png")
        panels = []
        for i, index in enumerate(order):
            label = chr(65 + i)
            profile = receipts[index]["profile"]
            mappings[unit][label] = profile
            with np.load(folders[index] / f"{unit}@{profile}.npz", allow_pickle=False) as result:
                crops = result["crops"]
                Image.fromarray(crops[12]).save(output / f"{unit}-{label}-tick12.png")
                encode(crops, output / f"{unit}-{label}-50.mp4", 50)
                if slow is not None:
                    encode(crops, output / f"{unit}-{label}-{slow}.mp4", slow)
            video = f'<video controls loop preload="metadata" src="{html.escape(unit)}-{label}-50.mp4"></video>'
            if slow is not None:
                video += f'<p>Slowed playback: {slow} fps (changed cadence)</p><video controls loop preload="metadata" src="{html.escape(unit)}-{label}-{slow}.mp4"></video>'
            panels.append(f'<div><b>{label}</b><p>Native 50 Hz</p>{video}<p>Exact RGB still, tick 12</p><img src="{html.escape(unit)}-{label}-tick12.png"></div>')
        if source_data is not None:
            assert deltas is not None
            sc = source_data[unit_index]
            Image.fromarray(sc[12]).save(output / f"{unit}-source-tick12.png")
            encode(sc, output / f"{unit}-source-50.mp4", 50)
            source_video = f'<video controls loop preload="metadata" src="{unit}-source-50.mp4"></video>'
            if slow is not None:
                encode(sc, output / f"{unit}-source-{slow}.mp4", slow)
                source_video += f'<p>Slowed playback: {slow} fps (changed cadence)</p><video controls loop preload="metadata" src="{unit}-source-{slow}.mp4"></video>'
            panels.insert(0, f'<div><b>Source player crop, native 50 Hz</b><p>Same per-tick boxes; preparation RGB conversion. Reference max RGB delta: {deltas[unit_index]}.</p>{source_video}<p>Exact source crop RGB still, tick 12</p><img src="{unit}-source-tick12.png"></div>')
        options = ''.join(f'<option>{chr(65+i)}</option>' for i in range(len(order))) if paired else ''
        sections.append(f'<section><h2>Player {unit_index + 1} — ticks {entry["t"][0]}–{entry["t"][-1]}</h2>'
                        f'<div class="grid"><div><b>Reference crop, tick 0</b><p>Conditioning reference; not source-ground-truth video.</p><img src="{html.escape(unit)}-reference.png"></div>{"".join(panels)}</div>'
                        f'<label>Preference <select data-unit="{html.escape(unit)}"><option value="">Unanswered</option>{options}{'<option>Tie</option><option>None acceptable</option>' if paired else ''}<option>Uncertain</option></select></label> '
                        f'<label>Notes <input data-note="{html.escape(unit)}"></label></section>')
    # Reveal data remains local in the page; it is hidden, not cryptographically secret.
    reveal = json.dumps(mappings, ensure_ascii=True).replace("<", "\\u003c")
    status = "Paired three-profile preview" if paired else "INCOMPLETE: one profile only; no paired comparison or ranking"
    source_description = ("Source bytes authenticated; window starts at source frame 0; reference crops match exactly." if source else "Source video bytes are not checked by this preview; the source identity is a frozen-plan declaration.")
    page = f'''<!doctype html><meta charset="utf-8"><title>Player context review</title>
<style>body{{font:16px sans-serif;margin:2em}}.grid{{display:flex;gap:1em;align-items:start}}.grid>div{{flex:1;min-width:0}}video,img{{width:100%}}section{{border-top:1px solid #aaa;padding:1em 0}}input{{width:20em}}</style>
<h1>{status}</h1><p>Development evidence only. Model-context qualification; no codec or bitrate comparison. Each clip contains 24 native ticks (0.48 s at 50 Hz). Reference is a still crop, not source video. {source_description} The clips are individually controlled and may not play synchronously. The loop joins the last frame back to the first; that seam is not a temporal measurement. Videos are BT.709 display proxies; PNG stills retain exact generated RGB.</p>
<p>Judge player body/clothing, pose, missing parts and stability. These are crops before foreground compositing; court texture is not the final coded background. Racket improvement is optional. Full-screen playback helps inspect detail.</p>
{''.join(sections)}<button id="reveal">Reveal profile labels</button><pre id="labels" hidden></pre><button id="export">Export review decisions</button>
<script>const mapping={reveal};const status={json.dumps(status)};document.querySelector('#reveal').onclick=()=>{{document.querySelector('#labels').hidden=false;document.querySelector('#labels').textContent=JSON.stringify(mapping,null,2)}};document.querySelector('#export').onclick=()=>{{let units={{}};document.querySelectorAll('select[data-unit]').forEach(x=>units[x.dataset.unit]={{preference:x.value,notes:document.querySelector('input[data-note="'+x.dataset.unit+'"]').value}});let data={{scope:'user development review',status,plan_sha256:{json.dumps(digest_json(plan))},labels_revealed:!document.querySelector('#labels').hidden,units}};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{{type:'application/json'}}));a.download='review-decisions.json';a.click();URL.revokeObjectURL(a.href)}};</script>'''
    (output / "index.html").write_text(page)
    manifest = {"scope": status, "plan_sha256": digest_json(plan), "config_sha256": config_hash,
                "seed": seed, "source_sha256": sha256(source) if source else None, "reference_max_rgb_deltas": deltas, "input_receipts": {r["profile"]: sha256(p / "receipt.json") for p, r in zip(folders, receipts)},
                "outputs": {p.name: sha256(p) for p in sorted(output.iterdir())}}
    encoder = shutil.which("ffmpeg")
    if encoder:
        version = subprocess.run([encoder, "-version"], capture_output=True, text=True, check=True, timeout=5).stdout.splitlines()[0]
        manifest["display_encoder"] = {"path": str(Path(encoder).resolve()), "sha256": sha256(Path(encoder)),
            "version": version,
            "video_color": "BT.709 limited-range YUV420 display proxy; not scored reconstruction"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--bind", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    parser.add_argument("--publish", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--slow-fps", type=int)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    build(args.config, args.config_sha256, args.bind, args.publish, args.output, args.seed, args.slow_fps, args.source)


if __name__ == "__main__":
    main()
