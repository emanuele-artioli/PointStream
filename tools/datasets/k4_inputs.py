"""Windows, staged inputs and fleet specifications for K4 and K5 (docs/experiments.md, 2026-10-13 K4, K5).

    python3 tools/datasets/k4_inputs.py --datasets DIR --dest DIR --environment TAR SHA256 --sam PT SHA256 \\
        --dcvc-image PTH SHA256 --dcvc-video PTH SHA256 --inspect BIN SHA256 --deadline ISO

Selection (`RULE`, from metadata only: clip names and frame counts): per broadcast match the
`BROADCAST_PER_MATCH` clips of at least `MIN_FRAMES` frames with the smallest
sha256("pointstream-k4:broadcast:<match>/<clip>"); the `TRACKNET_CLIPS` TrackNet clips of at least
`TRACKNET_MIN_FRAMES` frames with the smallest sha256("pointstream-k4:tracknet:<game>/<clip>"); each window its
first `MAX_FRAMES` frames. The archived clips, the whole match for the camera measure and the YOLO weights are
hard-linked under DEST/inputs (same filesystem; staged inputs must lie under the data root). The selected TrackNet
clips' JPEGs are packed into one tar (each member checked by the zip's CRC); their labels go into ``windows.json``.
Writes ``windows.json``, the content-addressed ``inputs-<sha12>.json`` and the specs ``k4-pilot.json``,
``k4-full-<k>.json``, ``k4-camera.json``, ``k5-pilot.json``, ``k5-full-<k>.json``. Standard library and ffprobe only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any

SEED = "pointstream-k4"
ARCHIVE = "archive/pre-reset-2026-10-05/tennis_games"
#: Matches with scene clips at 1080p or above, and where their clips are.
MATCHES = {"alcaraz_hurkacz": ".", "medvedev_struff": ".", "rublev_auger-aliassime": "videos",
           "djokovic_federer": "videos", "djokovic_zverev": "."}
RAW_MATCH = "raw_1080p/nadal_kyrgios.mp4"
BROADCAST_PER_MATCH = 2
MIN_FRAMES = 240
MAX_FRAMES = 240
TRACKNET_CLIPS = 8
TRACKNET_MIN_FRAMES = 120
TRACKNET_FPS = 30
YOLO = ("yolo26x-pose.pt", "yolo11x-pose.pt", "yolo26x.pt", "yolo11x.pt")
RULE = {
    "broadcast": f"per match of {sorted(MATCHES)}, the {BROADCAST_PER_MATCH} scene clips of at least {MIN_FRAMES} frames "
                 f"with the smallest sha256('{SEED}:broadcast:<match>/<clip>'); first {MAX_FRAMES} frames at 1920x1080 "
                 "(2160p downscaled, area)",
    "tracknet": f"the {TRACKNET_CLIPS} clips of at least {TRACKNET_MIN_FRAMES} frames with the smallest "
                f"sha256('{SEED}:tracknet:<game>/<clip>'); first <= {MAX_FRAMES} frames at 1280x720, {TRACKNET_FPS} frames/s",
    "camera": f"the whole of {RAW_MATCH}, every 0.5 s",
}
HOSTS = ["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"]
GPU_MODELS = ["RTX 6000 Ada", "RTX A6000"]
SHARING = {"mode": "shared", "basis": "SAM 3.1's masks, YOLO's detections, the SVT-AV1 and DCVC-UF streams and their "
           "bit accounting and scores do not depend on wall time; recorded seconds are not evidence"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def rank(key: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest(), 16)


def probe(path: Path) -> dict[str, Any]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": int(num) / int(den), "frames": int(s["nb_read_packets"])}


def link(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return target
        raise SystemExit(f"{target} exists and is not {source}")
    os.link(source, target)
    return target


def entry(name: str, path: Path, sha: str | None = None, extract: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "path": str(path), "sha256": sha or file_sha256(path)}
    return {**out, "extract": True} if extract else out


def safe(text: str) -> str:
    return text.replace("/", "_").replace("-", "_")


def tracknet(zip_path: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        jpgs: dict[str, list[str]] = {}
        for name in names:
            if name.endswith(".jpg"):
                jpgs.setdefault(name.rsplit("/", 1)[0], []).append(name)
        keys = sorted((d for d in jpgs if len(jpgs[d]) >= TRACKNET_MIN_FRAMES), key=lambda d: rank(f"tracknet:{d[len('Dataset/'):]}"))
        windows = []
        for member_dir in keys[:TRACKNET_CLIPS]:
            key = member_dir[len("Dataset/"):]
            files = sorted(jpgs[member_dir])
            if [Path(f).name for f in files] != [f"{i:04d}.jpg" for i in range(len(files))]:
                raise SystemExit(f"{key}: JPEG names are not 0000.jpg onwards")
            rows = list(csv.DictReader(io.TextIOWrapper(archive.open(f"{member_dir}/Label.csv"), "utf-8")))
            by_name = {r["file name"]: r for r in rows}
            frames = min(MAX_FRAMES, len(files))
            labels = []
            for f in files[:frames]:
                r = by_name[Path(f).name]
                vis = int(r["visibility"])
                labels.append([vis, int(float(r["x-coordinate"] or 0)), int(float(r["y-coordinate"] or 0))])
            jpgs[member_dir] = files[:frames]
            windows.append({"id": f"tracknet/{key}", "dataset": "tracknet", "size": [1280, 720], "fps": TRACKNET_FPS,
                            "frames": frames, "source_frames": len(files), "labels": labels,
                            "source": {"kind": "jpeg_dir", "staged": "tracknet", "member_dir": member_dir,
                                       "pattern": f"{member_dir}/%04d.jpg"}})
    return windows, {w["source"]["member_dir"]: jpgs[w["source"]["member_dir"]] for w in windows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True, help="the Datasets root")
    parser.add_argument("--models", required=True, help="the Models root")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--environment", nargs=2, required=True, metavar=("TAR", "SHA256"))
    parser.add_argument("--sam", nargs=2, required=True, metavar=("PT", "SHA256"))
    parser.add_argument("--dcvc-image", nargs=2, required=True, metavar=("PTH", "SHA256"))
    parser.add_argument("--dcvc-video", nargs=2, required=True, metavar=("PTH", "SHA256"))
    parser.add_argument("--inspect", nargs=2, required=True, metavar=("BIN", "SHA256"))
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    root, dest = Path(args.datasets), Path(args.dest).resolve()
    staged: dict[str, dict[str, Any]] = {}
    broadcast = []
    for match, sub in MATCHES.items():
        folder = root / ARCHIVE / match / sub
        clips = {p.stem: probe(p) for p in sorted(folder.glob("*.mp4"))}
        eligible = sorted((c for c, info in clips.items() if info["frames"] >= MIN_FRAMES), key=lambda c: rank(f"broadcast:{match}/{c}"))
        for clip in eligible[:BROADCAST_PER_MATCH]:
            name = f"clip_{safe(match)}_{clip}"
            staged[name] = entry(name, link(folder / f"{clip}.mp4", dest / "inputs" / "clips" / match / f"{clip}.mp4"))
            info = clips[clip]
            broadcast.append({"id": f"broadcast/{match}/{clip}", "dataset": "broadcast", "size": [1920, 1080], "fps": info["fps"],
                              "frames": MAX_FRAMES, "source_frames": info["frames"], "source_size": [info["width"], info["height"]],
                              "source": {"kind": "video", "staged": name}})
    broadcast.sort(key=lambda w: rank(f"order:{w['id']}"))
    tn, members = tracknet(root / "TrackNet" / "Dataset.zip")
    tar_path = dest / "inputs" / "tracknet-k4.tar"
    if not tar_path.exists():
        tmp = tar_path.with_suffix(".tmp")
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(root / "TrackNet" / "Dataset.zip") as archive, tarfile.open(tmp, "w") as out:
            for files in members.values():
                for name in files:
                    data = archive.read(name)  # zipfile checks each member's CRC-32
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    out.addfile(info, io.BytesIO(data))
        tmp.rename(tar_path)
    staged["tracknet"] = entry("tracknet", tar_path, extract=True)
    raw = probe(root / ARCHIVE / RAW_MATCH)
    staged["raw_match"] = entry("raw_match", link(root / ARCHIVE / RAW_MATCH, dest / "inputs" / "clips" / "nadal_kyrgios.mp4"))
    for weight in YOLO:
        name = safe(weight.removesuffix(".pt"))
        staged[name] = entry(name, link(Path(args.models) / "YOLO" / weight, dest / "inputs" / "yolo" / weight))
    for name, (path, sha) in {"sam": args.sam, "dcvc_image": args.dcvc_image, "dcvc_video": args.dcvc_video,
                              "inspect": args.inspect}.items():
        staged[name] = entry(name, Path(path), sha)
    windows = {"rule": RULE, "seed": SEED, "windows": tn + broadcast,
               "camera": {"id": "match/nadal_kyrgios", "staged": "raw_match", **raw}}
    windows_path = dest / "windows.json"
    windows_path.write_text(json.dumps(windows, indent=1) + "\n")
    staged["windows"] = entry("windows", windows_path)
    text = json.dumps({"files": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in sorted(staged.items())},
                       "rule": RULE}, indent=1, sort_keys=True) + "\n"
    inputs = dest / f"inputs-{hashlib.sha256(text.encode()).hexdigest()[:12]}.json"
    if not inputs.exists():
        inputs.write_text(text)

    def spec(module: str, arguments: list[str], names: list[str], scale: dict[str, Any], basis: str, memory: int) -> dict[str, Any]:
        return {
            "schema": 1, "entrypoint": ["-m", module], "environment": {"path": args.environment[0], "sha256": args.environment[1]},
            "inputs": [{"path": str(inputs), "sha256": file_sha256(inputs)}], "arguments": arguments,
            "staged_inputs": [staged[k] for k in names], "scale": scale, "smoke": {"seconds": 600, "representative_basis": basis},
            "full": {"seconds": 2700}, "validator": ["{python}", "-m", module, "validate"], "validator_seconds": 300,
            "required_commands": [], "budget_seconds": 4500, "deadline": args.deadline, "stall_seconds": 1800,
            "cpu_threads": 16, "device": "gpu", "hosts": HOSTS, "gpu_models": GPU_MODELS, "gpu_memory_mib": memory,
            "local_storage_gib": 80, "gpu_sharing": SHARING,
        }

    def sources(ws: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        names = sorted({w["source"]["staged"] for w in ws})
        return [a for n in names for a in ("--source", n, f"{{staged:{n}}}")], names

    def k4_spec(ws: list[dict[str, Any]], what: str) -> dict[str, Any]:
        src_args, src_names = sources(ws)
        smoke = ws[0]
        arguments = ["run", "--windows", "{staged:windows}", "--select", "{windows}", "--frames", "{frames}", *src_args,
                     "--sam-checkpoint", "{staged:sam}", "--pose", "yolo26x-pose", "{staged:yolo26x_pose}",
                     "--pose", "yolo11x-pose", "{staged:yolo11x_pose}", "--inspect", "{staged:inspect}"]
        basis = (f"K4 on {smoke['id']}: its first 48 frames through the full path (ffmpeg to 4:2:0, SAM 3.1's three text "
                 "prompts, both YOLO pose models, the parts, SVT-AV1 at all four CRFs with inspect's bit accounting, the "
                 f"camera measure, review frames); the stage runs {what} at {MAX_FRAMES} frames.")
        return spec("experiments.tennis.k4", arguments, ["windows", "sam", "yolo26x_pose", "yolo11x_pose", "inspect", *src_names],
                    {"windows": {"smoke": smoke["id"], "full": ",".join(w["id"] for w in ws)}, "frames": {"smoke": 48, "full": MAX_FRAMES}},
                    basis, 32000)

    def k5_spec(ws: list[dict[str, Any]], what: str) -> dict[str, Any]:
        src_args, src_names = sources(ws)
        arguments = ["run", "--windows", "{staged:windows}", "--select", "{windows}", "--frames", "{frames}", *src_args,
                     "--image-ckpt", "{staged:dcvc_image}", "--image-sha256", args.dcvc_image[1],
                     "--video-ckpt", "{staged:dcvc_video}", "--video-sha256", args.dcvc_video[1],
                     "--detector", "yolo26x", "{staged:yolo26x}", "--detector", "yolo11x", "{staged:yolo11x}"]
        basis = (f"K5 on {ws[0]['id']}: its first 48 frames through the full path (ffmpeg to 4:2:0, SVT-AV1 at the five CRFs "
                 "and DCVC-UF HT-L at the four QPs with decodes from bytes, the oracle composites, both detectors on the "
                 f"source, decodes and composites, ball-region and frame PSNR); the stage runs {what} at {MAX_FRAMES} frames.")
        return spec("experiments.tennis.k5", arguments, ["windows", "dcvc_image", "dcvc_video", "yolo26x", "yolo11x", *src_names],
                    {"windows": {"smoke": ws[0]["id"], "full": ",".join(w["id"] for w in ws)}, "frames": {"smoke": 48, "full": MAX_FRAMES}},
                    basis, 16000)

    specs: dict[str, dict[str, Any]] = {}
    pilot = tn[:2] + broadcast[:2]
    specs["k4-pilot"] = k4_spec(pilot, "2 TrackNet and 2 broadcast windows")
    rest = [w for w in tn[2:] + broadcast[2:]]
    rest.sort(key=lambda w: rank(f"part:{w['id']}"))
    for k in range(3):
        part = rest[k::3]
        specs[f"k4-full-{k + 1}"] = k4_spec(part, f"part {k + 1} ({len(part)} windows)")
    specs["k4-camera"] = spec("experiments.tennis.k4", ["camera", "--video", "{staged:raw_match}", "--fps", str(raw["fps"]),
                                                        "--limit-seconds", "{seconds}", "--pose", "{staged:yolo26x_pose}"],
                              ["raw_match", "yolo26x_pose"], {"seconds": {"smoke": 60, "full": 0}},
                              "K4 camera on the first 60 s of the match through the full path (ffmpeg decode every 0.5 s, "
                              "SIFT/MAGSAC pairs, YOLO26x-pose people); the stage runs the whole 2,310 s.", 8000)
    specs["k5-pilot"] = k5_spec(tn[:2], "the first 2 TrackNet windows")
    for k in range(2):
        part = tn[2 + 3 * k: 5 + 3 * k]
        specs[f"k5-full-{k + 1}"] = k5_spec(part, f"part {k + 1} ({len(part)} windows)")
    for name, body in specs.items():
        (dest / f"{name}.json").write_text(json.dumps(body, indent=1) + "\n")
    print(json.dumps({"inputs": str(inputs), "windows": [w["id"] for w in tn + broadcast], "specs": sorted(specs)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
