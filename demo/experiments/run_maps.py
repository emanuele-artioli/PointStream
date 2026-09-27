"""Run the maps-gallery extractors over demo clips and emit index.json.

One failed or skipped map does not abort the run. Payload kbps is the native
sidecar number; preview MP4/PNG sizes are recorded and never used as bitrate.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.evaluation.profile_map import gpu_name
from demo.pipeline.maps.contract import MAP_NAMES
from demo.pipeline.maps.model_paths import MISSING, MODELS, MODELS_ROOT

logger = logging.getLogger(__name__)

INDEX_SCHEMA = "pointstream.maps.index.v1"
DEFAULT_CLIPS = REPO_ROOT / "demo" / "outputs" / "pitch"
DEFAULT_OUT = REPO_ROOT / "demo" / "outputs" / "maps"
DEFAULT_MAPS = ("canny", "depth", "yoloe", "dino", "pose")
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SKIP_JSON_NAMES = {"skipped.json", "index.json", "classes.yaml", "bakeoff_pose.json"}

ALIAS_TO_FOLDER = {
    "canny": "canny",
    "depth": "depth",
    "yoloe": "yoloe_masks",
    "sam31": "sam31_masks",
    "dino": "dino_feat",
    "pose": "pose",
}

REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "depth": ("yolo26s_depth",),
    "yoloe": ("yoloe26_seg",),
    "sam31": (),
    "dino": ("dinov3_vits",),
    "pose": ("dwpose_pose", "dwpose_det"),
}

MAP_LABELS = {
    "canny": "Canny 240p",
    "canny_180": "Canny 180p",
    "canny_360": "Canny 360p",
    "canny_540": "Canny 540p",
    "canny_720": "Canny 720p",
    "canny_1080": "Canny 1080p",
    "depth": "Depth",
    "yoloe_masks": "YOLOE",
    "sam31_masks": "SAM 3.1",
    "dino_feat": "DINOv3 PCA",
    "dwpose": "DW-Pose",
    "dwpose_hands": "DW-Pose hands",
    "dwpose_face": "DW-Pose face",
    "dwpose_body": "DW-Pose body",
    "mediapipe_hands": "MediaPipe hands",
}


class MapSkip(Exception):
    """Backend weights or optional deps are missing — skip, do not download."""


def discover_clips(clips_arg: Path) -> list[Path]:
    if clips_arg.is_file():
        return [clips_arg]
    if not clips_arg.is_dir():
        raise FileNotFoundError(f"clips path not found: {clips_arg}")
    top = sorted(
        p for p in clips_arg.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    if top:
        return top
    nested = sorted(
        p for p in clips_arg.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    return nested


def _missing_keys(keys: Sequence[str]) -> list[str]:
    return [key for key in keys if MODELS.get(key) is None]


def _skip_reason(alias: str, keys: Sequence[str]) -> str:
    missing = _missing_keys(keys)
    return (
        f"map {alias!r} skipped: MODELS keys {missing} missing under {MODELS_ROOT}. "
        f"Known missing keys: {list(MISSING)}. Do not auto-download."
    )


def _run_canny(clip: Path, out: Path, max_frames: int | None, ctx: dict[str, Any]) -> None:
    from demo.pipeline.maps.canny import run_canny_clip

    run_canny_clip(clip, out, max_frames=max_frames)


def _run_depth(clip: Path, out: Path, max_frames: int | None, ctx: dict[str, Any]) -> None:
    from demo.pipeline.maps.depth import run_yolo26_depth

    run_yolo26_depth(clip, out, max_frames)


def _run_yoloe(clip: Path, out: Path, max_frames: int | None, ctx: dict[str, Any]) -> None:
    from demo.pipeline.maps.yoloe_masks import run_yoloe_clip

    run_yoloe_clip(clip, out, max_frames=max_frames)


def _run_sam31(clip: Path, out: Path, max_frames: int | None, ctx: dict[str, Any]) -> None:
    from demo.pipeline.maps.sam31_video import launch, resolve_checkpoint, resolve_python

    if resolve_checkpoint() is None or not Path(resolve_python()).is_file():
        raise MapSkip(
            "map 'sam31' skipped: sam3.1_multiplex.pt or the pointstream-sam31 "
            "python is missing. Do not auto-download."
        )
    launch(clip, out, max_frames=max_frames)


def _run_dino(clip: Path, out: Path, max_frames: int | None, ctx: dict[str, Any]) -> None:
    from demo.pipeline.maps import dinov3_features

    argv = ["--clip", str(clip), "--out", str(out)]
    if max_frames is not None:
        argv.extend(["--max-frames", str(max_frames)])
    code = dinov3_features.main(argv)
    if code == 2:
        raise MapSkip(
            f"map 'dino' skipped: dinov3_features exited 2. "
            f"Do not auto-download. Known missing keys: {list(MISSING)}."
        )
    if code != 0:
        raise RuntimeError(f"dinov3_features exited {code}")


def _run_pose(clip: Path, out: Path, max_frames: int | None, ctx: dict[str, Any]) -> None:
    from demo.pipeline.maps.dwpose import run_dwpose

    run_dwpose(clip, out, max_frames)


RUNNERS: dict[str, Callable[[Path, Path, int | None, dict[str, Any]], None]] = {
    "canny": _run_canny,
    "depth": _run_depth,
    "yoloe": _run_yoloe,
    "sam31": _run_sam31,
    "dino": _run_dino,
    "pose": _run_pose,
}


RGBA_SEQUENCE_MAPS = {"canny", "yoloe_masks", "sam31_masks", "dino_feat", "dwpose"}
POSE_PART_MAPS = {"dwpose_hands", "dwpose_face", "dwpose_body"}


def _is_rgba_sequence(map_name: str) -> bool:
    return str(map_name).startswith("canny") or str(map_name) in RGBA_SEQUENCE_MAPS


def _preview_kind(path: Path) -> str:
    if path.is_dir():
        return "image_dir"
    if path.suffix.lower() in {".mp4", ".webm"}:
        return "video"
    return "image"


def _first_preview_file(path: Path) -> Path | None:
    if path.is_file():
        return path
    if not path.is_dir():
        return None
    pngs = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    return pngs[0] if pngs else None


def _png_sequence_dir(preview: Path) -> Path | None:
    if preview.is_dir() and _first_preview_file(preview) is not None:
        return preview
    if preview.is_file() and preview.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        parent = preview.parent
        if _first_preview_file(parent) is not None:
            return parent
    return None


def _web_preview_file(preview: Path, *, map_name: str | None = None) -> Path | None:
    """RGBA maps stay PNG sequences so the inspector can stack transparent overlays."""
    if map_name is not None and _is_rgba_sequence(map_name):
        seq = _png_sequence_dir(preview)
        if seq is not None:
            return _first_preview_file(seq)
        return _first_preview_file(preview)
    candidates: list[Path] = []
    if preview.suffix.lower() in {".mp4", ".webm"}:
        candidates.append(preview)
    elif preview.suffix:
        candidates.append(preview.with_suffix(".mp4"))
    parent = preview if preview.is_dir() else preview.parent
    candidates.append(parent / "preview.mp4")
    candidates.append(parent / "preview.webm")
    if parent.name.startswith("preview"):
        candidates.append(parent.parent / "preview.mp4")
    seen: set[Path] = set()
    for cand in candidates:
        key = cand.resolve() if cand.exists() else cand
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file() and cand.suffix.lower() in {".mp4", ".webm"}:
            return cand
    return _first_preview_file(preview)


def _overlay_webm(seq_dir: Path | None, preview: Path) -> Path | None:
    candidates: list[Path] = []
    if preview:
        path = Path(preview)
        if path.suffix.lower() == ".webm":
            candidates.append(path)
        parent = path if path.is_dir() else path.parent
        candidates.append(parent / "preview.webm")
        if parent.name.startswith("preview"):
            candidates.append(parent.parent / "preview.webm")
    if seq_dir is not None:
        candidates.append(seq_dir.parent / "preview.webm")
        candidates.append(seq_dir / "preview.webm")
    seen: set[Path] = set()
    for cand in candidates:
        key = cand.resolve() if cand.exists() else cand
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return cand
    return None


def _rel_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def collect_sidecars(out_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    if not out_dir.is_dir():
        return docs
    for path in sorted(out_dir.rglob("*.json")):
        if path.name in SKIP_JSON_NAMES:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("map") not in MAP_NAMES:
            continue
        if "payload_bytes" not in data or "payload_kbps" not in data:
            continue
        data["_sidecar_path"] = str(path)
        docs.append(data)
    by_map: dict[str, dict[str, Any]] = {}
    for doc in docs:
        name = str(doc["map"])
        prev = by_map.get(name)
        prefer = Path(str(doc.get("_sidecar_path", ""))).name == "sidecar.json"
        if prev is None or prefer:
            by_map[name] = doc
    return list(by_map.values())


def canonicalize_sidecars(
    docs: Sequence[dict[str, Any]],
    *,
    clip_id: str,
    maps_root: Path,
) -> list[dict[str, Any]]:
    """Copy each extractor sidecar to outputs/maps/<clip>/<map>/sidecar.json."""
    written: list[dict[str, Any]] = []
    for doc in docs:
        dest_dir = maps_root / clip_id / str(doc["map"])
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "sidecar.json"
        payload = {k: v for k, v in doc.items() if not k.startswith("_")}
        dest.write_text(json.dumps(payload, indent=2) + "\n")
        payload["_sidecar_path"] = str(dest)
        written.append(payload)
    return written


def _entry_from_sidecar(doc: dict[str, Any], *, clip_id: str, maps_root: Path) -> dict[str, Any]:
    map_name = str(doc["map"])
    preview = Path(str(doc.get("preview_path") or ""))
    preview_file = (
        _web_preview_file(preview, map_name=map_name) if str(doc.get("preview_path") or "") else None
    )
    explicit_overlay = Path(str(doc.get("overlay_path") or ""))
    if explicit_overlay.is_file():
        seq_dir = None
        preview_file = explicit_overlay
        kind_src = explicit_overlay
        overlay = explicit_overlay
    else:
        seq_dir = _png_sequence_dir(preview) if _is_rgba_sequence(map_name) else None
        if seq_dir is not None:
            kind_src = seq_dir
            preview_file = _first_preview_file(seq_dir)
        else:
            kind_src = preview_file if preview_file is not None else preview
        overlay = _overlay_webm(seq_dir, preview)
    entry = {
        "clip": clip_id,
        "map": map_name,
        "label": MAP_LABELS.get(map_name, map_name),
        "backend": doc.get("backend"),
        "payload_kbps": doc["payload_kbps"],
        "payload_bytes": doc["payload_bytes"],
        "payload_path": doc.get("payload_path"),
        "payload_format": doc.get("payload_format"),
        "preview_bytes": doc.get("preview_bytes", 0),
        "preview_path": str(preview_file) if preview_file is not None else doc.get("preview_path"),
        "preview_kind": _preview_kind(kind_src) if kind_src.exists() else "image",
        "preview_url": _rel_to_root(preview_file, maps_root) if preview_file is not None else None,
        "preview_dir": _rel_to_root(seq_dir, maps_root) if seq_dir is not None else None,
        "extract_ms_p50": doc.get("extract_ms_p50"),
        "extract_ms_p95": doc.get("extract_ms_p95"),
        "pack_ms_p50": doc.get("pack_ms_p50"),
        "codec_ms_p50": doc.get("codec_ms_p50"),
        "decode_ms_p50": doc.get("decode_ms_p50"),
        "teleop_ok": doc.get("teleop_ok"),
        "gpu": doc.get("gpu"),
        "kind": doc.get("kind", "native"),
        "n_frames": doc.get("n_frames"),
        "fps": doc.get("fps"),
        "duration_s": doc.get("duration_s"),
        "status": "ok",
        "overlay_url": _rel_to_root(overlay, maps_root) if overlay is not None else None,
        "overlay_key": doc.get("overlay_key"),
    }
    return entry


def run_one_map(
    alias: str,
    clip: Path,
    *,
    clip_id: str,
    maps_root: Path,
    max_frames: int | None,
) -> dict[str, Any]:
    """Run one alias. Returns a result dict with status ok|skipped|failed."""
    if alias not in RUNNERS:
        return {
            "clip": clip_id,
            "map": alias,
            "status": "failed",
            "error": f"unknown map alias {alias!r}",
        }
    required = REQUIRED_KEYS.get(alias, ())
    if required and _missing_keys(required):
        return {
            "clip": clip_id,
            "map": alias,
            "status": "skipped",
            "reason": _skip_reason(alias, required),
        }

    folder = ALIAS_TO_FOLDER[alias]
    out = maps_root / clip_id / folder
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = {"clip_out": maps_root / clip_id, "maps_root": maps_root, "clip_id": clip_id}

    try:
        RUNNERS[alias](clip, out, max_frames, ctx)
    except MapSkip as exc:
        (out / "skipped.json").write_text(
            json.dumps({"map": alias, "status": "skipped", "reason": str(exc)}, indent=2) + "\n"
        )
        return {"clip": clip_id, "map": alias, "status": "skipped", "reason": str(exc)}
    except FileNotFoundError as exc:
        reason = str(exc)
        if "Do not auto-download" in reason or "MODELS[" in reason:
            return {"clip": clip_id, "map": alias, "status": "skipped", "reason": reason}
        return {"clip": clip_id, "map": alias, "status": "failed", "error": reason}
    except ImportError as exc:
        return {
            "clip": clip_id,
            "map": alias,
            "status": "skipped",
            "reason": f"{exc}. Do not auto-download weights.",
        }
    except Exception as exc:
        logger.exception("map %s failed on %s", alias, clip_id)
        return {
            "clip": clip_id,
            "map": alias,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }

    docs = collect_sidecars(out)
    if not docs:
        skipped = out / "skipped.json"
        if skipped.is_file():
            payload = json.loads(skipped.read_text())
            return {
                "clip": clip_id,
                "map": alias,
                "status": "skipped",
                "reason": payload.get("reason") or payload.get("error") or str(payload),
            }
        return {
            "clip": clip_id,
            "map": alias,
            "status": "failed",
            "error": f"no sidecar.json under {out}",
        }
    canonical = canonicalize_sidecars(docs, clip_id=clip_id, maps_root=maps_root)
    return {
        "clip": clip_id,
        "map": alias,
        "status": "ok",
        "sidecars": canonical,
    }


def build_index(
    *,
    clips: Sequence[Path],
    results: Sequence[dict[str, Any]],
    maps_root: Path,
) -> dict[str, Any]:
    by_clip: dict[str, dict[str, Any]] = {}
    for clip in clips:
        by_clip[clip.stem] = {"source": str(clip), "ok": {}}

    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for result in results:
        clip_id = result["clip"]
        status = result["status"]
        if status == "ok":
            for doc in result.get("sidecars") or []:
                if str(doc.get("map")) in POSE_PART_MAPS:
                    maps_on_clip = {
                        str(item.get("map"))
                        for item in (result.get("sidecars") or [])
                    }
                    if "dwpose" in maps_on_clip:
                        continue
                entry = _entry_from_sidecar(doc, clip_id=clip_id, maps_root=maps_root)
                entries.append(entry)
                by_clip.setdefault(clip_id, {"source": "", "ok": {}})
                by_clip[clip_id]["ok"][entry["map"]] = entry
        elif status == "skipped":
            skipped.append(
                {"clip": clip_id, "map": result["map"], "reason": result.get("reason")}
            )
        else:
            failed.append(
                {
                    "clip": clip_id,
                    "map": result["map"],
                    "error": result.get("error"),
                }
            )

    return {
        "schema": INDEX_SCHEMA,
        "gpu": gpu_name(),
        "models_root": str(MODELS_ROOT),
        "maps_root": str(maps_root),
        "caption": "Fused inspector: AV1/PS ladder plus stackable RGBA maps. Bitrate is native payload (Canny bitpack/sparse, YOLOE COCO-RLE, DW-Pose keypoints, DINOv3 int8). Preview PNGs are never counted.",
        "clips": by_clip,
        "entries": entries,
        "skipped": skipped,
        "failed": failed,
    }


def rebuild_index_from_disk(maps_root: Path) -> dict[str, Any]:
    """Rebuild index.json from existing sidecars without re-extracting."""
    maps_root = Path(maps_root)
    previous: dict[str, Any] = {}
    index_path = maps_root / "index.json"
    if index_path.is_file():
        try:
            previous = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            previous = {}
    prev_clips = previous.get("clips") or {}
    results: list[dict[str, Any]] = []
    clip_paths: list[Path] = []
    for clip_dir in sorted(p for p in maps_root.iterdir() if p.is_dir()):
        clip_id = clip_dir.name
        source = str((prev_clips.get(clip_id) or {}).get("source") or clip_dir)
        clip_paths.append(Path(source))
        docs: list[dict[str, Any]] = []
        for sidecar in sorted(clip_dir.rglob("*.json")):
            if sidecar.name in SKIP_JSON_NAMES:
                continue
            try:
                data = json.loads(sidecar.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("map") not in MAP_NAMES:
                continue
            if "payload_bytes" not in data or "payload_kbps" not in data:
                continue
            data["_sidecar_path"] = str(sidecar)
            docs.append(data)
        by_map: dict[str, dict[str, Any]] = {}
        for doc in docs:
            name = str(doc["map"])
            prev = by_map.get(name)
            prefer = Path(str(doc.get("_sidecar_path", ""))).name == "sidecar.json"
            if prev is None or prefer:
                by_map[name] = doc
        results.append(
            {
                "clip": clip_id,
                "map": "rebuild",
                "status": "ok",
                "sidecars": list(by_map.values()),
            }
        )
    index = build_index(clips=clip_paths, results=results, maps_root=maps_root)
    index["skipped"] = list(previous.get("skipped") or [])
    index["failed"] = list(previous.get("failed") or [])
    if previous.get("gpu"):
        index["gpu"] = previous["gpu"]
    if previous.get("models_root"):
        index["models_root"] = previous["models_root"]
    return index


def run_maps(
    *,
    clips: Sequence[Path],
    maps: Sequence[str],
    out: Path,
    max_frames: int | None = None,
) -> dict[str, Any]:
    maps_root = Path(out)
    maps_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for clip in clips:
        clip_id = clip.stem
        logger.info("clip %s (%s)", clip_id, clip)
        for alias in maps:
            logger.info("  map %s", alias)
            results.append(
                run_one_map(
                    alias,
                    clip,
                    clip_id=clip_id,
                    maps_root=maps_root,
                    max_frames=max_frames,
                )
            )
    index = build_index(clips=clips, results=results, maps_root=maps_root)
    index_path = maps_root / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    logger.info(
        "wrote %s  ok=%s skipped=%s failed=%s",
        index_path,
        len(index["entries"]),
        len(index["skipped"]),
        len(index["failed"]),
    )
    return index


def parse_maps(raw: str) -> list[str]:
    aliases = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [alias for alias in aliases if alias not in RUNNERS]
    if unknown:
        raise SystemExit(f"unknown map aliases {unknown}; expected one of {sorted(RUNNERS)}")
    return aliases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips",
        type=Path,
        default=DEFAULT_CLIPS,
        help="Directory of demo mp4s, or a single video (default: demo/outputs/pitch)",
    )
    parser.add_argument(
        "--maps",
        type=str,
        default=",".join(DEFAULT_MAPS),
        help="Comma-separated aliases: canny,depth,yoloe,sam31,dino,pose",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        clips = discover_clips(args.clips)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not clips:
        print(f"no video clips under {args.clips}", file=sys.stderr)
        return 2

    maps = parse_maps(args.maps)
    run_maps(clips=clips, maps=maps, out=args.out, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
