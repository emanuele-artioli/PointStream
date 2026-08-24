"""Load a real 4K point scene and check that crops paste back onto source pixels.

This dataset has two frame-id conventions. Crops are paired with metadata by
position in file order — a filename is never rebuilt from an id. The frame
each convention names is then decoded, and the convention whose opaque pixels
match the source is the one the clip is built from. If neither matches, stop.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from src.components.codec.tools import resolve_ffmpeg

REPO = Path(__file__).resolve().parents[2]
RAW_4K = REPO / "assets" / "raw_4k"
DATASET = REPO / "assets" / "dataset"
DERIVED_SUFFIXES = ("_skeleton", "_canny", "_pose_body", "_pose_racket")
OPAQUE = 128
PASTE_MAE_MAX = 2.0


class PasteBackError(RuntimeError):
    """Crops do not reproduce source pixels under either frame convention."""


@dataclass
class TrackPair:
    frame_id: int
    bbox: tuple[int, int, int, int]
    crop_path: Path
    position: int


@dataclass
class SceneClip:
    video: str
    scene: str
    video_path: Path
    t_start: float
    t_end: float
    cluster: str
    convention: str
    window_start: int
    n_frames: int
    frames: np.ndarray
    masks: np.ndarray
    player_area: float
    paste_back: dict[str, Any]
    ffmpeg: dict[str, str]


def _ffmpeg() -> tuple[str, str]:
    tool = resolve_ffmpeg()
    return tool.path, tool.version


def load_rgba(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.ndim != 3:
        raise ValueError(f"{path} is not an RGB(A) crop")
    if raw.shape[2] == 4:
        blue, green, red, alpha = cv2.split(raw)
        return cv2.merge((red, green, blue, alpha))
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    alpha = np.where(np.any(rgb > 8, axis=2), 255, 0).astype(np.uint8)
    return np.dstack((rgb, alpha))


def bbox_slices(
    bbox: tuple[int, int, int, int],
    crop_h: int,
    crop_w: int,
    frame_h: int,
    frame_w: int,
) -> tuple[slice, slice]:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (y2 - y1, x2 - x1) == (crop_h, crop_w):
        rows, cols = slice(y1, y2), slice(x1, x2)
    elif (y2 - y1 + 1, x2 - x1 + 1) == (crop_h, crop_w):
        rows, cols = slice(y1, y2 + 1), slice(x1, x2 + 1)
    else:
        rows, cols = slice(y1, y1 + crop_h), slice(x1, x1 + crop_w)
    if rows.start < 0 or cols.start < 0 or rows.stop > frame_h or cols.stop > frame_w:
        raise ValueError(
            f"bbox {bbox} crop {(crop_h, crop_w)} does not sit in {(frame_h, frame_w)}"
        )
    return rows, cols


def opaque_mae(
    frame_rgb: np.ndarray, crop_rgba: np.ndarray, rows: slice, cols: slice
) -> float:
    region = frame_rgb[rows, cols]
    rgb = crop_rgba[..., :3]
    alpha = crop_rgba[..., 3] >= OPAQUE
    if region.shape[:2] != rgb.shape[:2] or not np.any(alpha):
        return 255.0
    diff = np.abs(region.astype(np.int16) - rgb.astype(np.int16))
    return float(diff[alpha].mean())


def paste_crop(
    frame_rgb: np.ndarray, crop_rgba: np.ndarray, rows: slice, cols: slice
) -> np.ndarray:
    out = frame_rgb.copy()
    alpha = crop_rgba[..., 3] >= OPAQUE
    patch = out[rows, cols]
    patch[alpha] = crop_rgba[..., :3][alpha]
    out[rows, cols] = patch
    return out


def pair_track(scene_dir: Path, track_dir: Path) -> list[TrackPair]:
    """Positional pairing: sorted crops with metadata rows, never by rebuilt name."""
    meta_path = scene_dir / f"{track_dir.name}_metadata.json"
    if not meta_path.is_file():
        return []
    records = json.loads(meta_path.read_text())
    if isinstance(records, dict):
        records = records.get("frames") or records.get("entries") or []
    crops = sorted(track_dir.glob("frame_*.png"))
    if len(crops) != len(records):
        raise ValueError(
            f"{track_dir.name}: {len(crops)} crops vs {len(records)} metadata rows; "
            "refusing to zip-truncate"
        )
    pairs: list[TrackPair] = []
    for position, (record, crop) in enumerate(zip(records, crops)):
        if not isinstance(record, dict):
            continue
        frame_id = record.get("frame_id", record.get("frame_id"))
        bbox = record.get("bbox", record.get("bbox"))
        if frame_id is None or bbox is None:
            continue
        box = tuple(int(v) for v in bbox)
        if len(box) != 4:
            raise ValueError(f"{meta_path} row {position} bbox {bbox}")
        pairs.append(
            TrackPair(
                frame_id=int(frame_id),
                bbox=(box[0], box[1], box[2], box[3]),
                crop_path=crop,
                position=position,
            )
        )
    return pairs


def list_tracks(scene_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in scene_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("track_")
        and not path.name.endswith(DERIVED_SUFFIXES)
    )


def point_scenes(video: str) -> list[dict[str, Any]]:
    meta_path = DATASET / video / "scene_metadata.json"
    if not meta_path.is_file():
        return []
    payload = json.loads(meta_path.read_text())
    scenes = payload.get("scenes") or payload.get("segments") or []
    found: list[dict[str, Any]] = []
    for index, record in enumerate(scenes):
        if not isinstance(record, dict):
            continue
        cluster = str(record.get("cluster") or "")
        if "point" not in cluster.lower() or "interlude" in cluster.lower():
            continue
        folder = DATASET / video / "segmentations" / f"scene_{index:03d}"
        if not folder.is_dir():
            continue
        t_start = float(record.get("t_start", record.get("start", 0.0)))
        t_end = float(record.get("t_end", record.get("end", t_start)))
        duration = float(record.get("duration") or (t_end - t_start))
        if duration < 48 / 24 or duration > 30:
            continue
        if not list_tracks(folder):
            continue
        found.append(
            {
                "video": video,
                "scene": f"scene_{index:03d}",
                "index": index,
                "t_start": t_start,
                "t_end": t_end,
                "duration": duration,
                "cluster": cluster,
                "folder": folder,
            }
        )
    return found


def _run_ffmpeg(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def extract_24fps_pngs(
    video_path: Path,
    t_start: float,
    duration: float,
    out_dir: Path,
    *,
    ffmpeg: str,
) -> list[Path]:
    """Same recipe as scripts/process_dataset.py: -ss before -i, -r 24."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for leftover in out_dir.glob("frame_*.png"):
        leftover.unlink()
    argv = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{t_start:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.6f}",
        "-r",
        "24",
        str(out_dir / "frame_%06d.png"),
    ]
    _run_ffmpeg(argv)
    files = sorted(out_dir.glob("frame_*.png"))
    if not files:
        raise RuntimeError(f"ffmpeg wrote no frames: {argv}")
    return files


def _ffprobe(ffmpeg: str) -> str:
    return str(Path(ffmpeg).with_name("ffprobe"))


def probe_fps(video_path: Path, *, ffprobe: str) -> float:
    argv = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    raw = subprocess.check_output(argv, text=True).strip()
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        return float(numerator) / float(denominator)
    return float(raw)


def extract_24fps_frame(
    video_path: Path,
    t_start: float,
    index: int,
    dest: Path,
    *,
    ffmpeg: str,
) -> np.ndarray:
    """One 24 fps extract frame, same -ss-before--i recipe as process_dataset."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{t_start + index / 24.0:.6f}",
        "-i",
        str(video_path),
        "-r",
        "24",
        "-frames:v",
        "1",
        str(dest),
    ]
    _run_ffmpeg(argv)
    return _read_rgb(dest)


def extract_native_frame(
    video_path: Path,
    frame_id: int,
    dest: Path,
    *,
    ffmpeg: str,
    fps: float,
) -> np.ndarray:
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{frame_id / fps:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(dest),
    ]
    _run_ffmpeg(argv)
    image = cv2.imread(str(dest), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"native decode produced nothing for frame_id={frame_id}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _mae_for_pair(frame_rgb: np.ndarray, pair: TrackPair) -> dict[str, Any]:
    crop = load_rgba(pair.crop_path)
    rows, cols = bbox_slices(
        pair.bbox, crop.shape[0], crop.shape[1], frame_rgb.shape[0], frame_rgb.shape[1]
    )
    return {
        "frame_id": pair.frame_id,
        "position": pair.position,
        "bbox": list(pair.bbox),
        "crop": str(pair.crop_path),
        "crop_hw": [int(crop.shape[0]), int(crop.shape[1])],
        "mae": opaque_mae(frame_rgb, crop, rows, cols),
    }


def diagnose_convention(
    video_path: Path,
    scene: dict[str, Any],
    pairs: list[TrackPair],
    work_dir: Path,
    *,
    ffmpeg: str,
    pngs_24: list[Path] | None = None,
    n_samples: int = 3,
) -> dict[str, Any]:
    if not pairs:
        raise PasteBackError(f"{scene['scene']}: no track pairs")
    samples = [pairs[0], pairs[len(pairs) // 2], pairs[-1]][:n_samples]
    work_dir.mkdir(parents=True, exist_ok=True)
    fps = probe_fps(video_path, ffprobe=_ffprobe(ffmpeg))
    methods: dict[str, list[dict[str, Any]]] = {
        "extract_24_position": [],
        "extract_24_frame_id": [],
        "native_frame_id": [],
    }

    def mae_24_at(index: int, pair: TrackPair) -> dict[str, Any] | None:
        if pngs_24 is None or index < 0 or index >= len(pngs_24):
            return None
        return _mae_for_pair(_read_rgb(pngs_24[index]), pair)

    for pair in samples:
        at_position = mae_24_at(pair.position, pair)
        if at_position is not None:
            methods["extract_24_position"].append(at_position)
        at_id = mae_24_at(pair.frame_id, pair)
        if at_id is not None:
            methods["extract_24_frame_id"].append(at_id)
        native_path = work_dir / f"native_{pair.frame_id}.png"
        try:
            native = extract_native_frame(
                video_path, pair.frame_id, native_path, ffmpeg=ffmpeg, fps=fps
            )
            methods["native_frame_id"].append(_mae_for_pair(native, pair))
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            methods["native_frame_id"].append(
                {"frame_id": pair.frame_id, "error": str(exc)}
            )

    def mean_mae(rows: list[dict[str, Any]]) -> float | None:
        values = [float(row["mae"]) for row in rows if "mae" in row]
        if not values:
            return None
        return float(sum(values) / len(values))

    scores = {name: mean_mae(rows) for name, rows in methods.items()}
    winner_name: str | None = None
    winner_mae: float | None = None
    for name, mae in scores.items():
        if mae is None:
            continue
        if winner_mae is None or mae < winner_mae:
            winner_name, winner_mae = name, mae
    result: dict[str, Any] = {
        "samples": methods,
        "mean_mae": scores,
        "winner": winner_name,
        "winner_mae": winner_mae,
        "native_fps": fps,
        "n_extract_24": 0 if pngs_24 is None else len(pngs_24),
        "threshold": PASTE_MAE_MAX,
    }
    dump = work_dir / "diagnosis.json"
    dump.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        f"paste-back {scene['video']}/{scene['scene']} "
        f"winner={winner_name} mae={winner_mae} scores={scores}",
        flush=True,
    )
    if winner_name is None or winner_mae is None or winner_mae > PASTE_MAE_MAX:
        raise PasteBackError(
            f"{scene['video']}/{scene['scene']}: paste-back MAE {winner_mae} "
            f"(winner={winner_name}) exceeds {PASTE_MAE_MAX}. scores={scores}"
        )
    return result


def area_by_id(track_pairs: list[list[TrackPair]]) -> dict[int, int]:
    areas: dict[int, int] = {}
    for pairs in track_pairs:
        for pair in pairs:
            crop = load_rgba(pair.crop_path)
            areas[pair.frame_id] = areas.get(pair.frame_id, 0) + int(
                (crop[..., 3] >= OPAQUE).sum()
            )
    return areas


def best_window(areas: dict[int, int], n_frames: int) -> tuple[int, int]:
    if not areas:
        raise PasteBackError("no player pixels in any track")
    lo, hi = min(areas), max(areas)
    if hi - lo + 1 < n_frames:
        return lo, int(sum(areas.values()))
    best_start, best_sum = lo, -1
    for start in range(lo, hi - n_frames + 2):
        total = sum(areas.get(index, 0) for index in range(start, start + n_frames))
        if total > best_sum:
            best_start, best_sum = start, total
    return best_start, best_sum


def build_masks(
    track_pairs: list[list[TrackPair]],
    frame_ids: list[int],
    height: int,
    width: int,
) -> np.ndarray:
    masks = np.zeros((len(frame_ids), height, width), dtype=bool)
    index_of = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    for pairs in track_pairs:
        for pair in pairs:
            slot = index_of.get(pair.frame_id)
            if slot is None:
                continue
            crop = load_rgba(pair.crop_path)
            rows, cols = bbox_slices(
                pair.bbox, crop.shape[0], crop.shape[1], height, width
            )
            masks[slot, rows, cols] |= crop[..., 3] >= OPAQUE
    return masks


def load_rgb_stack(paths: list[Path]) -> np.ndarray:
    return np.stack([_read_rgb(path) for path in paths], axis=0)


def choose_two_point_scenes() -> list[dict[str, Any]]:
    preferred = (
        "alcaraz_highlights",
        "federer_djokovic",
        "sinner_alcaraz",
        "alcaraz_perricard",
    )
    videos = [name for name in preferred if (DATASET / name).is_dir()]
    videos.extend(
        sorted(
            path.name
            for path in DATASET.iterdir()
            if path.is_dir() and path.name not in videos
        )
    )
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for video in videos:
        if video in seen:
            continue
        scenes = point_scenes(video)
        if not scenes:
            continue
        chosen.append(scenes[0])
        seen.add(video)
        if len(chosen) >= 2:
            break
    if len(chosen) < 2:
        raise PasteBackError(f"need two point scenes; found {len(chosen)}")
    return chosen


def load_scene_clip(
    scene: dict[str, Any],
    work_dir: Path,
    *,
    n_frames: int = 48,
) -> SceneClip:
    ffmpeg_path, ffmpeg_version = _ffmpeg()
    video_path = RAW_4K / f"{scene['video']}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    scene_dir = Path(scene["folder"])
    track_pairs = [pair_track(scene_dir, track) for track in list_tracks(scene_dir)]
    track_pairs = [pairs for pairs in track_pairs if pairs]
    if not track_pairs:
        raise PasteBackError(f"{scene['video']}/{scene['scene']}: no usable tracks")
    flat = [pair for pairs in track_pairs for pair in pairs]
    print(
        f"extract 24fps {scene['video']}/{scene['scene']} "
        f"duration={scene['duration']:.2f}s",
        flush=True,
    )
    pngs_24 = extract_24fps_pngs(
        video_path,
        float(scene["t_start"]),
        float(scene["duration"]),
        work_dir / "extract_24",
        ffmpeg=ffmpeg_path,
    )
    diagnosis = diagnose_convention(
        video_path,
        scene,
        flat,
        work_dir / "paste_back",
        ffmpeg=ffmpeg_path,
        pngs_24=pngs_24,
    )
    convention = str(diagnosis["winner"])
    areas = area_by_id(track_pairs)
    window_start, _total = best_window(areas, n_frames)
    if convention.startswith("extract_24"):
        if window_start + n_frames > len(pngs_24):
            window_start = max(0, len(pngs_24) - n_frames)
        ids = list(range(window_start, window_start + n_frames))
        frames = load_rgb_stack(pngs_24[window_start : window_start + n_frames])
    else:
        ids = list(range(window_start, window_start + n_frames))
        fps = float(diagnosis["native_fps"])
        native_dir = work_dir / "native_window"
        native_pngs: list[Path] = []
        for frame_id in ids:
            dest = native_dir / f"frame_{frame_id:06d}.png"
            extract_native_frame(
                video_path, frame_id, dest, ffmpeg=ffmpeg_path, fps=fps
            )
            native_pngs.append(dest)
        frames = load_rgb_stack(native_pngs)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    masks = build_masks(track_pairs, ids, height, width)
    window_mae: list[float] = []
    for local, frame_id in enumerate(ids[:3]):
        for pairs in track_pairs:
            hits = [pair for pair in pairs if pair.frame_id == frame_id]
            if not hits:
                continue
            window_mae.append(float(_mae_for_pair(frames[local], hits[0])["mae"]))
            break
    diagnosis["window_mae"] = window_mae
    diagnosis["window_start"] = window_start
    (work_dir / "paste_back" / "diagnosis.json").write_text(
        __import__("json").dumps(diagnosis, indent=2, default=str) + "\n"
    )
    if window_mae and (sum(window_mae) / len(window_mae)) > PASTE_MAE_MAX:
        raise PasteBackError(
            f"{scene['video']}/{scene['scene']} window paste-back MAE "
            f"{sum(window_mae) / len(window_mae):.3f} exceeds {PASTE_MAE_MAX} "
            f"(winner={convention} sample_mae={diagnosis.get('winner_mae')})"
        )
    return SceneClip(
        video=str(scene["video"]),
        scene=str(scene["scene"]),
        video_path=video_path,
        t_start=float(scene["t_start"]),
        t_end=float(scene["t_end"]),
        cluster=str(scene["cluster"]),
        convention=convention,
        window_start=window_start,
        n_frames=int(frames.shape[0]),
        frames=frames,
        masks=masks,
        player_area=float(masks.mean()),
        paste_back=diagnosis,
        ffmpeg={"path": ffmpeg_path, "version": ffmpeg_version},
    )
