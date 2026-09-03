"""Extraction, feature measurement, and interval validation for BP46 long scenes.

Extracts 24 fps frames, pairs object tracks positionally, checks paste-back MAE,
computes background motion statistics and panorama canvas growth, and validates
intervals at exact frame counts 48, 96, 192, 384 frames.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.headroom.real import (
    PASTE_MAE_MAX,
    TrackPair,
    bbox_slices,
    diagnose_convention,
    extract_24fps_pngs,
    list_tracks,
    load_rgba,
    pair_track,
)
from experiments.long_scenes.schema import (
    MAX_CANVAS_GROWTH,
    MAX_CONSECUTIVE_MAD,
    TARGET_SPANS,
    CameraMotionFeatures,
    EligibilityFeatures,
    IntervalValidation,
    ObjectFeatures,
    PanoramaFeatures,
    PasteBackFeatures,
    SceneRecord,
    SourceMetadata,
)
from src.components.background.plate import _canvas, estimate_homographies
from src.components.codec.tools import resolve_ffmpeg
from src.contracts import paths as ps_paths

BP46_CLIPS_ROOT = ps_paths.outputs() / "bp46-long-scenes" / "clips"
BP21_CLIPS_ROOT = ps_paths.outputs() / "bp21-headroom" / "clips"
RAW_4K_ROOT = ps_paths.assets() / "raw_4k"
DATASET_ROOT = ps_paths.assets() / "dataset"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        # Read 64KB chunks
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_image(image: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(image.tobytes())
    return h.hexdigest()


def probe_video_source(video_path: Path, *, ffprobe: str) -> SourceMetadata:
    """Extract stream & container properties of raw 4K footage."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,pix_fmt,color_space,color_transfer,color_primaries",
        "-of",
        "json",
        str(video_path),
    ]
    res = subprocess.check_output(cmd, text=True)
    meta = json.loads(res)
    streams = meta.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video streams in {video_path}")
    v = streams[0]
    r_fps_raw = str(v.get("r_frame_rate", "24/1"))
    if "/" in r_fps_raw:
        num, den = r_fps_raw.split("/", 1)
        source_fps = float(num) / float(den)
    else:
        source_fps = float(r_fps_raw)

    # Compute fast hash of first 1MB of video file
    h = hashlib.sha256()
    with video_path.open("rb") as f:
        h.update(f.read(1024 * 1024))
    video_hash = h.hexdigest()

    return SourceMetadata(
        video_file=video_path.name,
        width=int(v.get("width", 3840)),
        height=int(v.get("height", 2160)),
        source_fps=round(source_fps, 4),
        working_fps=24.0,
        pix_fmt=str(v.get("pix_fmt", "yuv420p")),
        color_space=str(v.get("color_space", "bt709")),
        color_primaries=str(v.get("color_primaries", "bt709")),
        color_transfer=str(v.get("color_transfer", "bt709")),
        sha256=video_hash,
    )


def extract_or_reuse_24fps_frames(
    video: str,
    scene: str,
    video_path: Path,
    t_start: float,
    duration: float,
    max_track_fid: int = 0,
    *,
    ffmpeg: str,
) -> list[Path]:
    """Extract 24fps frames or reuse existing extracted frames."""
    dest_dir = BP46_CLIPS_ROOT / video / scene / "extract_24"
    if max_track_fid > 0:
        extract_duration = min(duration, (max_track_fid + 15) / 24.0)
        expected_min = max_track_fid + 1
    else:
        extract_duration = min(duration, 17.0)
        expected_min = min(int(duration * 23.5), 384)

    # Check if already extracted in bp46
    if dest_dir.is_dir():
        pngs = sorted(dest_dir.glob("frame_*.png"))
        if len(pngs) >= expected_min and len(pngs) > 0:
            return pngs

    # Check if extracted in bp21
    bp21_extract = BP21_CLIPS_ROOT / video / scene / "extract_24"
    if bp21_extract.is_dir():
        pngs = sorted(bp21_extract.glob("frame_*.png"))
        if len(pngs) >= expected_min and len(pngs) > 0:
            return pngs

    dest_dir.mkdir(parents=True, exist_ok=True)
    return extract_24fps_pngs(
        video_path,
        t_start,
        extract_duration,
        dest_dir,
        ffmpeg=ffmpeg,
    )


def measure_motion_and_panorama(
    frames: np.ndarray,
) -> tuple[CameraMotionFeatures, PanoramaFeatures]:
    """Measure frame-to-frame difference and panorama canvas growth."""
    diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    consecutive_mad = float(diffs.mean())
    vs_first_frame_mad = float(
        np.abs(frames[1:].astype(np.int16) - frames[:1].astype(np.int16)).mean()
    )
    last_vs_first_mad = float(
        np.abs(frames[-1].astype(np.int16) - frames[0].astype(np.int16)).mean()
    )
    motion = CameraMotionFeatures(
        consecutive_mad=round(consecutive_mad, 3),
        vs_first_frame_mad=round(vs_first_frame_mad, 3),
        last_vs_first_mad=round(last_vs_first_mad, 3),
    )

    # Estimate homographies on a sampled subset if frames is large, to preserve host I/O
    stride = max(1, len(frames) // 48)
    sampled = frames[::stride]
    H = estimate_homographies(sampled)
    c_res = _canvas(H, frames.shape[2], frames.shape[1])
    if c_res is not None:
        _adj, (cw, ch) = c_res
        growth = (cw * ch) / (frames.shape[2] * frames.shape[1])
        panorama = PanoramaFeatures(
            canvas_width=cw,
            canvas_height=ch,
            growth_factor=round(growth, 3),
            registration_ok=True,
        )
    else:
        panorama = PanoramaFeatures(
            canvas_width=frames.shape[2],
            canvas_height=frames.shape[1],
            growth_factor=99.0,
            registration_ok=False,
        )
    return motion, panorama


def evaluate_objects_and_tracks(
    track_pairs: list[list[TrackPair]],
    window_frame_ids: list[int],
    height: int,
    width: int,
) -> ObjectFeatures:
    """Analyze player tracks for coverage, size, separation and occlusion."""
    index_of = {fid: idx for idx, fid in enumerate(window_frame_ids)}
    n_frames = len(window_frame_ids)
    num_tracks = len(track_pairs)

    total_pixels = 0
    min_dist = float("inf")
    has_occlusion = False

    # Per-frame bboxes
    frame_boxes: dict[int, list[tuple[int, int, int, int]]] = {
        i: [] for i in range(n_frames)
    }

    track_coverage = [0] * num_tracks
    for t_idx, pairs in enumerate(track_pairs):
        for p in pairs:
            if p.frame_id in index_of:
                slot = index_of[p.frame_id]
                track_coverage[t_idx] += 1
                crop = load_rgba(p.crop_path)
                total_pixels += int((crop[..., 3] >= 128).sum())
                frame_boxes[slot].append(p.bbox)

    # Continuity: each track must cover at least 90% of requested window
    is_continuous = all(c >= int(0.9 * n_frames) for c in track_coverage) if num_tracks > 0 else False

    # Check separation and occlusion across frames
    for slot, boxes in frame_boxes.items():
        if len(boxes) >= 2:
            b1, b2 = boxes[0], boxes[1]
            # Centers
            c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
            c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
            dist = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
            if dist < min_dist:
                min_dist = dist
            # Box intersection
            x_left = max(min(b1[0], b1[2]), min(b2[0], b2[2]))
            y_top = max(min(b1[1], b1[3]), min(b2[1], b2[3]))
            x_right = min(max(b1[0], b1[2]), max(b2[0], b2[2]))
            y_bottom = min(max(b1[1], b1[3]), max(b2[1], b2[3]))
            if x_right > x_left and y_bottom > y_top:
                has_occlusion = True

    area_fraction = total_pixels / (n_frames * height * width) if n_frames > 0 else 0.0

    return ObjectFeatures(
        num_objects=num_tracks,
        object_class="player",
        player_pixel_fraction=round(area_fraction, 5),
        min_separation_px=round(min_dist, 1) if min_dist != float("inf") else 0.0,
        has_occlusion=has_occlusion,
        track_continuity=is_continuous,
    )


def get_player_track_pairs(track_pairs: list[list[TrackPair]]) -> list[list[TrackPair]]:
    """Select the two player tracks (longest track sequences)."""
    if len(track_pairs) <= 2:
        return track_pairs
    return sorted(track_pairs, key=len, reverse=True)[:2]


def load_and_downscale_stack(
    png_paths: list[Path],
    target_size: tuple[int, int] = (960, 540),
    max_workers: int = 12,
) -> np.ndarray:
    """Load frames and downscale to target_size in memory using parallel threads."""
    if not png_paths:
        return np.zeros((0, target_size[1], target_size[0], 3), dtype=np.uint8)

    def _read_one(p: Path) -> np.ndarray | None:
        img = cv2.imread(str(p))
        if img is None:
            return None
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        frames = list(ex.map(_read_one, png_paths))

    valid = [f for f in frames if f is not None]
    if not valid:
        return np.zeros((0, target_size[1], target_size[0], 3), dtype=np.uint8)
    return np.stack(valid, axis=0)


def find_simultaneous_player_window(
    player_tracks: list[list[TrackPair]],
    target_span: int,
    total_frames: int,
) -> tuple[int, int] | None:
    """Find a window [start, start + target_span] where both players are continuously tracked."""
    if len(player_tracks) < 2:
        return None
    s1 = set(p.frame_id for p in player_tracks[0])
    s2 = set(p.frame_id for p in player_tracks[1])
    common_fids = s1.intersection(s2)
    if len(common_fids) < int(0.85 * target_span):
        return None
    sorted_common = sorted(common_fids)
    lo = min(sorted_common)
    hi = max(sorted_common)
    best_start = None
    best_count = -1
    for start in range(lo, hi - target_span + 2):
        if start + target_span > total_frames:
            break
        window_set = set(range(start, start + target_span))
        c1 = len(s1.intersection(window_set))
        c2 = len(s2.intersection(window_set))
        min_cov = min(c1, c2)
        if min_cov >= int(0.85 * target_span) and min_cov > best_count:
            best_count = min_cov
            best_start = start
    if best_start is not None:
        return best_start, best_start + target_span
    return None


def evaluate_scene(
    video: str,
    scene: str,
    scene_info: dict[str, Any],
    role: str,
    context_id: str,
    *,
    ffmpeg: str,
    ffprobe: str,
) -> SceneRecord:
    """Extract, evaluate eligibility, and validate intervals for one scene."""
    video_path = RAW_4K_ROOT / f"{video}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Source video missing: {video_path}")

    source_meta = probe_video_source(video_path, ffprobe=ffprobe)
    t_start = float(scene_info["t_start"])
    t_end = float(scene_info["t_end"])
    duration = float(scene_info.get("duration") or (t_end - t_start))
    cluster = str(scene_info.get("cluster", ""))

    # Check track directories
    scene_dir = DATASET_ROOT / video / "segmentations" / scene
    tracks = list_tracks(scene_dir) if scene_dir.is_dir() else []
    track_pairs: list[list[TrackPair]] = []
    if tracks:
        track_pairs = [pair_track(scene_dir, t) for t in tracks]
        track_pairs = [p for p in track_pairs if p]

    # Select the two primary player tracks
    player_tracks = get_player_track_pairs(track_pairs)
    max_track_fid = max((max(p.frame_id for p in pairs) for pairs in player_tracks), default=0)

    # Extract or reuse 24fps frames
    work_dir = BP46_CLIPS_ROOT / video / scene
    pngs_24 = extract_or_reuse_24fps_frames(
        video, scene, video_path, t_start, duration, max_track_fid=max_track_fid, ffmpeg=ffmpeg
    )
    total_24_frames = len(pngs_24)

    ineligibility_reasons: list[str] = []
    if role != "control_ineligible":
        if "point" not in cluster.lower() or "interlude" in cluster.lower():
            ineligibility_reasons.append(f"cluster '{cluster}' is not a valid point camera")
        if not player_tracks or len(player_tracks) < 2:
            ineligibility_reasons.append(f"expected 2 player tracks, found {len(player_tracks)}")
    else:
        ineligibility_reasons.append(f"cluster '{cluster}' is designated ineligible control")

    # Find base 48-frame simultaneous player window
    base_window = find_simultaneous_player_window(
        player_tracks, min(total_24_frames, 48), total_24_frames
    )
    if base_window is not None:
        base_start, base_end = base_window
    else:
        base_start, base_end = 0, min(total_24_frames, 48)
        if role != "control_ineligible":
            ineligibility_reasons.append("player tracks do not simultaneously cover base window")

    # Convention & paste-back
    flat = [p for pairs in player_tracks for p in pairs if base_start <= p.frame_id < base_end]
    if not flat:
        flat = [p for pairs in player_tracks for p in pairs]
    convention = "unknown"
    opaque_paste_mae = 999.0
    passes_paste = False
    if flat and role != "control_ineligible":
        try:
            diag = diagnose_convention(
                video_path,
                {"video": video, "scene": scene},
                flat,
                work_dir / "paste_back",
                ffmpeg=ffmpeg,
                pngs_24=pngs_24,
            )
            convention = str(diag.get("winner", "unknown"))
            opaque_paste_mae = float(diag.get("winner_mae", 999.0))
            passes_paste = opaque_paste_mae <= PASTE_MAE_MAX
            if not passes_paste:
                ineligibility_reasons.append(
                    f"paste-back MAE {opaque_paste_mae:.2f} exceeds {PASTE_MAE_MAX}"
                )
        except Exception as exc:
            ineligibility_reasons.append(f"paste-back error: {exc}")
    elif role == "control_ineligible":
        opaque_paste_mae = 999.0
        passes_paste = False
    else:
        ineligibility_reasons.append("no track pairs for paste-back check")

    # Base feature evaluation on base window
    base_pngs = pngs_24[base_start:base_end]
    base_540p = load_and_downscale_stack(base_pngs, target_size=(960, 540))
    if len(base_540p) >= 2:
        motion_feats, pano_feats = measure_motion_and_panorama(base_540p)
    else:
        motion_feats = CameraMotionFeatures(0.0, 0.0, 0.0)
        pano_feats = PanoramaFeatures(source_meta.width, source_meta.height, 1.0, False)

    if pano_feats.growth_factor > MAX_CANVAS_GROWTH and role != "control_ineligible":
        ineligibility_reasons.append(
            f"canvas growth {pano_feats.growth_factor:.2f}x exceeds {MAX_CANVAS_GROWTH}x"
        )
    if motion_feats.consecutive_mad > MAX_CONSECUTIVE_MAD and role != "control_ineligible":
        ineligibility_reasons.append(
            f"consecutive frame MAD {motion_feats.consecutive_mad:.2f} exceeds {MAX_CONSECUTIVE_MAD}"
        )

    # Object features over base window
    eval_fids = list(range(base_start, base_end))
    obj_feats = evaluate_objects_and_tracks(
        player_tracks, eval_fids, source_meta.height, source_meta.width
    )
    if obj_feats.has_occlusion and role != "control_ineligible":
        ineligibility_reasons.append("player occlusion / crossing detected")

    paste_feats = PasteBackFeatures(
        convention=convention,
        opaque_mae=round(opaque_paste_mae, 3),
        threshold=PASTE_MAE_MAX,
        passes_threshold=passes_paste,
    )

    is_eligible = len(ineligibility_reasons) == 0
    routing = "pointstream" if is_eligible else "conventional_fallback"

    eligibility = EligibilityFeatures(
        duration_24fps_frames=total_24_frames,
        camera_motion=motion_feats,
        panorama=pano_feats,
        objects=obj_feats,
        paste_back=paste_feats,
        route=routing,
        ineligibility_reasons=ineligibility_reasons,
    )

    # Validate intervals (48, 96, 192, 384 frames) on each interval's own frames
    interval_records: dict[str, IntervalValidation] = {}
    for span in TARGET_SPANS:
        window = find_simultaneous_player_window(player_tracks, span, total_24_frames)
        if window is None:
            reasons = []
            if total_24_frames < span:
                reasons.append(f"source frames {total_24_frames} < {span}")
            if not player_tracks or len(player_tracks) < 2:
                reasons.append(f"expected 2 player tracks, found {len(player_tracks)}")
            else:
                s1 = set(p.frame_id for p in player_tracks[0])
                s2 = set(p.frame_id for p in player_tracks[1])
                overlap = len(s1.intersection(s2))
                if overlap < span:
                    reasons.append(f"simultaneous player overlap {overlap} < {span} frames")
                else:
                    reasons.append(f"player tracks lack continuous coverage for {span} frames")

            interval_records[str(span)] = IntervalValidation(
                frame_count=span,
                start_frame=0,
                end_frame=min(span, total_24_frames),
                status="insufficient_duration" if total_24_frames < span or (player_tracks and min(len(p) for p in player_tracks) < span) else "ineligible",
                frame_hashes={},
                paste_back_mae=0.0,
                canvas_growth=0.0,
                failure_reasons=reasons,
            )
            continue

        start_f, end_f = window
        span_pngs = pngs_24[start_f:end_f]
        h_first = _sha256_file(span_pngs[0])
        h_mid = _sha256_file(span_pngs[len(span_pngs) // 2])
        h_last = _sha256_file(span_pngs[-1])
        frame_hashes = {"first": h_first, "mid": h_mid, "last": h_last}

        # True interval-specific motion and canvas growth measured on the exact span frames
        span_540p = load_and_downscale_stack(span_pngs, target_size=(960, 540))
        span_motion, span_pano = measure_motion_and_panorama(span_540p)
        span_growth = span_pano.growth_factor
        span_mad = span_motion.consecutive_mad

        interval_reasons: list[str] = []
        if span_growth > MAX_CANVAS_GROWTH and role != "control_ineligible":
            interval_reasons.append(f"canvas growth {span_growth:.2f}x exceeds {MAX_CANVAS_GROWTH}x")
        if span_mad > MAX_CONSECUTIVE_MAD and role != "control_ineligible":
            interval_reasons.append(f"consecutive frame MAD {span_mad:.2f} exceeds {MAX_CONSECUTIVE_MAD}")
        if not span_passes_paste and role != "control_ineligible":
            interval_reasons.append(f"interval paste-back MAE {span_paste_mae:.2f} exceeds {PASTE_MAE_MAX}")

        intv_status = "eligible" if len(interval_reasons) == 0 else "ineligible"

        interval_records[str(span)] = IntervalValidation(
            frame_count=span,
            start_frame=start_f,
            end_frame=end_f,
            status=intv_status,
            frame_hashes=frame_hashes,
            paste_back_mae=round(span_paste_mae, 3),
            canvas_growth=round(span_growth, 3),
            failure_reasons=interval_reasons,

    return SceneRecord(
        video=video,
        scene=scene,
        t_start=t_start,
        t_end=t_end,
        duration=round(duration, 3),
        cluster=cluster,
        context_id=context_id,
        role=role,
        source_metadata=source_meta,
        eligibility=eligibility,
        intervals=interval_records,
    )


def save_long_scene_cache(
    video: str,
    scene: str,
    span: int,
    record: SceneRecord,
    *,
    ffmpeg: str,
) -> None:
    """Cache window frames and player masks for verified long scenes."""
    intv = record.intervals.get(str(span))
    if intv is None or intv.status != "eligible":
        return

    work_dir = BP46_CLIPS_ROOT / video / scene
    window_dir = work_dir / f"window_{span}"
    window_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = work_dir / "extract_24"
    if not extract_dir.is_dir():
        extract_dir = BP21_CLIPS_ROOT / video / scene / "extract_24"
    pngs = sorted(extract_dir.glob("frame_*.png"))[intv.start_frame : intv.end_frame]
    if len(pngs) != span:
        return

    # Link or copy frames into window_{span}
    for idx, png in enumerate(pngs):
        target = window_dir / f"frame_{idx:06d}.png"
        if not target.exists():
            target.symlink_to(png.resolve())

    # Build and cache masks
    scene_dir = DATASET_ROOT / video / "segmentations" / scene
    tracks = list_tracks(scene_dir) if scene_dir.is_dir() else []
    track_pairs = [pair_track(scene_dir, t) for t in tracks]
    frame_ids = list(range(intv.start_frame, intv.end_frame))
    masks = np.zeros((span, record.source_metadata.height, record.source_metadata.width), dtype=bool)
    index_of = {fid: idx for idx, fid in enumerate(frame_ids)}
    for pairs in track_pairs:
        for p in pairs:
            slot = index_of.get(p.frame_id)
            if slot is not None:
                crop = load_rgba(p.crop_path)
                rows, cols = bbox_slices(
                    p.bbox, crop.shape[0], crop.shape[1],
                    record.source_metadata.height, record.source_metadata.width
                )
                masks[slot, rows, cols] |= crop[..., 3] >= 128

    np.savez_compressed(work_dir / f"masks_{span}.npz", masks=masks.astype(np.uint8))


def run_extraction_campaign(
    manifest_out: Path | None = None,
) -> dict[str, Any]:
    """Run the full candidate extraction, evaluation and manifest generation."""
    import datetime

    ffmpeg_tool = resolve_ffmpeg()
    ffmpeg = ffmpeg_tool.path
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))

    # Roster of candidate scenes: strictly isolated splits
    # Diagnostic videos (E1 search) - strictly from alcaraz_highlights
    # Confirmation videos (E2 Gate B confirmation) - 6 independent tournament matches
    # Ineligible controls - high motion / crowd fallback
    roster: list[dict[str, Any]] = [
        # Diagnostic (E1 search): near-static and smooth-pan cases
        {"video": "alcaraz_highlights", "scene": "scene_000", "role": "diagnostic_near_static", "context_id": "alcaraz_highlights_main_court"},
        {"video": "alcaraz_highlights", "scene": "scene_028", "role": "diagnostic_near_static", "context_id": "alcaraz_highlights_main_court"},
        {"video": "alcaraz_highlights", "scene": "scene_010", "role": "diagnostic_smooth_pan", "context_id": "alcaraz_highlights_main_court"},
        {"video": "alcaraz_highlights", "scene": "scene_018", "role": "diagnostic_smooth_pan", "context_id": "alcaraz_highlights_main_court"},
        {"video": "alcaraz_highlights", "scene": "scene_026", "role": "diagnostic_smooth_pan", "context_id": "alcaraz_highlights_main_court"},

        # Ineligible control (high motion / crowd / non-court camera)
        {"video": "alcaraz_highlights", "scene": "scene_006", "role": "control_ineligible", "context_id": "alcaraz_highlights_crowd_side"},

        # Confirmation Match 1: Alcaraz vs Perricard
        {"video": "alcaraz_perricard", "scene": "scene_002", "role": "confirmation", "context_id": "alcaraz_perricard_main_court"},
        {"video": "alcaraz_perricard", "scene": "scene_003", "role": "confirmation", "context_id": "alcaraz_perricard_main_court"},
        {"video": "alcaraz_perricard", "scene": "scene_004", "role": "confirmation", "context_id": "alcaraz_perricard_main_court"},
        {"video": "alcaraz_perricard", "scene": "scene_005", "role": "confirmation", "context_id": "alcaraz_perricard_main_court"},
        {"video": "alcaraz_perricard", "scene": "scene_006", "role": "confirmation", "context_id": "alcaraz_perricard_main_court"},
        {"video": "alcaraz_perricard", "scene": "scene_007", "role": "confirmation", "context_id": "alcaraz_perricard_main_court"},
        {"video": "alcaraz_perricard", "scene": "scene_010", "role": "confirmation", "context_id": "alcaraz_perricard_main_court"},

        # Confirmation Match 2: Alcaraz vs Ruud
        {"video": "alcaraz_ruud", "scene": "scene_002", "role": "confirmation", "context_id": "alcaraz_ruud_main_court"},
        {"video": "alcaraz_ruud", "scene": "scene_004", "role": "confirmation", "context_id": "alcaraz_ruud_main_court"},

        # Confirmation Match 3: Djokovic vs Federer (Wimbledon 2019)
        {"video": "djokovic_federer", "scene": "scene_003", "role": "confirmation", "context_id": "djokovic_federer_main_court"},
        {"video": "djokovic_federer", "scene": "scene_005", "role": "confirmation", "context_id": "djokovic_federer_main_court"},
        {"video": "djokovic_federer", "scene": "scene_007", "role": "confirmation", "context_id": "djokovic_federer_main_court"},
        {"video": "djokovic_federer", "scene": "scene_009", "role": "confirmation", "context_id": "djokovic_federer_main_court"},
        {"video": "djokovic_federer", "scene": "scene_011", "role": "confirmation", "context_id": "djokovic_federer_main_court"},
        {"video": "djokovic_federer", "scene": "scene_013", "role": "confirmation", "context_id": "djokovic_federer_main_court"},
        {"video": "djokovic_federer", "scene": "scene_015", "role": "confirmation", "context_id": "djokovic_federer_main_court"},
        {"video": "djokovic_federer", "scene": "scene_017", "role": "confirmation", "context_id": "djokovic_federer_main_court"},
        {"video": "djokovic_federer", "scene": "scene_020", "role": "confirmation", "context_id": "djokovic_federer_main_court"},
        {"video": "djokovic_federer", "scene": "scene_022", "role": "confirmation", "context_id": "djokovic_federer_main_court"},

        # Confirmation Match 4: Djokovic vs Zverev
        {"video": "djokovic_zverev", "scene": "scene_000", "role": "confirmation", "context_id": "djokovic_zverev_main_court"},
        {"video": "djokovic_zverev", "scene": "scene_001", "role": "confirmation", "context_id": "djokovic_zverev_main_court"},
        {"video": "djokovic_zverev", "scene": "scene_002", "role": "confirmation", "context_id": "djokovic_zverev_main_court"},
        {"video": "djokovic_zverev", "scene": "scene_003", "role": "confirmation", "context_id": "djokovic_zverev_main_court"},
        {"video": "djokovic_zverev", "scene": "scene_004", "role": "confirmation", "context_id": "djokovic_zverev_main_court"},
        {"video": "djokovic_zverev", "scene": "scene_005", "role": "confirmation", "context_id": "djokovic_zverev_main_court"},
        {"video": "djokovic_zverev", "scene": "scene_006", "role": "confirmation", "context_id": "djokovic_zverev_main_court"},
        {"video": "djokovic_zverev", "scene": "scene_007", "role": "confirmation", "context_id": "djokovic_zverev_main_court"},

        # Diagnostic Video 2: Federer vs Djokovic (Cincinnati 2015, smooth-pan search)
        {"video": "federer_djokovic", "scene": "scene_001", "role": "diagnostic_smooth_pan", "context_id": "federer_djokovic_main_court"},
        {"video": "federer_djokovic", "scene": "scene_003", "role": "diagnostic_smooth_pan", "context_id": "federer_djokovic_main_court"},
        {"video": "federer_djokovic", "scene": "scene_005", "role": "diagnostic_smooth_pan", "context_id": "federer_djokovic_main_court"},
        {"video": "federer_djokovic", "scene": "scene_007", "role": "diagnostic_smooth_pan", "context_id": "federer_djokovic_main_court"},
        {"video": "federer_djokovic", "scene": "scene_009", "role": "diagnostic_smooth_pan", "context_id": "federer_djokovic_main_court"},
        {"video": "federer_djokovic", "scene": "scene_011", "role": "diagnostic_smooth_pan", "context_id": "federer_djokovic_main_court"},
        {"video": "federer_djokovic", "scene": "scene_013", "role": "diagnostic_smooth_pan", "context_id": "federer_djokovic_main_court"},
        {"video": "federer_djokovic", "scene": "scene_015", "role": "diagnostic_smooth_pan", "context_id": "federer_djokovic_main_court"},
        {"video": "federer_djokovic", "scene": "scene_017", "role": "diagnostic_smooth_pan", "context_id": "federer_djokovic_main_court"},
        {"video": "federer_djokovic", "scene": "scene_019", "role": "diagnostic_smooth_pan", "context_id": "federer_djokovic_main_court"},

        # Confirmation Match 5: Sinner vs Alcaraz
        {"video": "sinner_alcaraz", "scene": "scene_001", "role": "confirmation", "context_id": "sinner_alcaraz_main_court"},
    print(f"Evaluating {len(roster)} candidate scenes...", flush=True)

    submitted_count = len(roster)
    succeeded_by_span = {span: 0 for span in TARGET_SPANS}
    failed_by_span = {span: 0 for span in TARGET_SPANS}
    eligible_count = 0
    ineligible_count = 0

    print(f"Pre-extracting / validating frame caches for {len(roster)} scenes in parallel...", flush=True)

    def _pre_extract(item: dict[str, Any]) -> None:
        v = item["video"]
        s = item["scene"]
        meta_p = DATASET_ROOT / v / "scene_metadata.json"
        if meta_p.exists():
            d = json.loads(meta_p.read_text())
            for i, sc in enumerate(d.get("scenes") or d.get("segments") or []):
                if sc.get("scene") == s or f"scene_{i:03d}" == s:
                    t_s = float(sc["t_start"])
                    t_e = float(sc["t_end"])
                    dur = float(sc.get("duration") or (t_e - t_s))
                    vp = RAW_4K_ROOT / f"{v}.mp4"
                    if vp.is_file():
                        extract_or_reuse_24fps_frames(v, s, vp, t_s, dur, ffmpeg=ffmpeg)
                    break

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_pre_extract, roster))

    print("Pre-extraction complete. Running per-interval feature evaluation...", flush=True)

    for item in roster:
        video = item["video"]
        scene = item["scene"]
        role = item["role"]
        context_id = item["context_id"]

        # Read scene metadata
        meta_path = DATASET_ROOT / video / "scene_metadata.json"
        scene_info = {}
        if meta_path.exists():
            data = json.loads(meta_path.read_text())
            scenes = data.get("scenes") or data.get("segments") or []
            for i, s in enumerate(scenes):
                if s.get("scene") == scene or f"scene_{i:03d}" == scene:
                    scene_info = s
                    break

        print(f"Processing {video}/{scene} (role: {role})...", flush=True)
        rec = evaluate_scene(
            video, scene, scene_info, role, context_id, ffmpeg=ffmpeg, ffprobe=ffprobe
        )
        evaluated_scenes.append(rec)

        if rec.eligibility.route == "pointstream":
            eligible_count += 1
        else:
            ineligible_count += 1

        for span in TARGET_SPANS:
            intv = rec.intervals.get(str(span))
            if intv and intv.status == "eligible":
                succeeded_by_span[span] += 1
            else:
                failed_by_span[span] += 1

    summary = {
        "submitted_scenes": submitted_count,
        "pointstream_eligible_scenes": eligible_count,
        "conventional_fallback_scenes": ineligible_count,
        "succeeded_by_span": {f"{k}_frames": v for k, v in succeeded_by_span.items()},
        "failed_by_span": {f"{k}_frames": v for k, v in failed_by_span.items()},
    }

    diagnostic_vids = sorted(list(set(s.video for s in evaluated_scenes if "diagnostic" in s.role)))
    confirmation_vids = sorted(list(set(s.video for s in evaluated_scenes if s.role == "confirmation")))
    controls = [f"{s.video}/{s.scene}" for s in evaluated_scenes if s.role == "control_ineligible"]

    manifest_dict = {
        "schema": "pointstream.long_scenes.v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_data_root": str(ps_paths.data_root()),
        "target_spans": list(TARGET_SPANS),
        "diagnostic_videos": diagnostic_vids,
        "confirmation_videos": confirmation_vids,
        "ineligible_controls": controls,
        "summary": summary,
        "scenes": [s.to_dict() for s in evaluated_scenes],
    }

    # Write manifest to repo and to outputs
    repo_manifest = ps_paths.repo_root() / "manifests" / "bp46_long_tennis_scenes.json"
    repo_manifest.parent.mkdir(parents=True, exist_ok=True)
    repo_manifest.write_text(json.dumps(manifest_dict, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote repository manifest to {repo_manifest}", flush=True)

    output_manifest = ps_paths.outputs() / "bp46-long-scenes" / "manifest.json"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest_dict, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote output manifest to {output_manifest}", flush=True)

    if manifest_out:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps(manifest_dict, indent=2) + "\n", encoding="utf-8")

    return manifest_dict


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Extract and validate BP46 long tennis scenes.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output manifest path.")
    args = parser.parse_args(argv)

    manifest = run_extraction_campaign(args.out)
    print("\n=== Extraction & Validation Summary ===")
    print(json.dumps(manifest["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
