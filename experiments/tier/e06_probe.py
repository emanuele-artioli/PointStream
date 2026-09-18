"""E06 four-setting full-codec diagnostic on saved display_low Federer scene 007."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import cv2
import numpy as np

from experiments.headroom.real import bbox_slices, list_tracks, load_rgba, pair_track
from experiments.jobs.monitor import publish_progress
from experiments.tier.e03b_persist import DecodeCountError
from experiments.tier.e03b_source import EXPECTED_SHAPE, stack_sha256
from experiments.tier.e06_transport import (
    PREDICTOR_BBOX_RESIZE,
    PREDICTOR_PER_FRAME,
    SIDE_DATA_BYTES,
    background_view,
    client_placements,
    ordinary_composite,
    reconcile_ledger,
    reconstruct_standalone,
    require_exact_count,
    require_pixel_parity,
    require_predictor,
    serialize_setting,
)
from experiments.tier.low_rate_measure import score_headlines
from scripts.background_probe import compute_array_sha256, unpack_panorama_side_data
from src.components.background.sidecar import build_sidecar
from src.contracts.codecs import RateControl
from src.contracts.config import ResidualConfig
from src.contracts.lattice import STAGE_RESIDUAL, StageLattice
from src.pipeline.reconstruction.background import BackgroundResolver
from src.pipeline.residual.codec import encode_residual_to_bitstream
from src.pipeline.residual.signal import compute_residual

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARD = REPO_ROOT / "manifests" / "evaluation_20260916_e06_probe_card.json"
CONFIRMATION_RESERVATION = (
    REPO_ROOT / "manifests" / "evaluation_20260916_coordinator_confirmation_reservation.json"
)

PREPARED_SHA256 = "1f02475a5bbc3d94e4bae2e904dc29c3af3082be0c0c160e027b706a6950f6f8"
MASKS_SHA256 = "a65f16975d5cf8ded6ab64642324971030a30a051180410de381f3021e82d45c"
E03B_REPORT_SHA256 = "62ac0b15228001f7598958d501df133003d8c6faea0b5b6947670a3839fc45f7"
E03B_BOUNDS_SHA256 = "77c30b5b2d81da8bd8abe50e41150571db9d7615bcbdf3ee0657f1cbdb5c2d9a"
E04A_REPORT_SHA256 = "e65ec04dedda40cdc472ace90fcce31e26b10218c15060097c6ca39f5010251d"
PANORAMA_QP47_SHA256 = "31937000b96293420e741283985452233f896075748e657dff142304322947a6"

E03B_RUN = Path(
    "/home/itec/emanuele/pointstream-data/outputs/evaluation-20260914/e03b/"
    "run-20260916-federer007"
)
E04A_RUN = Path(
    "/home/itec/emanuele/pointstream-data/outputs/evaluation-20260914/e04a/"
    "run-20260916-federer007"
)
E06_OUT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/evaluation-20260914/e06/"
    "run-20260916-federer007-perframe-bbox"
)

PROTECTED_NAMES = (
    "probe_report.json",
    "bounds.json",
    "campaign_rows.json",
    "prepared_rgb.npy",
)

BOUNDS: dict[str, Any] = {
    "written_before_headline_scores": True,
    "size_bytes_T": {
        "low": 1000,
        "high": 33177600,
        "basis": "Uncompressed 360p 48f 4:2:0 plus container; tool-floor >1 kB.",
    },
    "psnr_y_db": {
        "low": 15.0,
        "high": 50.0,
        "basis": "Colour/decode mismatch below 15; mild residual ceiling below 50. Severe-blur on this grid 17.73 dB.",
    },
    "ssim": {"low": 0.15, "high": 1.0, "basis": "Unrelated natural tennis can sit near 0.2; identity is 1.0."},
    "vmaf": {
        "low": 0.0,
        "high": 100.0,
        "basis": "libvmaf floors at 0 on this 360p grid (severe-blur/spatial-null); unrelated-clip 1.38; identity ~98.",
    },
    "encode_plus_decode_s_per_setting": {
        "low": 0.2,
        "high": 420.0,
        "basis": "Stop if one setting exceeds 7 minutes.",
    },
    "residual_off_R_bytes": {"low": 0, "high": 0},
}

SETTINGS: tuple[dict[str, Any], ...] = (
    {"id": "per_frame_crop_residual_off", "predictor": PREDICTOR_PER_FRAME, "residual_on": False},
    {"id": "per_frame_crop_residual_on", "predictor": PREDICTOR_PER_FRAME, "residual_on": True},
    {"id": "bbox_resized_first_reference_residual_off", "predictor": PREDICTOR_BBOX_RESIZE, "residual_on": False},
    {"id": "bbox_resized_first_reference_residual_on", "predictor": PREDICTOR_BBOX_RESIZE, "residual_on": True},
)

RESIDUAL = ResidualConfig(
    codec="av1",
    rate_control=RateControl.CRF,
    rate=45,
    preset="8",
    block_size=8,
    block_threshold=4.0,
    background_downscale=2,
)


def sha256_path(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def refuse_overwrite(run_dir: Path) -> None:
    found = [name for name in PROTECTED_NAMES if (run_dir / name).is_file()]
    if found:
        raise FileExistsError(
            f"refusing to overwrite E06 artifacts in {run_dir}: {', '.join(found)}"
        )


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_immutable_reports() -> dict[str, str]:
    report = sha256_path(E03B_RUN / "probe_report.json")
    bounds = sha256_path(E03B_RUN / "bounds.json")
    e04a = sha256_path(E04A_RUN / "probe_report.json")
    if report != E03B_REPORT_SHA256:
        raise ValueError(f"E03B probe_report mutated: {report}")
    if bounds != E03B_BOUNDS_SHA256:
        raise ValueError(f"E03B bounds mutated: {bounds}")
    if e04a != E04A_REPORT_SHA256:
        raise ValueError(f"E04A probe_report mutated: {e04a}")
    return {"e03b_report": report, "e03b_bounds": bounds, "e04a_report": e04a, "prepared": PREPARED_SHA256}


def load_prepared_stack() -> np.ndarray:
    prepared = np.load(E03B_RUN / "prepared_rgb.npy")
    if tuple(prepared.shape) != EXPECTED_SHAPE:
        raise ValueError(f"prepared shape {prepared.shape} != {EXPECTED_SHAPE}")
    digest = stack_sha256(prepared)
    if digest != PREPARED_SHA256:
        raise ValueError(f"prepared SHA-256 {digest} != {PREPARED_SHA256}")
    return prepared


def load_union_masks() -> np.ndarray:
    extract_dir = E03B_RUN / "extract_24"
    pngs = sorted(extract_dir.glob("frame_*.png"))
    if len(pngs) < 96:
        raise FileNotFoundError(f"E03B extract_24 needs 96 pngs; found {len(pngs)}")
    first = cv2.imread(str(pngs[0]))
    if first is None:
        raise FileNotFoundError(pngs[0])
    height_4k, width_4k = int(first.shape[0]), int(first.shape[1])
    selected = list(range(0, 96, 2))
    scene_dir = Path("/home/itec/emanuele/pointstream-data/assets/dataset/federer_djokovic/segmentations/scene_007")
    tracks = list_tracks(scene_dir)
    if not tracks:
        raise FileNotFoundError(scene_dir)
    masks_4k = np.zeros((48, height_4k, width_4k), dtype=bool)
    for track in tracks:
        for pair in pair_track(scene_dir, track):
            if pair.frame_id not in selected:
                continue
            slot = selected.index(pair.frame_id)
            crop_rgba = load_rgba(pair.crop_path)
            rows, cols = bbox_slices(
                pair.bbox,
                crop_rgba.shape[0],
                crop_rgba.shape[1],
                height_4k,
                width_4k,
            )
            opaque = crop_rgba[..., 3] >= 128
            masks_4k[slot, rows, cols] |= opaque
    masks = np.stack(
        [
            cv2.resize(mask.astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST).astype(bool)
            for mask in masks_4k
        ],
        axis=0,
    )
    digest = compute_array_sha256(masks)
    if digest != MASKS_SHA256:
        raise ValueError(f"union-mask SHA-256 {digest} != {MASKS_SHA256}")
    return masks


def load_saved_panorama() -> tuple[bytes, bytes, np.ndarray, np.ndarray]:
    bitstream_path = E04A_RUN / "bitstreams" / "registered_panorama_qp47.vvc"
    side_path = E04A_RUN / "bitstreams" / "registered_panorama_qp47_side.bin"
    bitstream = bitstream_path.read_bytes()
    side = side_path.read_bytes()
    digest = sha256_path(bitstream_path)
    if digest != PANORAMA_QP47_SHA256:
        raise ValueError(f"panorama QP47 bitstream {digest} != {PANORAMA_QP47_SHA256}")
    if len(side) != SIDE_DATA_BYTES:
        raise ValueError(f"side data {len(side)} != {SIDE_DATA_BYTES}")
    homographies, _plate_shape, _frame_shape, _fps = unpack_panorama_side_data(side)
    plate = build_sidecar("vvc", intra_qp=47, intra_preset="faster").decode(bitstream)
    if plate.ndim != 3 or plate.shape[-1] != 3:
        raise DecodeCountError(f"decoded panorama plate shape {plate.shape}")
    return bitstream, side, plate, homographies


def residual_lattice(residual_on: bool) -> StageLattice:
    if residual_on:
        return StageLattice.of(STAGE_RESIDUAL)
    return StageLattice.of()


def encode_residual(
    source: np.ndarray,
    reconstruction: np.ndarray,
    masks: np.ndarray,
    work_dir: Path,
) -> tuple[Any, int]:
    result = compute_residual(
        source,
        reconstruction,
        lattice=residual_lattice(True),
        residual=RESIDUAL,
        actor_mask=masks,
    )
    frames = result.payload.frames
    if frames is None:
        raise ValueError("residual-on produced no payload frames")
    transmitted, _decoded = encode_residual_to_bitstream(
        np.asarray(frames, dtype=np.uint8),
        RESIDUAL.encode_request(),
        mode=str(result.payload.mode),
        scale=float(result.payload.scale),
        offset=float(result.payload.offset),
        fps=12.0,
        work_dir=work_dir,
    )
    if transmitted.byte_count <= 0:
        raise ValueError("residual-on R must be > 0")
    return transmitted, 1


def check_reuse() -> dict[str, Any]:
    identities = verify_immutable_reports()
    load_prepared_stack()
    bitstream, side, _plate, _h = load_saved_panorama()
    reservation = json.loads(CONFIRMATION_RESERVATION.read_text(encoding="utf-8"))
    if reservation.get("scoring_authorized") is not False:
        raise ValueError("confirmation scoring must remain unauthorized")
    return {
        "identities": identities,
        "background": {
            "bitstream_sha256": PANORAMA_QP47_SHA256,
            "bitstream_bytes": len(bitstream),
            "side_bytes": len(side),
            "panorama_B": len(bitstream) + len(side),
        },
        "code_head": code_head(),
        "encodes_launched": 0,
        "confirmation_scoring_authorized": False,
        "predictors": [PREDICTOR_PER_FRAME, PREDICTOR_BBOX_RESIZE],
        "limitations": {
            "union_mask": True,
            "multiple_player": "one union mask; not per-object pose warp",
            "pose_present": False,
            "native_pts_equal_pngs": "1/48",
        },
    }


def _in_band(value: float, band: dict[str, float]) -> bool:
    return float(band["low"]) <= float(value) <= float(band["high"])


def run_setting(
    setting: dict[str, Any],
    *,
    frames: np.ndarray,
    masks: np.ndarray,
    background_frames: np.ndarray,
    view: Any,
    panorama_b: int,
    run_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    predictor = require_predictor(setting["predictor"])
    residual_on = bool(setting["residual_on"])
    residual_calls = 0
    transmitted = None
    if residual_on:
        base = ordinary_composite(
            background_frames, frames, masks, predictor, residual=None, quality=50
        )
        work = run_dir / setting["id"]
        work.mkdir(parents=True, exist_ok=True)
        transmitted, residual_calls = encode_residual(frames, base, masks, work)
    elif residual_calls != 0:
        raise AssertionError("residual-off must not call the residual encoder")

    placements, references, actor_f = client_placements(frames, masks, predictor, quality=50)
    payload = serialize_setting(
        background=view,
        frames=frames,
        masks=masks,
        predictor=predictor,
        residual=transmitted,
        quality=50,
    )
    residual_r = 0 if transmitted is None else int(transmitted.byte_count)
    if not residual_on:
        if residual_r != 0 or residual_calls != 0:
            raise ValueError("residual-off requires R=0 and zero residual codec calls")
    elif residual_r <= 0 or residual_calls < 1:
        raise ValueError("residual-on requires R>0 and at least one residual codec call")

    ledger = reconcile_ledger(
        payload,
        panorama_b=panorama_b,
        actor_reference_f=actor_f,
        residual_r=residual_r,
    )
    ordinary = ordinary_composite(
        background_frames, frames, masks, predictor, residual=transmitted, quality=50
    )
    standalone = reconstruct_standalone(payload)
    require_exact_count(
        ordinary,
        expected=int(frames.shape[0]),
        height=int(frames.shape[1]),
        width=int(frames.shape[2]),
        source=f"{setting['id']}-ordinary",
    )
    require_exact_count(
        standalone,
        expected=int(frames.shape[0]),
        height=int(frames.shape[1]),
        width=int(frames.shape[2]),
        source=f"{setting['id']}-standalone",
    )
    require_pixel_parity(ordinary, standalone, source=setting["id"])
    if residual_on:
        off_pixels = ordinary_composite(
            background_frames, frames, masks, predictor, residual=None, quality=50
        )
        if np.array_equal(ordinary, off_pixels):
            raise ValueError(f"{setting['id']}: residual-on did not change pixels")

    score_started = time.perf_counter()
    scores = score_headlines(frames, ordinary)
    score_s = time.perf_counter() - score_started
    elapsed = time.perf_counter() - started
    if elapsed > 420:
        raise TimeoutError(f"{setting['id']} exceeded 7 minutes ({elapsed:.1f}s)")
    setting_dir = run_dir / setting["id"]
    setting_dir.mkdir(parents=True, exist_ok=True)
    (setting_dir / "transport.npz").write_bytes(payload)
    _write(
        setting_dir / "row.json",
        {
            "id": setting["id"],
            "predictor": predictor,
            "residual_on": residual_on,
            "generation": "off",
            "pose_present": False,
            "n_placements": len(placements),
            "n_reference_payloads": len(references) or (0 if predictor == PREDICTOR_BBOX_RESIZE else len(placements)),
            "parts": ledger,
            "scores": scores,
            "timing": {"setting_seconds": round(elapsed, 3), "scoring_seconds": round(score_s, 3)},
            "residual_codec_calls": residual_calls,
            "transport_sha256": sha256_path(setting_dir / "transport.npz"),
            "limitations": "union-mask / multiple-player; bbox resize is not a geometry warp",
        },
    )
    return json.loads((setting_dir / "row.json").read_text(encoding="utf-8"))


def run_probe(run_dir: Path) -> dict[str, Any]:
    refuse_overwrite(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write(run_dir / "bounds.json", BOUNDS)
    publish_progress("bounds_written", 0)
    reuse = check_reuse()
    frames = load_prepared_stack()
    masks = load_union_masks()
    bitstream, side, plate, homographies = load_saved_panorama()
    panorama_b = len(bitstream) + len(side)
    view = background_view(
        bitstream=bitstream,
        side=side,
        plate=plate,
        homographies=homographies,
        width=640,
        height=360,
    )
    bg_frames, _decision = BackgroundResolver().frames_for(
        view, frame_count=48, height=360, width=640
    )
    calib_path = E03B_RUN / "metric-calibration.json"
    calibration = json.loads(calib_path.read_text(encoding="utf-8"))
    rows = []
    wall0 = time.perf_counter()
    for index, setting in enumerate(SETTINGS, start=1):
        publish_progress(setting["id"], index)
        rows.append(
            run_setting(
                setting,
                frames=frames,
                masks=masks,
                background_frames=np.asarray(bg_frames),
                view=view,
                panorama_b=panorama_b,
                run_dir=run_dir,
            )
        )
    wall = time.perf_counter() - wall0
    if wall > 1800:
        raise TimeoutError(f"E06 wall {wall:.1f}s exceeded 30 minutes")
    alarms = []
    for row in rows:
        psnr = float(row["scores"]["psnr_y"])
        ssim = float(row["scores"]["ssim"])
        if not _in_band(psnr, BOUNDS["psnr_y_db"]):
            alarms.append(f"{row['id']}: psnr_y={psnr} outside band")
        if not _in_band(ssim, BOUNDS["ssim"]):
            alarms.append(f"{row['id']}: ssim={ssim} outside band")
        vmaf = row["scores"].get("vmaf")
        if isinstance(vmaf, (int, float)) and not _in_band(float(vmaf), BOUNDS["vmaf"]):
            alarms.append(f"{row['id']}: vmaf={vmaf} outside band")
        if not _in_band(float(row["parts"]["transport_total"]), BOUNDS["size_bytes_T"]):
            alarms.append(f"{row['id']}: T outside band")
        if not row["residual_on"] and int(row["parts"]["residual"]) != 0:
            alarms.append(f"{row['id']}: residual-off R!=0")
    report = {
        "schema": "pointstream.e06_probe_report.v1",
        "code_head": code_head(),
        "host": socket.gethostname(),
        "run_dir": str(run_dir),
        "reuse": reuse,
        "calibration_path": str(calib_path),
        "calibration_sha256": sha256_path(calib_path),
        "calibration": {
            "identical_vmaf": calibration.get("identical_vmaf"),
            "unrelated_vmaf": calibration.get("unrelated_vmaf"),
            "severe_blur_vmaf": calibration.get("severe_blur_vmaf"),
        },
        "confirmation_scoring_authorized": False,
        "development_source": "federer_djokovic/scene_007",
        "n": 1,
        "settings": rows,
        "wall_seconds": round(wall, 3),
        "alarms": alarms,
        "competitive_note": (
            "Paired residual-on vs residual-off >=0.5 dB is a residual-effect signal, "
            "not evidence the rate cost pays. No BD-rate. n=1 development scene."
        ),
    }
    _write(run_dir / "probe_report.json", report)
    _write(run_dir / "campaign_rows.json", rows)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-reuse", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=E06_OUT)
    args = parser.parse_args(argv)
    if args.check_reuse:
        print(json.dumps(check_reuse(), indent=2, default=str))
        return 0
    if args.launch:
        print(json.dumps(run_probe(args.run_dir), indent=2, default=str))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
