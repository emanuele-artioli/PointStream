"""Evaluate lossless E06 floor arms on the saved bbox residual-off compact.

No affine/interpolation predictor, confirmation scoring, GPU, training, or
native video re-encodes. Packing success is recorded separately from inherited
parent quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from typing import Any, Callable

from experiments.tier.e03b_persist import DecodeCountError
from experiments.tier.e06_floor import (
    MODE_INTRA,
    MODE_XOR_KEY,
    is_floor_pack,
    pack_mask_rle,
    pack_thin_placement,
    physical_ledger,
    required_decode_fields,
    unpack_floor_pack,
    zip_inventory_floor,
)
from experiments.tier.e06_pack import zip_inventory
from experiments.tier.e06_pack_audit import (
    AUDIT_OUT,
    BOUNDS as PACK_BOUNDS,
    PARENT_BOUNDS_SHA256,
    PARENT_REPORT_SHA256,
    PARENT_ROWS_SHA256,
    TRANSPORT_SHA256,
    client_timing_to_evidence,
    code_head,
    load_calibration,
    pixel_sha256,
    profile_client,
    sha256_bytes,
)
from experiments.tier.e06_probe import (
    E06_OUT,
    PREPARED_SHA256,
    sha256_path,
    verify_immutable_reports,
)
from experiments.tier.e06_transport import (
    reconstruct_standalone,
    require_exact_count,
    require_pixel_parity,
)

CONTROL_ID = "bbox_resized_first_reference_residual_off"
CONTROL_COMPACT_SHA256 = "f5e7160ffc289fd9b369bf76aa56b686b896a946046b56d6ecf8bae78e1d46b1"
CONTROL_PIXEL_SHA256 = "5f2047dce73e5766d4083a42393dd7094120ee9b7afa8092a4126fcccffeb39d"
VVC_QP47_BYTES = 21288
AV1_QP63_BYTES = 19116

FLOOR_OUT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/evaluation-20260914/e06/"
    "probe-20260917-floor-arms"
)

# Written before headline T / timing. Do not edit after reading compact T.
BOUNDS: dict[str, Any] = {
    "written_before_headline_scores": True,
    "written_before_client_timing": True,
    "T_bytes": {
        "low": 2500,
        "high": 30297,
        "basis": "Cannot exceed current compact control; tool floor above empty zip.",
    },
    "psnr_y_db": {
        "low": 18.0,
        "high": 28.0,
        "basis": "Floor arms inherit control 20.7 dB; below 18 is a decode alarm.",
    },
    "client_s": {
        "low": 0.05,
        "high": 60.0,
        "basis": "Saved compact client 0.34–0.56 s n=3 on gpu5.",
    },
    "floor_pixel_mismatch": {"low": 0, "high": 0},
}

SETTINGS: tuple[dict[str, Any], ...] = (
    {
        "id": "control_bbox_compact_residual_off",
        "axis": "control",
        "action": "reuse_saved_compact",
        "pack": "identity",
    },
    {
        "id": "mask_rle_per_frame",
        "axis": "floor",
        "action": "repack_only",
        "pack": "rle_intra",
    },
    {
        "id": "mask_keyframe_xor_every_4",
        "axis": "floor",
        "action": "repack_only",
        "pack": "rle_xor_key4",
    },
    {
        "id": "placement_delta_int16_thin_envelope",
        "axis": "floor",
        "action": "repack_only",
        "pack": "thin_placement",
    },
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def refuse_overwrite(path: Path) -> None:
    if (path / "floor_report.json").is_file():
        raise FileExistsError(f"refusing to overwrite floor probe in {path}")


def _in_band(value: float, band: dict[str, float]) -> bool:
    return float(band["low"]) <= float(value) <= float(band["high"])


def _pack(compact: bytes, kind: str) -> bytes:
    if kind == "identity":
        return compact
    if kind == "rle_intra":
        return pack_mask_rle(compact, mode=MODE_INTRA)
    if kind == "rle_xor_key4":
        return pack_mask_rle(compact, mode=MODE_XOR_KEY)
    if kind == "thin_placement":
        return pack_thin_placement(compact)
    raise ValueError(f"unknown pack {kind!r}")


def run_probe(
    out_dir: Path,
    *,
    repeats: int = 2,
    warmup: int = 1,
    profile_client_fn: Callable[..., dict[str, Any]] = profile_client,
) -> dict[str, Any]:
    refuse_overwrite(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "bounds.json", BOUNDS)
    identities = verify_immutable_reports()
    parent_report = sha256_path(E06_OUT / "probe_report.json")
    parent_bounds = sha256_path(E06_OUT / "bounds.json")
    parent_rows = sha256_path(E06_OUT / "campaign_rows.json")
    if parent_report != PARENT_REPORT_SHA256:
        raise ValueError(f"parent probe_report mutated: {parent_report}")
    if parent_bounds != PARENT_BOUNDS_SHA256:
        raise ValueError(f"parent bounds mutated: {parent_bounds}")
    if parent_rows != PARENT_ROWS_SHA256:
        raise ValueError(f"parent rows mutated: {parent_rows}")
    original_rows = json.loads((E06_OUT / "campaign_rows.json").read_text(encoding="utf-8"))
    original = {row["id"]: row for row in original_rows}[CONTROL_ID]
    parent_transport = (E06_OUT / CONTROL_ID / "transport.npz").read_bytes()
    if sha256_bytes(parent_transport) != TRANSPORT_SHA256[CONTROL_ID]:
        raise ValueError("parent bbox residual-off transport hash mismatch")
    compact_path = AUDIT_OUT / CONTROL_ID / "transport_compact.npz"
    compact = compact_path.read_bytes()
    if sha256_bytes(compact) != CONTROL_COMPACT_SHA256:
        raise ValueError("saved compact hash mismatch")
    calibration = load_calibration()
    control_frames = reconstruct_standalone(compact)
    require_exact_count(control_frames, expected=48, height=360, width=640, source="control")
    control_pixels = pixel_sha256(control_frames)
    if control_pixels != CONTROL_PIXEL_SHA256:
        raise ValueError(f"control pixel SHA-256 {control_pixels}")
    parent_scores = original["scores"]
    derived_rows: list[dict[str, Any]] = []
    alarms: list[str] = []
    wall0 = time.perf_counter()
    host = socket.gethostname()
    for setting in SETTINGS:
        setting_id = str(setting["id"])
        packed = _pack(compact, str(setting["pack"]))
        if setting["pack"] != "identity" and not is_floor_pack(packed):
            raise ValueError(f"{setting_id} is not a floor pack")
        envelope = unpack_floor_pack(packed)
        required_decode_fields(envelope)
        packed_frames = reconstruct_standalone(packed)
        require_exact_count(packed_frames, expected=48, height=360, width=640, source=setting_id)
        pixel_ok = True
        pixel_error = None
        try:
            require_pixel_parity(control_frames, packed_frames, source=setting_id)
        except DecodeCountError as exc:
            pixel_ok = False
            pixel_error = str(exc)
            alarms.append(f"{setting_id}: pixel mismatch ({pixel_error})")
        inventory = zip_inventory(packed) if setting["pack"] == "identity" else zip_inventory_floor(packed)
        ledger = physical_ledger(packed)
        profile = profile_client_fn(packed, warmup=warmup, repeats=repeats)
        require_pixel_parity(packed_frames, profile["frames"], source=f"{setting_id}-timed")
        mean_t = float(profile["total_s"]["mean"] or 0)
        total_bytes = int(inventory["transport_total"])
        if total_bytes != len(packed) or total_bytes != int(ledger["transport_total"]):
            raise ValueError(f"{setting_id}: T is not the physical file length")
        if not _in_band(float(total_bytes), BOUNDS["T_bytes"]):
            alarms.append(f"{setting_id}: T={total_bytes} outside band")
        if not _in_band(mean_t, BOUNDS["client_s"]):
            alarms.append(f"{setting_id}: client seconds outside band")
        packing_success = bool(pixel_ok and ledger["reconciled"])
        setting_dir = out_dir / setting_id
        setting_dir.mkdir(parents=True, exist_ok=True)
        packed_path = setting_dir / "transport_floor.npz"
        packed_path.write_bytes(packed)
        timing_evidence = client_timing_to_evidence(profile, setting_id=setting_id, host=host)
        if int(timing_evidence["n_repeats"]) < 2:
            raise ValueError(f"{setting_id}: need n_repeats>=2")
        row = {
            "id": setting_id,
            "axis": setting["axis"],
            "action": setting["action"],
            "pack": setting["pack"],
            "packing_success": packing_success,
            "pixel_match_control": pixel_ok,
            "pixel_error": pixel_error,
            "predictor_quality": {
                "status": "inherited_not_rescored" if packing_success else "not_applicable_pack_failed",
                "note": (
                    "Lossless floor arms do not change the bbox-resized predictor. "
                    "Do not read packing T as a quality result."
                ),
                "parent_scores": parent_scores if packing_success else None,
            },
            "T_bytes": total_bytes,
            "ledger": {k: v for k, v in ledger.items() if k != "entries"},
            "compact_inventory": inventory,
            "parent_compact_sha256": CONTROL_COMPACT_SHA256,
            "packed_sha256": sha256_path(packed_path),
            "pixel_sha256": pixel_sha256(packed_frames),
            "client_timing": {k: v for k, v in profile.items() if k != "frames"},
            "timing_evidence": timing_evidence,
            "artifact_path": str(packed_path),
        }
        _write(setting_dir / "row.json", row)
        derived_rows.append(row)
        del profile["frames"]
    wall = time.perf_counter() - wall0
    if wall > 900:
        raise TimeoutError(f"floor probe wall {wall:.1f}s exceeded 15 minutes")
    packed_ts = [int(row["T_bytes"]) for row in derived_rows]
    packing_ok = all(bool(row["packing_success"]) for row in derived_rows)
    lowest_t = min(packed_ts)
    report = {
        "schema": "pointstream.e06_floor_probe.v1",
        "code_head": code_head(),
        "host": host,
        "run_dir": str(out_dir),
        "parent_compact": str(compact_path),
        "parent_hashes": {
            "probe_report": parent_report,
            "bounds": parent_bounds,
            "campaign_rows": parent_rows,
            "prepared": PREPARED_SHA256,
            "e03b_report": identities["e03b_report"],
            "e03b_bounds": identities["e03b_bounds"],
            "e04a_report": identities["e04a_report"],
            "control_compact": CONTROL_COMPACT_SHA256,
            "control_pixels": CONTROL_PIXEL_SHA256,
        },
        "calibration": calibration,
        "confirmation_scoring_authorized": False,
        "native_reencodes": 0,
        "gpu_hours": 0,
        "predictor_arms_run": False,
        "n_repeats": repeats,
        "settings": derived_rows,
        "wall_seconds": round(wall, 3),
        "alarms": alarms,
        "packing": {
            "all_lossless": packing_ok,
            "lowest_T": lowest_t,
            "control_T": 30297,
            "below_vvc_qp47": lowest_t < VVC_QP47_BYTES,
            "below_av1_qp63": lowest_t < AV1_QP63_BYTES,
        },
        "predictor_quality": {
            "evaluated": False,
            "inherited_control_psnr_y": parent_scores["psnr_y"] if packing_ok else None,
            "note": "Affine/interpolation arms were not run.",
        },
        "next_decision": (
            "If packing is lossless and lowest T is still >= 21288 B, stop this floor path "
            "and do not start predictor arms on this envelope. If T < 21288 B with exact "
            "pixels, cost the affine/first-last residual-off predictor card next."
        ),
        "pack_bounds_unused_high": PACK_BOUNDS["compact_T_bytes"]["high"],
    }
    _write(out_dir / "floor_report.json", report)
    _write(out_dir / "campaign_rows.json", derived_rows)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=FLOOR_OUT)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    if args.launch:
        print(json.dumps(run_probe(args.out_dir, repeats=args.repeats), indent=2, default=str))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
