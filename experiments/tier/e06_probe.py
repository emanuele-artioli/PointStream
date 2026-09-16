"""E06 smallest full-codec development probe: paste vs warped-reference, residual off/on.

Does not encode. Reuses the E03B prepared stack and E03B/E04A bitstreams.
Launch waits on coordinator acceptance of the costed card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.e03b_source import EXPECTED_SHAPE, stack_sha256
from src.components.appearance.compressed import CompressedImageAppearance
from src.pipeline.reconstruction.compositor import Placement
from src.runner.mask_wire import encode_mask

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARD = REPO_ROOT / "manifests" / "evaluation_20260916_e06_probe_card.json"

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
    "run-20260916-federer007-paste-warp"
)

LEDGER_PARTS = (
    "panorama",
    "actor_reference",
    "residual",
    "metadata",
    "transport_total",
)

PROTECTED_NAMES = (
    "probe_report.json",
    "bounds.json",
    "campaign_rows.json",
    "prepared_rgb.npy",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_overwrite(run_dir: Path) -> None:
    found = [name for name in PROTECTED_NAMES if (run_dir / name).is_file()]
    if found:
        raise FileExistsError(
            f"refusing to overwrite E06 artifacts in {run_dir}: {', '.join(found)}"
        )


def empty_mask_bbox_error(frame_index: int) -> ValueError:
    return ValueError(f"frame {frame_index} has an empty mask; cannot place an object")


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 2:
        raise ValueError(f"mask must be (H, W); got {selected.shape}")
    rows, cols = np.nonzero(selected)
    if rows.size == 0:
        raise empty_mask_bbox_error(-1)
    y1, y2 = int(rows.min()), int(rows.max()) + 1
    x1, x2 = int(cols.min()), int(cols.max()) + 1
    return (x1, y1, x2, y2)


def crop_at_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.ascontiguousarray(frame[y1:y2, x1:x2])


def load_prepared_stack(run_dir: Path = E03B_RUN) -> np.ndarray:
    prepared = np.load(run_dir / "prepared_rgb.npy")
    if tuple(prepared.shape) != EXPECTED_SHAPE:
        raise ValueError(f"prepared shape {prepared.shape} != {EXPECTED_SHAPE}")
    digest = stack_sha256(prepared)
    if digest != PREPARED_SHA256:
        raise ValueError(f"prepared SHA-256 {digest} != {PREPARED_SHA256}")
    return prepared


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
    return {
        "e03b_report": report,
        "e03b_bounds": bounds,
        "e04a_report": e04a,
        "prepared": PREPARED_SHA256,
    }


def verify_saved_background() -> dict[str, Any]:
    payload = E04A_RUN / "bitstreams" / "registered_panorama_qp47.vvc"
    digest = sha256_path(payload)
    if digest != PANORAMA_QP47_SHA256:
        raise ValueError(f"panorama QP47 bitstream {digest} != {PANORAMA_QP47_SHA256}")
    return {
        "path": str(payload),
        "sha256": digest,
        "bytes": payload.stat().st_size,
        "representation": "registered_panorama",
        "qp": 47,
        "reuse": "charge saved B; do not re-encode background",
    }


def appearance_bytes(crop: np.ndarray, *, quality: int = 50) -> bytes:
    encoder = CompressedImageAppearance(quality=quality, format="webp")
    _descriptor, payload = encoder.encode(crop)
    return payload


def paste_placements(frames: np.ndarray, masks: np.ndarray) -> tuple[Placement, ...]:
    if frames.shape[0] != masks.shape[0]:
        raise ValueError("frames and masks must share a time axis")
    items: list[Placement] = []
    for index, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
        if not np.any(mask):
            continue
        bbox = bbox_from_mask(mask)
        items.append(
            Placement(
                crop=crop_at_bbox(frame, bbox),
                bbox=bbox,
                mask=np.asarray(mask, dtype=bool),
                object_id="union",
                frame_index=index,
            )
        )
    if not items:
        raise ValueError("no occupied frames for paste placements")
    return tuple(items)


def warped_reference_placements(
    frames: np.ndarray, masks: np.ndarray
) -> tuple[Placement, ...]:
    paste = paste_placements(frames, masks)
    reference = paste[0].crop.copy()
    return tuple(
        Placement(
            crop=reference,
            bbox=item.bbox,
            mask=item.mask,
            object_id=item.object_id,
            frame_index=item.frame_index,
        )
        for item in paste
    )


def charge_controls(
    frames: np.ndarray,
    masks: np.ndarray,
    *,
    residual_on: bool,
    predictor: str,
) -> dict[str, Any]:
    if predictor == "paste":
        placements = paste_placements(frames, masks)
        appearance_payloads = [appearance_bytes(item.crop) for item in placements]
    elif predictor == "warped_reference":
        placements = warped_reference_placements(frames, masks)
        appearance_payloads = [appearance_bytes(placements[0].crop)]
    else:
        raise ValueError(f"unknown predictor {predictor!r}")
    mask_payload = encode_mask(np.asarray(masks, dtype=np.uint8))
    pose_bytes = 0
    headers = sum(16 for _ in placements)
    metadata = len(mask_payload) + pose_bytes + headers
    actor = int(sum(len(payload) for payload in appearance_payloads))
    residual = None if residual_on else 0
    return {
        "predictor": predictor,
        "residual_on": residual_on,
        "n_placements": len(placements),
        "n_appearance_payloads": len(appearance_payloads),
        "parts": {
            "panorama": None,
            "actor_reference": actor,
            "residual": residual,
            "metadata": metadata,
            "transport_total": None,
        },
        "metadata_subledger": {
            "mask_payload": len(mask_payload),
            "pose_motion": pose_bytes,
            "placement_headers": headers,
            "generator_metadata": 0,
            "envelope_overhead": 0,
            "pose_present": False,
        },
        "generation": "off",
        "raw_parts": (),
    }


def required_ledger_keys(parts: dict[str, Any]) -> None:
    missing = [name for name in LEDGER_PARTS if name not in parts]
    if missing:
        raise ValueError(f"ledger missing {missing}")


def check_reuse() -> dict[str, Any]:
    identities = verify_immutable_reports()
    load_prepared_stack()
    background = verify_saved_background()
    return {"identities": identities, "background": background, "encodes_launched": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-reuse",
        action="store_true",
        help="Verify prepared-stack and saved-anchor hashes. Does not encode.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Forbidden until the coordinator accepts the costed card.",
    )
    args = parser.parse_args(argv)
    if args.launch:
        raise SystemExit("E06 encodes are not authorized by this card; drop --launch")
    if args.check_reuse:
        print(json.dumps(check_reuse(), indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
