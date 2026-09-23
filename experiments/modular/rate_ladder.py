"""PointStream Modular Rate Ladder Runner.

Assembles and evaluates PointStream's measured modular pipeline across rungs:
  C0: compact background plate + initial keyframe crop
  C1: C0 + Adaptive Keyframe Crops (OKS >= 0.80, wire <= 12 kB)
  C2: C1 + Steered Cropped Actor Residual (wire <= 4.5 kB)
  C3: C2 + Band-Limited Background Residual Target

Evaluates against conventional video codecs (VVC, SVT-AV1) across:
  Short Horizon (48 frames, Federer-Djokovic scene 007)
  Long Horizon (192 frames, Alcaraz scene 000)

Dual-perspective evaluation:
  1. Saliency-Weighted Quality: PSNR_weighted = 0.7 * PSNR_fg + 0.3 * PSNR_bg
  2. Full Decomposed Transparency: PSNR_overall, PSNR_fg, PSNR_bg, pose_oks
All evaluated against pristine 4K original ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modular.measured_ladder import (  # noqa: E402
    MeasuredInputError,
    MeasuredRung,
    beats,
    load_sequence,
    measure_native_anchor,
    measure_rungs,
    write_comparison_strip,
)

DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "modular_rate_ladder.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "modular" / "rate_ladder"
DEFAULT_VISUALS_DIR = REPO_ROOT / "outputs" / "modular" / "visuals"

_VVC_QP = 46
_AV1_QP = 54


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _saliency_weights(manifest: dict[str, Any]) -> tuple[float, float]:
    scope = manifest.get("evaluation_scope") or {}
    weights = scope.get("saliency_weights") if isinstance(scope, dict) else None
    if isinstance(weights, dict) and "foreground" in weights and "background" in weights:
        return float(weights["foreground"]), float(weights["background"])
    saliency = manifest.get("saliency", {})
    if not isinstance(saliency, dict):
        saliency = {}
    fg_weight = float(saliency.get("fg_weight", manifest.get("fg_weight", 0.70)))
    bg_weight = float(saliency.get("bg_weight", manifest.get("bg_weight", 0.30)))
    return fg_weight, bg_weight


def _n_frames(source: dict[str, Any], manifest: dict[str, Any]) -> int:
    if "n_frames" in source:
        return int(source["n_frames"])
    horizon_id = source.get("horizon") or source.get("horizon_id")
    for horizon in manifest.get("horizons") or []:
        if isinstance(horizon, dict) and horizon.get("id") == horizon_id and "n_frames" in horizon:
            return int(horizon["n_frames"])
    raise MeasuredInputError("source is missing n_frames and does not name a horizon that has one")


def _manifest_sources(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    sources = manifest.get("sources")
    if sources is None:
        sources = manifest.get("horizons")
    if sources is None:
        return []
    if not isinstance(sources, list):
        raise MeasuredInputError("manifest sources must be a list")
    return sources


def _require_source_paths(sources: list[dict[str, Any]]) -> None:
    for index, source in enumerate(sources):
        if not source.get("frames_dir") or not source.get("mask_path"):
            raise MeasuredInputError(
                f"source[{index}] requires frames_dir and mask_path"
            )


def _try_native_anchor(
    frames_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    codec: str,
    qp: int,
    work_dir: Path,
) -> Any | None:
    try:
        anchor = measure_native_anchor(
            frames_rgb,
            codec=codec,
            qp=qp,
            work_dir=work_dir,
            mask=mask,
        )
    except FileNotFoundError:
        return None
    return anchor


def _summary_verdict(
    c1: MeasuredRung,
    *,
    beats_vvc_rate: bool,
    anchor_vvc: Any | None,
) -> str:
    anchor_weighted = getattr(anchor_vvc, "psnr_weighted", None)
    weighted_is_higher = bool(
        anchor_weighted is not None
        and c1.psnr_weighted is not None
        and np.isfinite(c1.psnr_weighted)
        and np.isfinite(anchor_weighted)
        and float(c1.psnr_weighted) > float(anchor_weighted)
    )
    if not beats_vvc_rate:
        if weighted_is_higher:
            return "C1 is higher on weighted quality; rate comparison is withheld."
        return "Rate comparison is withheld."
    if (
        weighted_is_higher
    ):
        return "C1 beats VVC on rate and weighted quality is higher."
    return "C1 beats VVC on rate; weighted quality is not higher."


def _rung_to_json(
    rung: MeasuredRung,
    *,
    beats_vvc_rate: bool,
    beats_av1_rate: bool,
    anchor_vvc: Any | None = None,
    anchor_av1: Any | None = None,
) -> dict[str, Any]:
    def weighted_beats(anchor: Any | None) -> bool:
        quality = getattr(anchor, "psnr_weighted", None)
        return bool(
            quality is not None
            and rung.psnr_weighted is not None
            and np.isfinite(quality)
            and np.isfinite(rung.psnr_weighted)
            and float(rung.psnr_weighted) > float(quality)
        )

    return {
        "rung_id": rung.rung_id,
        "bytes_background": rung.bytes_background,
        "bytes_appearance": rung.bytes_appearance,
        "bytes_metadata": rung.bytes_metadata,
        "bytes_residual": rung.bytes_residual,
        "bytes_container": rung.bytes_container,
        "total_bytes": rung.total_bytes,
        "psnr_weighted": rung.psnr_weighted,
        "psnr_overall": rung.psnr_overall,
        "psnr_fg": rung.psnr_fg,
        "psnr_bg": rung.psnr_bg,
        "pose_oks": None,
        "beats_vvc_rate": beats_vvc_rate,
        "beats_av1_rate": beats_av1_rate,
        "beats_vvc_weighted_quality": weighted_beats(anchor_vvc),
        "beats_av1_weighted_quality": weighted_beats(anchor_av1),
    }


def run_rate_ladder(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    visuals_dir: Path = DEFAULT_VISUALS_DIR,
    dry_run: bool = False,
    generate_visuals: bool = False,
    enforce_gpu: bool = True,
    measure_anchors: bool = False,
) -> dict[str, Any]:
    """Encode every manifest source and write the measured ladder.

    Each source object must include ``frames_dir`` and ``mask_path``. Frames
    and the mask are loaded with ``load_sequence`` for that horizon's
    ``n_frames``, then scored with ``measure_rungs`` using the manifest
    saliency weights. ``dry_run`` and ``enforce_gpu`` do not skip measurement
    and do not invent byte counts: JPEG and WebP do not need a GPU.

    When ``measure_anchors`` is true, each source is also passed to
    ``measure_native_anchor`` for VVC QP 46 and SVT-AV1 QP 54. A missing
    binary leaves that anchor's byte and PSNR fields null. ``beats_vvc_rate``
    and ``beats_av1_rate`` on each rung come only from ``beats``. Weighted
    quality is compared only when both sides have finite regional scores from
    the same mask and weights. The verdict distinguishes a weighted-quality
    win from a rate win. Anchors are scored with the same foreground mask and
    saliency weights as PointStream, so ``beats_*_weighted_quality`` is a real
    quality comparison, not a whole-frame surrogate.

    The JSON schema is ``pointstream.modular_rate_ladder.v2`` with
    ``measurement`` set to ``encoded``. Rung objects omit ``reconstruction``.
    ``pose_oks`` is JSON null. Visuals, when requested, are
    ``write_comparison_strip`` of the source and the C1 reconstruction.

    Raises:
        MeasuredInputError: a source omits ``frames_dir`` or ``mask_path``,
            or the frames fail to load.

    A caller relies on two sources with different pixels not receiving
    identical appearance-byte and foreground-PSNR pairs, and on a report
    without a measured anchor containing no true ``beats_vvc_rate``.
    """
    del dry_run, enforce_gpu

    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise MeasuredInputError("manifest root must be an object")

    sources = _manifest_sources(manifest)
    _require_source_paths(sources)
    fg_weight, bg_weight = _saliency_weights(manifest)

    horizons: list[dict[str, Any]] = []
    for source in sources:
        horizon_id = str(source.get("horizon_id") or source.get("horizon") or source.get("id") or "")
        n_frames = _n_frames(source, manifest)
        scene = str(source.get("scene", ""))
        video = str(source.get("video") or source.get("source_video") or "")
        frames_dir = _resolve_path(source["frames_dir"])
        mask_path = _resolve_path(source["mask_path"])

        frames_rgb, mask = load_sequence(frames_dir, mask_path, n_frames)
        rungs = measure_rungs(
            frames_rgb,
            mask,
            fg_weight=fg_weight,
            bg_weight=bg_weight,
        )

        anchor_vvc: Any | None = None
        anchor_av1: Any | None = None
        if measure_anchors:
            anchor_root = Path(output_dir) / "anchors" / horizon_id
            anchor_vvc = _try_native_anchor(
                frames_rgb,
                mask,
                codec="vvc",
                qp=_VVC_QP,
                work_dir=anchor_root / "vvc",
            )
            anchor_av1 = _try_native_anchor(
                frames_rgb,
                mask,
                codec="av1",
                qp=_AV1_QP,
                work_dir=anchor_root / "av1",
            )

        rung_rows: list[dict[str, Any]] = []
        c1: MeasuredRung | None = None
        c1_beats_vvc = False
        for rung in rungs:
            beats_vvc_rate = beats(
                rung.total_bytes,
                None if anchor_vvc is None else anchor_vvc.total_bytes,
            )
            beats_av1_rate = beats(
                rung.total_bytes,
                None if anchor_av1 is None else anchor_av1.total_bytes,
            )
            if rung.rung_id == "C1_adaptive_keyframes":
                c1 = rung
                c1_beats_vvc = beats_vvc_rate
            rung_rows.append(
                _rung_to_json(
                    rung,
                    beats_vvc_rate=beats_vvc_rate,
                    beats_av1_rate=beats_av1_rate,
                    anchor_vvc=anchor_vvc,
                    anchor_av1=anchor_av1,
                )
            )

        if c1 is None:
            raise MeasuredInputError("measure_rungs did not return C1_adaptive_keyframes")

        if generate_visuals:
            Path(visuals_dir).mkdir(parents=True, exist_ok=True)
            strip_path = (
                Path(visuals_dir)
                / f"comparison_{video}_{scene}_{horizon_id}.png"
            )
            write_comparison_strip(
                frames_rgb,
                c1.reconstruction,
                strip_path,
                summary=f"{horizon_id} C1",
            )

        horizons.append(
            {
                "id": horizon_id,
                "n_frames": n_frames,
                "scene": scene,
                "source_video": video,
                "anchor_vvc_bytes": None if anchor_vvc is None else anchor_vvc.total_bytes,
                "anchor_vvc_psnr": None if anchor_vvc is None else anchor_vvc.psnr_overall,
                "anchor_vvc_psnr_fg": None if anchor_vvc is None else anchor_vvc.psnr_fg,
                "anchor_vvc_psnr_bg": None if anchor_vvc is None else anchor_vvc.psnr_bg,
                "anchor_vvc_psnr_weighted": None if anchor_vvc is None else anchor_vvc.psnr_weighted,
                "anchor_vvc_encoder_path": None if anchor_vvc is None else anchor_vvc.encoder_path,
                "anchor_vvc_encoder_version": None if anchor_vvc is None else anchor_vvc.encoder_version,
                "anchor_vvc_ffmpeg_path": None if anchor_vvc is None else anchor_vvc.ffmpeg_path,
                "anchor_vvc_ffmpeg_version": None if anchor_vvc is None else anchor_vvc.ffmpeg_version,
                "anchor_av1_bytes": None if anchor_av1 is None else anchor_av1.total_bytes,
                "anchor_av1_psnr": None if anchor_av1 is None else anchor_av1.psnr_overall,
                "anchor_av1_psnr_fg": None if anchor_av1 is None else anchor_av1.psnr_fg,
                "anchor_av1_psnr_bg": None if anchor_av1 is None else anchor_av1.psnr_bg,
                "anchor_av1_psnr_weighted": None if anchor_av1 is None else anchor_av1.psnr_weighted,
                "anchor_av1_encoder_path": None if anchor_av1 is None else anchor_av1.encoder_path,
                "anchor_av1_encoder_version": None if anchor_av1 is None else anchor_av1.encoder_version,
                "anchor_av1_ffmpeg_path": None if anchor_av1 is None else anchor_av1.ffmpeg_path,
                "anchor_av1_ffmpeg_version": None if anchor_av1 is None else anchor_av1.ffmpeg_version,
                "rungs": rung_rows,
                "summary_verdict": _summary_verdict(
                    c1,
                    beats_vvc_rate=c1_beats_vvc,
                    anchor_vvc=anchor_vvc,
                ),
            }
        )

    report: dict[str, Any] = {
        "schema": "pointstream.modular_rate_ladder.v2",
        "measurement": "encoded",
        "horizons": horizons,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="PointStream Modular Rate Ladder Runner")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--visuals-dir", type=Path, default=DEFAULT_VISUALS_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Execute without hardware GPU allocation")
    parser.add_argument("--no-gpu-check", action="store_true", help="Skip GPU preflight check")
    parser.add_argument("--generate-visuals", action="store_true", help="Generate 4-panel visual comparison strips")
    parser.add_argument(
        "--measure-anchors",
        action="store_true",
        help="Encode VVC QP46 and SVT-AV1 QP54 on the same frames",
    )

    args = parser.parse_args()

    report = run_rate_ladder(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        visuals_dir=args.visuals_dir,
        dry_run=args.dry_run,
        generate_visuals=args.generate_visuals,
        enforce_gpu=not args.no_gpu_check,
        measure_anchors=args.measure_anchors,
    )

    print("Rate Ladder Analysis Complete.")
    for h in report["horizons"]:
        print(f"\nHorizon: {h['id']} ({h['n_frames']} frames)")
        print(f"Verdict: {h['summary_verdict']}")
        for rg in h["rungs"]:
            print(
                f"  Rung {rg['rung_id']}: {rg['total_bytes']:,} B | "
                f"Weighted PSNR: {rg['psnr_weighted']} dB | "
                f"FG PSNR: {rg['psnr_fg']} dB | "
                f"BG PSNR: {rg['psnr_bg']} dB | "
                f"OKS: {rg['pose_oks']} | "
                f"Beats VVC: {rg['beats_vvc_rate']}"
            )


if __name__ == "__main__":
    main()
