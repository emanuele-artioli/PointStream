"""What does a stitched panorama cost, and what does it save? (BP29, stream D)

`build_plate` has existed since the rewrite and the runner never called it, so
`background.method` selected a *transmission strategy* over the first source
frame and a panorama's whole argument — amortising one background across the
clip — had never been available. `make_background` now calls it.

This module measures the trade the wiring exposes. Two arms, one code path:

* **keyframe (control)** — `make_background(ctx, span=1)`. `build_plate` over a
  one-frame span is an identity warp and a median of one sample, so it returns
  that frame unchanged. This reproduces exactly what the runner transmitted
  before BP29, which is what makes it a control rather than a second
  implementation.
* **panorama** — `make_background(ctx)`, the shipped default: the whole chunk,
  players excluded from the median, plus the homographies to warp it back.
* **panorama, no registration (control)** — `make_background(ctx,
  register=False)`. The same median over the same span with every homography
  forced to the identity. A plate does two separable things: it compensates
  camera motion, and it averages away whatever differs between frames. This arm
  keeps only the second, so a win cannot be credited to registration without
  showing that removing registration takes it away.

Three further checks travel with the numbers, because the result on the moving
clip is a large one and a large result is exactly where a missing control gets
skipped:

* the **degenerate control** — the `span=1` arm's plate must be the first source
  frame, byte for byte, which is what makes it the pre-BP29 behaviour rather
  than an approximation of it;
* the **plate-bytes decomposition** — the panorama plate costs about the same as
  a single frame while covering more area, and that needs attributing rather
  than asserting: the same span is encoded warped, unwarped, and cropped back to
  frame size, so area and denoising can be told apart;
* the **span actually amortised over**, recorded as a resolved integer rather
  than as `None`.

Two clips, chosen as the ends of the motion range in the eight cached BP21
windows (`plans/done/BP24-findings.md` §11): `alcaraz_highlights/scene_000`
(inter-frame MAD 0.33) and `federer_djokovic/scene_003` (7.70). A *different*
video spelled `djokovic_federer` exists in both trees and is not this one; the
directory actually read is recorded in the output so the label is never the
evidence.

**One clip per motion regime is not a corpus.** Nothing here supports a claim
about content in general.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.clip import BP21_CLIPS, TierClip, load_tier_clip
from experiments.headroom.real import DATASET
from src.components.codec.frames import rgb_to_luma
from src.contracts import paths as ps_paths
from src.contracts.config import PointstreamConfig
from src.contracts.lattice import ART_BACKGROUND_MODEL, STAGE_BACKGROUND
from src.pipeline.reconstruction.background import BackgroundModelView, BackgroundResolver
from src.runner import RunResult, run
from src.runner.config_io import load_tier
from src.runner.routing import bind_evaluator, generation_params
from src.runner.stages import StageContext, make_background

OUT_DIR = ps_paths.outputs() / "bp29-panorama"

#: The two ends of the motion range in the cached windows, by inter-frame MAD.
CLIPS: tuple[tuple[str, str, str, float], ...] = (
    ("static", "alcaraz_highlights", "scene_000", 0.33),
    ("dynamic", "federer_djokovic", "scene_003", 7.70),
)


def pooled_luma_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """One BT.601 Y-PSNR over the whole clip's MSE — BP24's axis, restated.

    Restated rather than imported: `experiments.tier.ladder` owns the ladder
    and this stream must not touch it, and importing it drags ~37 s of module
    load for four lines of arithmetic.
    """
    ref = rgb_to_luma(np.asarray(reference, dtype=np.float64)).astype(np.float64)
    got = rgb_to_luma(np.asarray(candidate, dtype=np.float64)).astype(np.float64)
    if ref.shape != got.shape:
        raise ValueError(f"psnr shape mismatch: {ref.shape} vs {got.shape}")
    mse = float(np.mean((ref - got) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * float(np.log10((255.0**2) / mse))


def _context(config: PointstreamConfig) -> StageContext:
    """A StageContext for binding one stage outside `run`.

    `make_background`'s stage reads only `ctx.config`, but StageContext is
    constructed whole so a field the runner starts passing shows up as a type
    error here rather than as a silently different control arm.
    """
    return StageContext(
        lattice=config.stages,
        residual=config.residual,
        generator=None,
        evaluator=bind_evaluator(None, config),
        resolver=BackgroundResolver(),
        seed=config.run.seed,
        params=generation_params(config),
        config=config,
    )


def _view(result: RunResult) -> BackgroundModelView:
    bag = result.chunks[0].bag
    view = bag.get(ART_BACKGROUND_MODEL) or bag.get(STAGE_BACKGROUND)
    if not isinstance(view, BackgroundModelView):
        raise AssertionError(
            "the background stage did not put a BackgroundModelView on the bag; "
            "there is nothing to attribute a plate cost to"
        )
    return view


def independent_sidecar_bytes(
    config: PointstreamConfig, clip: TierClip, *, span: int | None, register: bool = True
) -> int:
    """Encode the same plate outside the runner and return its length.

    Stream A's cross-check: the runner's `sizes.panorama` must equal the sidecar
    bytes measured independently. A disagreement is a bug in the wiring, not a
    finding about panoramas.
    """
    from src.components.background.plate import build_plate
    from src.components.background.sidecar import build_sidecar

    count = clip.frames.shape[0] if span is None else min(span, clip.frames.shape[0])
    masks = None if count < 2 else clip.union_mask[:count].astype(np.uint8)
    plate, _ = build_plate(
        np.asarray(clip.frames[:count], dtype=np.uint8), masks=masks, register=register
    )
    sidecar = build_sidecar(
        config.background.codec, jpeg_quality=config.background.jpeg_quality
    )
    return len(sidecar.encode(plate))


def run_arm(
    *,
    name: str,
    tier: str,
    clip: TierClip,
    span: int | None,
    register: bool = True,
    order: int = 0,
) -> dict[str, Any]:
    """One tier config on one clip with the background span and warp pinned."""
    config = load_tier(tier)
    ctx = _context(config)
    resolved_span = int(clip.frames.shape[0] if span is None else min(span, clip.frames.shape[0]))
    started = time.time()
    result = run(
        config,
        [clip.frames],
        backends={STAGE_BACKGROUND: make_background(ctx, span=span, register=register)},
        objects=(clip.objects,),
    )
    seconds = time.time() - started
    view = _view(result)
    plate = np.asarray(view.plate)
    delivered = result.delivered_frames
    sizes = result.sizes
    independent = independent_sidecar_bytes(config, clip, span=span, register=register)
    return {
        "arm": name,
        "tier": tier,
        "background_span_requested": span,
        "background_span_resolved_frames": resolved_span,
        "registration": register,
        "arm_order_in_process": order,
        "wall_clock_seconds": round(seconds, 1),
        "wall_clock_note": (
            "arm_order_in_process 0 pays this process's one-off model load and "
            "import cost; it is not a property of the arm"
        ),
        "background": {
            "method": config.background.method,
            "codec": config.background.codec,
            "jpeg_quality": config.background.jpeg_quality,
            "mode": view.mode,
            "plate_shape": [int(v) for v in plate.shape],
            "canvas_wh": [int(plate.shape[1]), int(plate.shape[0])],
            "frame_wh": [int(clip.frames.shape[2]), int(clip.frames.shape[1])],
            "plate_area_ratio": round(
                float(plate.shape[0] * plate.shape[1])
                / float(clip.frames.shape[1] * clip.frames.shape[2]),
                4,
            ),
            "n_homographies": len(view.homographies),
            "homographies_all_identity": bool(
                all(
                    np.allclose(np.asarray(h, dtype=np.float64).reshape(3, 3), np.eye(3), atol=1e-7)
                    for h in view.homographies
                )
            ),
        },
        "sizes_bytes": result.sizes_bytes,
        "ledger_cross_check": {
            "runner_sizes_panorama": int(sizes.panorama),
            "independently_encoded_sidecar_bytes": int(independent),
            "agree": bool(int(sizes.panorama) == int(independent)),
        },
        "delivered_y_psnr_dB": round(pooled_luma_psnr(clip.frames, delivered), 4),
        "delivered_rgb_psnr_dB": round(_rgb_psnr(clip.frames, delivered), 4),
        "reconstruction_y_psnr_dB": round(
            pooled_luma_psnr(clip.frames, result.frames), 4
        ),
    }


def _rgb_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    mse = float(np.mean((ref - got) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * float(np.log10((255.0**2) / mse))


def plate_bytes_decomposition(config: PointstreamConfig, clip: TierClip) -> dict[str, Any]:
    """Why does a plate covering more area not cost more?

    The headline needs this. On the moving clip the panorama covers 1.8% more
    area than one frame and codes to 0.6% *fewer* bytes, and "more coverage for
    free" is a claim that has to be attributed rather than enjoyed. Four
    encodes of the same content separate the two effects:

    * ``keyframe`` — the single source frame, the thing being replaced;
    * ``median_no_warp`` — the same span median-composited with no
      registration, on a frame-sized canvas. Same pixel count as the keyframe,
      so any difference is the median averaging away sensor noise, dither and
      the masked players — not area;
    * ``panorama_cropped`` — the registered panorama cut back to frame size, so
      it is comparable pixel-for-pixel with the two above;
    * ``panorama`` — the whole registered canvas, which is what is transmitted.

    ``panorama`` minus ``panorama_cropped`` is what the extra coverage costs.
    ``median_no_warp`` minus ``keyframe`` is what the median saves.
    """
    from src.components.background.plate import build_plate
    from src.components.background.sidecar import build_sidecar

    frames = np.asarray(clip.frames, dtype=np.uint8)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    masks = clip.union_mask.astype(np.uint8)
    sidecar = build_sidecar(
        config.background.codec, jpeg_quality=config.background.jpeg_quality
    )
    keyframe, _ = build_plate(frames[:1])
    no_warp, _ = build_plate(frames, masks=masks, register=False)
    panorama, _ = build_plate(frames, masks=masks)
    cropped = panorama[:height, :width]
    sizes = {
        "keyframe": len(sidecar.encode(keyframe)),
        "median_no_warp": len(sidecar.encode(no_warp)),
        "panorama_cropped_to_frame": len(sidecar.encode(cropped)),
        "panorama": len(sidecar.encode(panorama)),
    }
    return {
        "codec": sidecar.codec_id,
        "bytes": sizes,
        "shapes": {
            "keyframe": [int(v) for v in keyframe.shape],
            "median_no_warp": [int(v) for v in no_warp.shape],
            "panorama": [int(v) for v in panorama.shape],
        },
        "median_saves_bytes": sizes["keyframe"] - sizes["median_no_warp"],
        "extra_coverage_costs_bytes": sizes["panorama"] - sizes["panorama_cropped_to_frame"],
        "degenerate_control": {
            "question": "is a one-frame span the pre-BP29 plate, exactly?",
            "plate_equals_first_source_frame": bool(np.array_equal(keyframe, frames[0])),
            "bytes": sizes["keyframe"],
        },
    }


def _ratio(new: float, old: float) -> float | None:
    return None if old == 0 else round(float(new) / float(old), 4)


def compare(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    cs, ts = control["sizes_bytes"], treatment["sizes_bytes"]
    return {
        "plate_bytes": [cs["panorama"], ts["panorama"]],
        "plate_ratio_panorama_over_keyframe": _ratio(ts["panorama"], cs["panorama"]),
        "residual_bytes": [cs["residual"], ts["residual"]],
        "residual_ratio_panorama_over_keyframe": _ratio(ts["residual"], cs["residual"]),
        "transport_total_bytes": [cs["transport_total"], ts["transport_total"]],
        "total_ratio_panorama_over_keyframe": _ratio(
            ts["transport_total"], cs["transport_total"]
        ),
        "delivered_y_psnr_dB": [
            control["delivered_y_psnr_dB"],
            treatment["delivered_y_psnr_dB"],
        ],
        "delivered_y_psnr_delta_dB": round(
            treatment["delivered_y_psnr_dB"] - control["delivered_y_psnr_dB"], 4
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", nargs="*", default=["fast", "balanced"])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--out", default=str(OUT_DIR / "report.json"))
    args = parser.parse_args(argv)

    records: list[dict[str, Any]] = []
    order = 0
    for label, video, scene, mad in CLIPS:
        clip = load_tier_clip(video=video, scene=scene, n_frames=args.frames)
        provenance = {
            "label": label,
            "inter_frame_MAD_from_BP24_findings_11": mad,
            "window_dir_read": str(BP21_CLIPS / video / scene / "window"),
            "dataset_dir_read": str(DATASET / video / "segmentations" / scene),
            "clip": clip.describe(),
        }
        print(f"clip {label} {video}/{scene} {clip.describe()}", flush=True)
        for tier in args.tiers:
            arms = []
            for name, span, register in (
                ("keyframe (control)", 1, True),
                ("panorama", None, True),
                ("panorama, no registration (control)", None, False),
            ):
                arm = run_arm(
                    name=name,
                    tier=tier,
                    clip=clip,
                    span=span,
                    register=register,
                    order=order,
                )
                order += 1
                arms.append(arm)
                print(
                    f"  {tier:9s} {name:36s} {arm['wall_clock_seconds']:7.1f}s  "
                    f"span={arm['background_span_resolved_frames']}  "
                    f"canvas={arm['background']['canvas_wh']}  "
                    f"plate={arm['sizes_bytes']['panorama']} B  "
                    f"residual={arm['sizes_bytes']['residual']} B  "
                    f"Y={arm['delivered_y_psnr_dB']} dB",
                    flush=True,
                )
            control, treatment, unregistered = arms
            records.append(
                {
                    "clip": provenance,
                    "tier": tier,
                    "arms": arms,
                    "trade": compare(control, treatment),
                    "registration_control": compare(unregistered, treatment),
                    "registration_control_reads": (
                        "panorama against the same median with no registration. "
                        "A residual ratio well below 1 here is registration "
                        "doing the work; a ratio near 1 would mean the win came "
                        "from the temporal median and the homographies are "
                        "decoration."
                    ),
                    "plate_bytes_decomposition": plate_bytes_decomposition(
                        load_tier(tier), clip
                    ),
                }
            )
            destination = Path(args.out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(
                    {
                        "brief": "BP29 stream D — the panorama the runner never called",
                        "bounds_written_before_measurement": (
                            "outputs/bp29-panorama/bounds-before-run.json"
                        ),
                        "scope": (
                            "One clip per motion regime, eight frames, one still-image "
                            "codec per tier. Eight frames is the least favourable "
                            "amortisation a fixed plate cost can get. Not a corpus."
                        ),
                        "results": records,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"  wrote {destination}", flush=True)
    return 0


__all__ = [
    "CLIPS",
    "OUT_DIR",
    "compare",
    "main",
    "plate_bytes_decomposition",
    "pooled_luma_psnr",
    "run_arm",
]


if __name__ == "__main__":
    raise SystemExit(main())
