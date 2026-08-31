"""Two cheap questions about the plate, which is 88-91% of PointStream's rate.

Both come out of `plans/BP24-ladder-report.md`, and both are asked before any
code is written for them, because either answer redirects the plate work.

**1. Is JPEG the wrong codec for a 4K still?** At `jpeg:30` the plate cost
283,483 B while av1 coded the *entire eight-frame clip* for 85,995 B at higher
quality. A modern intra frame is not a JPEG. If av1-intra or vvc-intra codes the
same plate at a fraction of the size and the same fidelity, that is a factor on
90% of the payload for no architectural change at all.

**2. Do scenes from one match share a background?** If the camera returns to the
same court view, one plate could be amortised across scenes — and unlike a
codec, which must start a fresh intra frame at every cut, PointStream could
carry it across. This measures whether the premise holds before anyone builds
on it: how close is scene B's first frame to scene A's?
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from experiments.headroom.real import load_rgb_stack
from experiments.tier.clip import BP21_CLIPS
from src.contracts import paths as ps_paths

OUT = ps_paths.outputs() / "bp24-ladder" / "plate-probe.json"

#: Quality knobs to sweep, per route. JPEG quality and codec QP are different
#: scales; the comparison is made at matched *fidelity*, not matched knob.
JPEG_QUALITIES = (10, 30, 50, 75, 90)
CODEC_QPS = (25, 35, 45, 55)


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    mse = float(np.mean((ref - got) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * float(np.log10((255.0**2) / mse))


def _first_frame(video: str, scene: str) -> np.ndarray:
    window = BP21_CLIPS / video / scene / "window"
    pngs = sorted(window.glob("frame_*.png"))
    if not pngs:
        raise FileNotFoundError(f"no cached window at {window}")
    return load_rgb_stack(pngs[:1])[0]


def jpeg_versus_intra(plate: np.ndarray) -> dict[str, Any]:
    """Same still, two routes, both measured on what came back."""
    import cv2

    from src.components.codec.measure import coded_roundtrip
    from src.contracts.codecs import EncodeRequest, RateControl

    rows: list[dict[str, Any]] = []

    for quality in JPEG_QUALITIES:
        ok, buf = cv2.imencode(".jpg", plate[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            continue
        payload = buf.tobytes()
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        rows.append(
            {
                "route": "jpeg",
                "knob": f"q{quality}",
                "bytes": len(payload),
                "psnr_dB": _psnr(plate, decoded),
            }
        )

    for codec_name, preset in (("av1", "10"), ("vvc", "faster")):
        for qp in CODEC_QPS:
            try:
                coded_bytes, decoded = coded_roundtrip(
                    plate[np.newaxis, ...],
                    request=EncodeRequest(
                        codec_name=codec_name,
                        rate_control=RateControl.QP,
                        rate=qp,
                        preset=preset,
                        pix_fmt="yuv420p",
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
                rows.append({"route": codec_name, "knob": f"qp{qp}", "error": repr(exc)})
                continue
            rows.append(
                {
                    "route": codec_name,
                    "knob": f"qp{qp}",
                    "preset": preset,
                    "bytes": int(coded_bytes),
                    "psnr_dB": _psnr(plate, decoded[0]),
                }
            )
    return {"rows": rows}


def cross_scene(video: str, scenes: tuple[str, ...]) -> dict[str, Any]:
    """How close is one scene's first frame to another's, in the same match?"""
    frames = {}
    for scene in scenes:
        try:
            frames[scene] = _first_frame(video, scene)
        except FileNotFoundError:
            continue
    if len(frames) < 2:
        return {"video": video, "note": "fewer than two cached scenes"}
    names = sorted(frames)
    base = frames[names[0]]
    out: list[dict[str, Any]] = []
    for other in names[1:]:
        candidate = frames[other]
        if candidate.shape != base.shape:
            out.append({"pair": f"{names[0]} vs {other}", "note": "shape mismatch"})
            continue
        out.append(
            {
                "pair": f"{names[0]} vs {other}",
                "psnr_dB": _psnr(base, candidate),
                "mean_abs_diff": float(
                    np.abs(base.astype(np.int16) - candidate.astype(np.int16)).mean()
                ),
            }
        )
    return {"video": video, "pairs": out}


def main() -> int:
    plate = _first_frame("alcaraz_highlights", "scene_000")
    payload: dict[str, Any] = {
        "question_1": (
            "is JPEG the wrong codec for a 4K plate? Compared against av1-intra "
            "and vvc-intra on the same still, matched on fidelity not on knob."
        ),
        "plate": {
            "clip": "alcaraz_highlights/scene_000 frame 0",
            "shape": list(plate.shape),
            "raw_bytes": int(plate.nbytes),
        },
        "jpeg_versus_intra": jpeg_versus_intra(plate),
        "question_2": (
            "do scenes from one match share a background? A codec must start a "
            "fresh intra frame at every cut; a plate would not have to."
        ),
        "cross_scene": [
            cross_scene("alcaraz_highlights", ("scene_000", "scene_010")),
            cross_scene("federer_djokovic", ("scene_001", "scene_003")),
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in payload["jpeg_versus_intra"]["rows"]:
        print(
            f"  {row['route']:>4} {row['knob']:>5}  "
            f"{row.get('bytes', -1):>9,} B  {row.get('psnr_dB', float('nan')):6.2f} dB"
            f"{'  ' + row['error'][:60] if 'error' in row else ''}",
            flush=True,
        )
    for block in payload["cross_scene"]:
        for pair in block.get("pairs", []):
            print(
                f"  {block['video']}  {pair['pair']}  "
                f"{pair.get('psnr_dB', float('nan')):.2f} dB  "
                f"MAD {pair.get('mean_abs_diff', float('nan')):.2f}",
                flush=True,
            )
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
