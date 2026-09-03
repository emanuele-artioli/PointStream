"""Is a plate cheaper sent as a delta against a previous scene's plate?

`plans/done/BP24-findings.md` §17 closed this door on the wrong evidence. It measured
whether two scenes' first frames are *the same* (they are not: 13.75 dB) and
concluded that reuse is unavailable. But nobody proposed reusing them unchanged.
The proposal was to send a **residual against the previous plate**, and a 13.75 dB
difference says nothing about what that residual costs to code — a large but
smooth or spatially concentrated difference can code very cheaply.

So this asks the question that was actually asked:

* **fresh** — code scene B's plate on its own, intra.
* **delta** — code ``B - A`` biased into uint8, intra, at the same encoder
  setting, and reconstruct ``A + decoded_delta``.

Both are measured on what came back, and the delta arm's quality is scored on
the reconstruction rather than on the delta, so the two arms are comparable.

**What this cannot settle.** The cached windows give only two within-match
pairs, and which of them are *points* rather than replays or interludes was not
controlled — the distinction that matters for the proposal, since PointStream
would only ever be applied to points. Two pairs is not a result; it is a signal
about whether the idea deserves a properly selected experiment.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from experiments.headroom.real import load_rgb_stack
from experiments.tier.clip import BP21_CLIPS
from src.contracts import paths as ps_paths

OUT = ps_paths.outputs() / "bp24-ladder" / "plate-delta.json"

OFFSET = 128
QPS = (25, 35, 45)


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    mse = float(np.mean((ref - got) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * float(np.log10((255.0**2) / mse))


def _first_frame(video: str, scene: str) -> np.ndarray:
    pngs = sorted((BP21_CLIPS / video / scene / "window").glob("frame_*.png"))
    if not pngs:
        raise FileNotFoundError(f"{video}/{scene}")
    return load_rgb_stack(pngs[:1])[0]


def _code(frames: np.ndarray, qp: int) -> tuple[int, np.ndarray]:
    from src.components.codec.measure import coded_roundtrip
    from src.contracts.codecs import EncodeRequest, RateControl

    return coded_roundtrip(
        frames,
        request=EncodeRequest(
            codec_name="av1", rate_control=RateControl.QP, rate=qp,
            preset="10", pix_fmt="yuv420p",
        ),
    )


def compare(video: str, scene_a: str, scene_b: str) -> dict[str, Any]:
    plate_a = _first_frame(video, scene_a)
    plate_b = _first_frame(video, scene_b)
    if plate_a.shape != plate_b.shape:
        return {"pair": f"{video} {scene_a}->{scene_b}", "note": "shape mismatch"}

    rows: list[dict[str, Any]] = []
    for qp in QPS:
        fresh_bytes, fresh = _code(plate_b[np.newaxis, ...], qp)

        # The delta the decoder would receive, biased into uint8 the way the
        # residual path already does. Differences outside [-128, 127] clip, and
        # that clipping is part of what the arm costs.
        signed = plate_b.astype(np.int16) - plate_a.astype(np.int16)
        biased = np.clip(signed + OFFSET, 0, 255).astype(np.uint8)
        delta_bytes, decoded_delta = _code(biased[np.newaxis, ...], qp)
        rebuilt = np.clip(
            plate_a.astype(np.int16) + (decoded_delta[0].astype(np.int16) - OFFSET),
            0, 255,
        ).astype(np.uint8)

        rows.append(
            {
                "qp": qp,
                "fresh_bytes": int(fresh_bytes),
                "fresh_psnr_dB": _psnr(plate_b, fresh[0]),
                "delta_bytes": int(delta_bytes),
                "delta_psnr_dB": _psnr(plate_b, rebuilt),
                "delta_over_fresh": round(delta_bytes / max(1, fresh_bytes), 3),
            }
        )
    return {
        "pair": f"{video} {scene_a} -> {scene_b}",
        "plate_similarity_dB": _psnr(plate_a, plate_b),
        "rows": rows,
    }


def main() -> int:
    pairs = [
        ("alcaraz_highlights", "scene_000", "scene_010"),
        ("federer_djokovic", "scene_001", "scene_003"),
    ]
    results = []
    for video, a, b in pairs:
        try:
            results.append(compare(video, a, b))
        except FileNotFoundError as exc:
            results.append({"pair": f"{video} {a}->{b}", "error": repr(exc)})

    payload = {
        "question": (
            "is a plate cheaper sent as a delta against a previous scene's "
            "plate than sent fresh?"
        ),
        "why_findings_17_did_not_answer_it": (
            "It measured whether two plates are identical, not what the "
            "difference costs to code. A 13.75 dB gap can still be a cheap "
            "residual if it is smooth or spatially concentrated."
        ),
        "scope": (
            "Two within-match pairs from the cached BP21 windows. Whether each "
            "scene is a POINT rather than a replay or interlude was not "
            "controlled, and that is the distinction the proposal rests on. "
            "This is a signal, not a result."
        ),
        "arms": {
            "fresh": "code scene B's plate intra, av1 preset 10",
            "delta": "code (B - A + 128) intra at the same QP; quality scored on A + decoded delta",
        },
        "pairs": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for block in results:
        print(f"== {block['pair']}  plates differ by {block.get('plate_similarity_dB', float('nan')):.2f} dB")
        for row in block.get("rows", []):
            print(
                f"   qp{row['qp']:>3}  fresh {row['fresh_bytes']:>8,} B @ {row['fresh_psnr_dB']:5.2f} dB"
                f"   delta {row['delta_bytes']:>8,} B @ {row['delta_psnr_dB']:5.2f} dB"
                f"   ratio {row['delta_over_fresh']}",
                flush=True,
            )
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
