"""Does the Canny score rank references the way real encodes do?

`best-scored` picks a reference by comparing **edge maps**, not pixels. Brief §3
gives the reasoning: findings §18 measured the pair *further apart* in PSNR
(federer, 15.10 dB) saving more than the closer one (alcaraz, 13.75 dB), so
pixel distance does not predict coding distance -- what a codec spends bits on
is residual structure after motion compensation.

That is a reason to prefer an edge score over an MSE. It is **not** evidence
that this particular edge score works, and a proxy that has never been checked
against the thing it proxies is how a search confidently picks the wrong
reference. So this runs both: for every candidate reference of a target scene,
the Canny IoU *and* the real trial encode, and asks whether they rank the same
way.

**What the answer changes.** If they agree, `best-scored` is worth its edge pass.
If they do not, brief §3 is explicit about the fallback: say so and recommend
`first`, which costs nothing and is already worth 31-53%.

Run: ``python -m experiments.tier.canny_validate``
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np

from experiments.tier.scene_plates import extract_plates, list_scenes, load_plates
from src.components.background.stream import canny_iou, encode_chain, ffmpeg_provenance
from src.contracts import paths as ps_paths

OUT_PATH = ps_paths.outputs() / "bp30-background" / "canny-validation.json"


def spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation, ties averaged. Written out to avoid a scipy dependency.

    Returns ``None`` for fewer than three points or a constant series, where a
    correlation is not defined rather than zero.
    """
    if len(a) < 3 or len(a) != len(b):
        return None
    ranks_a, ranks_b = _ranks(a), _ranks(b)
    if len(set(ranks_a)) < 2 or len(set(ranks_b)) < 2:
        return None
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2 + 1
        for index in order[position : end + 1]:
            ranks[index] = shared
        position = end + 1
    return ranks


def trial_encode(reference: np.ndarray, target: np.ndarray, *, codec: str, crf: int) -> int:
    """Marginal bytes for ``target`` coded as a P-frame against ``reference``.

    This is the quantity `best-scored` is trying to minimise, measured rather
    than predicted -- the same two-frame measurement findings §18 and §19 used,
    so the numbers sit on the same axis as the 31-53% already recorded.
    """
    encoded = encode_chain([reference, target], codec=codec, crf=crf)
    if encoded.picture_types != ("I", "P"):
        raise RuntimeError(
            f"expected IP, got {''.join(encoded.picture_types)}; the trial is not measuring inter"
        )
    return encoded.marginal_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="alcaraz_highlights")
    parser.add_argument("--codec", default="av1")
    parser.add_argument("--crf", type=int, default=38)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument(
        "--targets", type=int, default=4,
        help="how many of the later scenes to score every earlier scene against",
    )
    args = parser.parse_args()

    scenes = list_scenes(args.video)[: args.scenes]
    plates = load_plates(extract_plates(args.video, scenes, height=args.height))
    print(f"{args.video}: {len(plates)} plates {plates[0].shape}", flush=True)

    started = time.time()
    rows: list[dict[str, Any]] = []
    pooled_scores: list[float] = []
    pooled_bytes: list[float] = []

    # Targets are the later scenes, so each has several candidates to choose
    # between. A target with one candidate cannot disagree about a ranking.
    first_target = max(2, len(plates) - args.targets)
    for target in range(first_target, len(plates)):
        candidates: list[dict[str, Any]] = []
        for reference in range(target):
            score = canny_iou(plates[target], plates[reference])
            marginal = trial_encode(plates[reference], plates[target], codec=args.codec, crf=args.crf)
            candidates.append({"reference": reference, "canny_iou": round(score, 5), "bytes": marginal})
            pooled_scores.append(score)
            pooled_bytes.append(float(marginal))
            print(
                f"  target {target:>2} <- ref {reference:>2}: iou={score:.4f} "
                f"marginal={marginal:>9,} B",
                flush=True,
            )
        scores = [c["canny_iou"] for c in candidates]
        sizes = [float(c["bytes"]) for c in candidates]
        # A better reference means a *higher* IoU and *fewer* bytes, so a proxy
        # that works shows a negative correlation. Negated here so "agrees"
        # reads positive.
        rho = spearman(scores, sizes)
        best_by_score = min(candidates, key=lambda c: -float(c["canny_iou"]))
        best_by_encode = min(candidates, key=lambda c: float(c["bytes"]))
        penalty = (
            float(best_by_score["bytes"]) / float(best_by_encode["bytes"])
            if best_by_encode["bytes"] else None
        )
        rows.append(
            {
                "target": target,
                "candidates": candidates,
                "spearman_agreement": round(-rho, 4) if rho is not None else None,
                "canny_pick": best_by_score["reference"],
                "oracle_pick": best_by_encode["reference"],
                "picks_match": best_by_score["reference"] == best_by_encode["reference"],
                # What choosing by Canny costs against choosing by trial encode.
                # 1.0 means the proxy found the best reference available.
                "canny_pick_cost_over_oracle": round(penalty, 4) if penalty else None,
            }
        )
        print(
            f"  -> target {target}: agreement={rows[-1]['spearman_agreement']} "
            f"canny picked {best_by_score['reference']}, oracle {best_by_encode['reference']}, "
            f"cost x{rows[-1]['canny_pick_cost_over_oracle']}",
            flush=True,
        )

    pooled = spearman(pooled_scores, pooled_bytes)
    agreements = [r["spearman_agreement"] for r in rows if r["spearman_agreement"] is not None]
    penalties = [
        r["canny_pick_cost_over_oracle"] for r in rows
        if r["canny_pick_cost_over_oracle"] is not None
    ]
    verdict = (
        "Canny does not track trial encodes; brief §3 says recommend `first`"
        if not agreements or statistics.fmean(agreements) < 0.3
        else "Canny ranks references broadly as trial encodes do"
    )
    payload = {
        "question": "does the Canny edge score rank candidate references the way real encodes do?",
        "reading": (
            "`spearman_agreement` is +1 when a higher edge IoU always means fewer "
            "coded bytes and -1 when it always means more. "
            "`canny_pick_cost_over_oracle` is what picking by Canny costs against "
            "picking by trial encode: 1.0 means the proxy found the best reference."
        ),
        "video": args.video,
        "ffmpeg": ffmpeg_provenance(),
        "codec": args.codec,
        "crf": args.crf,
        "plate_height": args.height,
        "n_targets": len(rows),
        "n_pairs": len(pooled_scores),
        "pooled_spearman_agreement": round(-pooled, 4) if pooled is not None else None,
        "mean_spearman_agreement": round(statistics.fmean(agreements), 4) if agreements else None,
        "targets_where_canny_picked_the_oracle_reference": sum(1 for r in rows if r["picks_match"]),
        "mean_canny_pick_cost_over_oracle": round(statistics.fmean(penalties), 4) if penalties else None,
        "verdict": verdict,
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n{verdict}", flush=True)
    print(f"wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
