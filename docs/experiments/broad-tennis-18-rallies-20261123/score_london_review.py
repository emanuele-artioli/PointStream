"""Score paid London ball displays on the frozen original-frame author review."""

import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch

from heldout_display_metrics import HERE, decode, digest
from score_tracknet_ball import (
    ALEX_SHA256,
    EXPECTED,
    WEIGHTS,
    image_psnr,
    quality,
    region_scores,
    sha,
)

from experiments.tennis.k5c_independent_ball import render_frames
from src.pointstream import ball, ball_stream


def read_js(line):
    return json.loads(line.split("=", 1)[1].strip().removesuffix(";"))


def inputs(review, export):
    packet = read_js((review / "frames.js").read_text())
    proposal_file = review / "proposals/person-proposals.js"
    masks_js, provenance_js = proposal_file.read_text().splitlines()
    proposals = read_js(masks_js)
    decisions = json.loads(export.read_text())
    assert decisions["schema"] == "pointstream-tenniset-source-ball-first-v2"
    assert decisions["manifest"] == packet["manifest"]
    assert decisions["person_proposal_source"] == read_js(provenance_js)
    assert decisions["frame_size"] == [1280, 720]
    frozen = json.loads((HERE / "source-marks.json").read_text())["frames"]
    assert len(frozen) == len(packet["frames"]) == len(decisions["frames"]) == 48
    assert len({row["id"] for row in decisions["frames"]}) == 48
    rows = {}
    for record, original, reviewed in zip(
        frozen, packet["frames"], decisions["frames"], strict=True
    ):
        key = f"{record['rally_id']}:{record['tick']}"
        assert key == original["id"] == reviewed["id"]
        png = review / original["src"]
        assert sha(png) == original["source_sha256"] == reviewed["source_sha256"]
        rgb = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
        assert rgb.shape == (720, 1280, 3) and digest(rgb) == record["rgb_sha256"]
        candidates = proposals[key]
        assert reviewed["person_candidates"] == [
            {field: p[field] for field in ("id", "mask_sha256", "box")} for p in candidates
        ]
        accepts = []
        for proposal in candidates:
            path = review / proposal["mask"]
            assert sha(path) == proposal["mask_sha256"]
            if reviewed["proposals"].get(proposal["id"]) == "accept":
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
                assert mask.shape == (720, 1280) and mask.any()
                accepts.append(mask)
        mark = reviewed["ball"]
        assert mark["kind"] in ("visible", "absent", "unsure")
        if mark["kind"] == "visible":
            x, y = mark["xy"]
            assert 0 <= x < 1280 and 0 <= y < 720
        else:
            assert mark["xy"] is None
        rows.setdefault(record["rally_id"], []).append((record, rgb, accepts, mark))
    return rows


def score_regions(source, displayed, masks, mark, models):
    scores = {"players": []}
    foreground = np.zeros(source.shape[:2], bool)
    for mask in masks:
        ys, xs = np.where(mask)
        window = np.s_[
            max(0, ys.min() - 32) : ys.max() + 33,
            max(0, xs.min() - 32) : xs.max() + 33,
        ]
        scores["players"].append(
            region_scores(source[window], displayed[window], mask[window], *models)
        )
        foreground |= mask
    if mark["kind"] == "visible":
        x, y = mark["xy"]
        left = max(0, min(1280 - 64, int(x) - 32))
        top = max(0, min(720 - 64, int(y) - 32))
        window = np.s_[top : top + 64, left : left + 64]
        foreground[window] = True
        scores["ball"] = region_scores(
            source[window], displayed[window], np.ones((64, 64), bool), *models
        )
    scores["remainder_psnr"] = image_psnr(source, displayed, ~foreground)
    return scores


def run(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = inputs(args.review, args.export)
    alex = WEIGHTS.parent / "alexnet-owt-7be5be79.pth"
    assert sha(alex) == ALEX_SHA256 and sha(WEIGHTS) == EXPECTED
    models = quality.load_dists(WEIGHTS, "cpu"), quality.load_lpips(alex, "cpu")
    bank = json.loads((HERE / "tenniset-ball-summary.json").read_text())["rows"]
    tight = json.loads((HERE / "heldout-rate-recheck.json").read_text())["results"]
    tight = {(r["rally"], str(r["ceiling_bps"])): r for r in tight}
    scored = []
    with TemporaryDirectory() as scratch_dir:
        scratch = Path(scratch_dir)
        for fixture in bank[:1] if args.smoke else bank:
            name, n = fixture["rally_id"], fixture["frames"]
            for price, selected in fixture["selected"].items():
                width = selected["coded_size"][0]
                packet = args.artifacts / name / f"{width}-{selected['ceiling']}.psbl"
                assert sha(packet) == selected["sha256"]
                stream = ball_stream.unpack(packet.read_bytes())
                assert stream.frames == n and stream.fps == Fraction(25)
                base = decode(stream.native, selected["coded_size"], n, scratch)
                points, looks = ball.decode_frames(stream.ball_payload, n, 960, 544)
                rendered = render_frames(base, points, looks)
                assert digest(rendered) == selected["displayed_sha256"]
                displays = {"same_base": base, "ball_stream": rendered}
                matched = tight.get((name, price))
                if matched:
                    control = (
                        args.artifacts
                        / "tightened-whole-picture"
                        / f"london-{name}-{price}-closest-av1.ivf"
                    )
                    assert sha(control) == matched["selected"]["sha256"]
                    displays["matched_av1"] = decode(
                        control.read_bytes(), selected["coded_size"], n, scratch
                    )
                for record, source, masks, mark in rows[name]:
                    tick = record["tick"]
                    point = points[tick]
                    error = None
                    if point is not None and mark["kind"] == "visible":
                        error = math.dist(mark["xy"], (point[0] * 1280 / 960, point[1] * 720 / 544))
                    scored.append(
                        {
                            "rally": name,
                            "tick": tick,
                            "route": record["route"],
                            "ceiling_bps": int(price),
                            "mark": mark["kind"],
                            "assisted": mark["assisted"],
                            "emitted": point is not None,
                            "error_px": error,
                            "accepted_masks": len(masks),
                            "scores": {
                                arm: score_regions(
                                    source,
                                    cv2.resize(
                                        frames[tick], (1280, 720), interpolation=cv2.INTER_CUBIC
                                    ),
                                    masks,
                                    mark,
                                    models,
                                )
                                for arm, frames in displays.items()
                            },
                        }
                    )
            print(name, len(scored), flush=True)
    annotations = [r for r in scored if r["ceiling_bps"] == 20400]
    counts = Counter(
        f"{r['mark']}_{'assisted' if r['assisted'] else 'independent'}" for r in annotations
    )
    args.out.write_text(
        json.dumps(
            {
                "review_sha256": sha(args.export),
                "scorer_sha256": sha(Path(__file__)),
                "ball_delivery_summary_sha256": sha(HERE / "tenniset-ball-summary.json"),
                "matched_controls_sha256": sha(HERE / "heldout-rate-recheck.json"),
                "counts": counts,
                "accepted_masks": sum(r["accepted_masks"] for r in annotations),
                "model_sha256": {"DISTS": sha(WEIGHTS), "LPIPS": sha(alex)},
                "rows": scored,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("review", "export", "artifacts", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
