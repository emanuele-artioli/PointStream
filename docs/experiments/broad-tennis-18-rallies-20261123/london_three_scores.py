"""Score three fixed London paid displays and size-matched ordinary controls."""
from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import torch

from heldout_display_metrics import decode, digest
from score_london_review import inputs, score_regions
from score_tracknet_ball import ALEX_SHA256, EXPECTED, WEIGHTS, quality, sha
from experiments.tennis.k5c_independent_ball import render_frames
from src.pointstream import ball, ball_stream

ROOT = Path("/private/tmp/ps-london-three-match-20261123")
REVIEW = Path("/Users/manu/.codex/visualizations/2026/11/18/01a1edc1-dd4e-7ab3-b819-fbd88da8db05/tenniset-ball-first-review-20261123")
EXPORT = Path("/Users/manu/Downloads/tenniset-ball-first-decisions.json")
HERE = Path(__file__).resolve().parent


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source_rows = inputs(REVIEW, EXPORT)
    candidates = {r["rally_id"]: r for r in json.loads((HERE / "tenniset-ball-summary.json").read_text())["rows"]}
    controls = json.loads((ROOT / "result.json").read_text())
    alex = WEIGHTS.parent / "alexnet-owt-7be5be79.pth"
    assert sha(alex) == ALEX_SHA256 and sha(WEIGHTS) == EXPECTED
    models = quality.load_dists(WEIGHTS, "cpu"), quality.load_lpips(alex, "cpu")
    rows = []
    with TemporaryDirectory(prefix="ps-london-three-score-") as temp:
        scratch = Path(temp)
        for result in controls["results"]:
            name = result["rally"]
            spec = candidates[name]
            selected = spec["selected"]["20400"]
            packet_path = ROOT / "artifacts" / name / f"640-{selected['ceiling']}.psbl"
            assert sha(packet_path) == selected["sha256"] == result["ball_delivery_sha256"]
            stream = ball_stream.unpack(packet_path.read_bytes())
            assert stream.frames == spec["frames"] and stream.fps == Fraction(25)
            base = decode(stream.native, [640, 360], spec["frames"], scratch)
            points, looks = ball.decode_frames(stream.ball_payload, spec["frames"], 960, 544)
            rendered = render_frames(base, points, looks)
            assert digest(rendered) == selected["displayed_sha256"]
            av1_path = ROOT / f"london-{name}-20400-closest-av1.ivf"
            assert sha(av1_path) == result["selected"]["sha256"]
            av1 = decode(av1_path.read_bytes(), [640, 360], spec["frames"], scratch)
            for record, source, masks, mark in source_rows[name]:
                images = {}
                for label, frames in (("av1", av1), ("ball", rendered)):
                    image = cv2.resize(frames[record["tick"]], (1280, 720), interpolation=cv2.INTER_CUBIC)
                    images[label] = score_regions(source, image, masks, mark, models)
                rows.append({"rally": name, "tick": record["tick"], "route": record["route"],
                    "mark": mark["kind"], "assisted": mark["assisted"], "scores": images})
            print(name, len(rows), flush=True)
    out = {"scope": "three physical London points; separate assisted and independent ball marks",
           "scorer_sha256": sha(Path(__file__)), "review_sha256": sha(EXPORT),
           "controls_sha256": sha(ROOT / "result.json"), "dists_sha256": sha(WEIGHTS),
           "lpips_sha256": sha(alex), "rows": rows}
    (ROOT / "scores.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
