"""Read ball centers from the two exact combined deliveries and score source marks."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from src.pointstream import ball
from experiments.tennis import common_player_packet, w8_payload

HERE = Path(__file__).resolve().parent
LABEL = "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"


def main(args):
    review = args.labels.read_bytes()
    assert hashlib.sha256(review).hexdigest() == LABEL
    labels = json.loads(review)["frames"]
    assert [r["tick"] for r in labels] == list(range(10, 240, 20))
    charged = json.loads((HERE / "result.json").read_text())["streams"]
    previous = json.loads((HERE.parent / "ball-machine-twenty-20261122/result.json").read_text())
    arms = {}
    for cap, receipt in charged.items():
        payload = (args.packets / f"joint-{cap}.psw").read_bytes()
        assert (
            hashlib.sha256(payload).hexdigest() == receipt["wire_sha256"]
            and len(payload) == receipt["bytes"]
        )
        profile = common_player_packet._embedded_profile(payload)
        sections, _ = w8_payload.unpack(payload, profile)
        w, h = profile["ball_canvas"]
        centers, looks = ball.decode_frames(sections["trajectory"], 240, w, h)
        assert ball.encode_frames(centers, looks, w, h) == sections["trajectory"]
        errors = []
        for f in labels:
            assert f["ball"]["status"] == "visible"
            t = f["tick"]
            x, y = f["ball"]["center_xy"]
            error = (
                None
                if centers[t] is None
                else math.hypot(centers[t][0] * 1920 / w - x, centers[t][1] * 1080 / h - y)
            )
            archived = previous["direct_errors"][str(t)]
            assert (
                error is None
                and archived is None
                or error is not None
                and archived is not None
                and abs(error - archived) < 1e-8
            )
            errors.append({"tick": t, "native_pixel_error": error})
        arms[cap] = {
            "wire_sha256": receipt["wire_sha256"],
            "symbols_sha256": hashlib.sha256(sections["trajectory"]).hexdigest(),
            "native_errors": errors,
            "hits_10px": sum(
                v["native_pixel_error"] is not None and v["native_pixel_error"] <= 10
                for v in errors
            ),
            "hits_5px": sum(
                v["native_pixel_error"] is not None and v["native_pixel_error"] <= 5 for v in errors
            ),
            "missing_symbols_over_240": sum(c is None for c in centers),
        }
    ordinary = next(
        p
        for p in json.loads((HERE / "control-scores.json").read_text())["points"]
        if p["mode"] == "plain" and p["width"] == 240 and p["cap"] == 20
    )
    bound = json.loads((HERE.parent / "starved-ball-twenty-20261122/result.json").read_text())[
        "anchors"
    ]["plain240Q63"]
    judged = previous["results"]["plain240Q63"]
    assert (
        ordinary["decoded_sha256"] == bound["decoded_sha256"]
        and ordinary["bytes"] == bound["file_bytes"] == judged["bytes"]
    )
    baseline = {
        "bitstream_sha256": bound["stream_sha256"],
        "frame_sha256": ordinary["decoded_sha256"],
        "bytes": judged["bytes"],
        "kbps": judged["kbps"],
        "judge_input_rgb_sha256": judged["input_rgb_sha256"],
        "hits_10px": sum(r["hit10"] for r in judged["rows"]),
        "hits_5px": sum(r["hit5"] for r in judged["rows"]),
        "record": "../ball-machine-twenty-20261122/result.json#plain240Q63",
    }
    result = {
        "arms": arms,
        "matched_ordinary": baseline,
        "labels_sha256": LABEL,
        "native_size": [1920, 1080],
        "canvas": [w, h],
        "visible": len(labels),
        "not_visible": 0,
        "scope": "directly decoded positions on the two combined deliveries; no new detector inference on these displays",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print([(k, v["hits_5px"], v["hits_10px"]) for k, v in arms.items()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packets", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    main(parser.parse_args())
