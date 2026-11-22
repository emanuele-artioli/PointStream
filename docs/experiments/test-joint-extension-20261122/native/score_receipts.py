"""Compare genuine full-event rate points only after all original source ticks and masks agree."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def family(name):
    if name.startswith(("av1-", "ordinary-")):
        return "SVT-AV1"
    if name.startswith("roi-"):
        return "SVT-AV1 ROI"
    if name.startswith("dcvc-"):
        return "DCVC-UF"
    assert name.startswith("joint-crf")
    return "PointStream"


def summarize(a):
    sender = json.loads(a.sender.read_text())
    rows = []
    matched = {}
    for path in sorted(a.scores.glob("*/score.json")):
        s = json.loads(path.read_text())
        v = json.loads((path.parent / "full_validation.json").read_text())
        assert v["passed"] and v["checks"]["frames"] == s["frames"] == 240
        assert len(s["rows"]) == 480 and v["checks"]["roles"] == 2
        assert hashlib.sha256(path.read_bytes()).hexdigest() == v["checks"]["score_sha256"]
        assert (
            s["source_sha256"] == sender["source_sha256"]
            and s["native_fps"] == sender["native_fps"]
        )
        assert s["name"] not in matched and s["event"] == a.event
        matched[s["name"]] = s
        rows.append(
            dict(
                name=s["name"],
                family=family(s["name"]),
                bytes=s["bytes"],
                kbps=s["kbps"],
                source_grid=s["source_grid_sha256"],
                support=s["support_rle_sha256"],
                dists=s["pooled"]["dists"],
                near=s["near"]["dists"],
                far=s["far"]["dists"],
                body_psnr=s["pooled"]["rgb_psnr"],
                full_psnr=s["whole_rgb_psnr"],
                score_job=path.parent.name,
                producer=s["producer"],
                typed_decode_sha256=s["decoded_sha256"],
            )
        )
    assert rows and len({r["source_grid"] for r in rows}) == len({r["support"] for r in rows}) == 1
    results = []
    for j in rows:
        if j["family"] != "PointStream":
            continue
        identities = [(r["tick"], r["role"], r["area"]) for r in matched[j["name"]]["rows"]]
        physical = next(r for r in sender["rates"] if "joint-crf" + str(r["crf"]) == j["name"])
        assert physical["joint_bytes"] == j["bytes"]
        for label, ceiling in (("warm", j["bytes"]), ("cold", j["bytes"] + physical["init_bytes"])):
            for codec in ("SVT-AV1", "SVT-AV1 ROI", "DCVC-UF"):
                rs = [r for r in rows if r["family"] == codec and r["bytes"] <= ceiling]
                if not rs:
                    continue
                b = min(rs, key=lambda r: (r["dists"], r["bytes"]))
                observations = matched[b["name"]]["rows"]
                assert [(r["tick"], r["role"], r["area"]) for r in observations] == identities
                signed = [
                    x["body_dists"] - y["body_dists"]
                    for x, y in zip(matched[j["name"]]["rows"], observations, strict=True)
                ]
                assert abs(mean(signed) - (j["dists"] - b["dists"])) < 1e-8
                results.append(
                    dict(
                        joint=j["name"],
                        state=label,
                        ceiling_bytes=ceiling,
                        baseline=b["name"],
                        baseline_bytes=b["bytes"],
                        relative_player_dists_gain=1 - j["dists"] / b["dists"],
                        joint_lower_player_observations=sum(x < 0 for x in signed),
                    )
                )
    a.out.write_text(
        json.dumps(
            dict(
                event=a.event,
                source_sha256=sender["source_sha256"],
                fps=sender["native_fps"],
                frames=240,
                source_grid_sha256=rows[0]["source_grid"],
                source_support_sha256=rows[0]["support"],
                measured=rows,
                comparisons=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", required=True)
    for k in ("sender", "scores", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    summarize(p.parse_args())
