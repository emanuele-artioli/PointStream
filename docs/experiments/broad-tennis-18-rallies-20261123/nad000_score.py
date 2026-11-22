"""Score the new native output and recheck the previous output on identical source supports."""

import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "nadal_previous_score", HERE.parent / "starved-base-local-20261122/score.py"
)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def main(run):
    start = time.process_time()
    positive_dir = HERE.parent / "joint-trajectory-20261122"
    recorded = json.loads((positive_dir / "scores.json").read_text())
    old = {(r["t"], r["region"]): r["dists"] for r in recorded["per_tick"] if r["arm"] == "20"}
    receipt = json.loads((run / "display.json").read_text())
    assert score.sha(run / "crops.zip") == receipt["crop_zip_sha256"]
    assert receipt["positive_all_240_original"] and score.sha(score.WEIGHTS) == score.EXPECTED
    score.torch.set_num_threads(2)
    score.torch.set_num_interop_threads(1)
    score.cv2.setNumThreads(1)
    net = score.quality.load_dists(score.WEIGHTS, "cpu")
    with zipfile.ZipFile(run / "crops.zip") as z:
        t0 = time.process_time()
        smoke = score.scorer.score_tick(net, z, 10, ["source", "160", "64"])
        assert max(abs(r["dists"]) for r in smoke if r["arm"] == "source") < 1e-5
        est = (time.process_time() - t0) * 12 * 1.5 + 60 + receipt["cpu_seconds"] + t0 - start
        assert est < 720, est
        rows = [
            r
            for tick in receipt["reviewed_ticks"]
            for r in score.scorer.score_tick(net, z, tick, ["160", "64"])
        ]
    delta = max(abs(old[r["t"], r["region"]] - r["dists"]) for r in rows if r["arm"] == "160")
    assert delta < 0.0001, delta
    pts = {}
    for w in ("160", "64"):
        pt = {
            k: sum(r["dists"] for r in rows if r["arm"] == w and r["region"] == k) / 12
            for k in ("p1", "p2", "ball")
        }
        pt["players"] = (pt["p1"] + pt["p2"]) / 2
        pt["equal_class_fg"] = (pt["players"] + pt["ball"]) / 2
        pt["reviewed_full_frame_psnr"] = receipt["reviewed_full_rgb_psnr"][w]
        pts[w] = pt
    av = max(
        json.loads((positive_dir / "strict-20-ordinary.json").read_text())["points"],
        key=lambda x: x["bytes"],
    )
    dc = json.loads((positive_dir / "dcvc-q5-human-regions.json").read_text())
    anchors = {
        "AV1": {
            "kbps": av["kbps"],
            "players": av["players"],
            "ball": av["ball"],
            "wire_sha256": av["stream_sha256"],
        },
        "DCVC-UF": {
            "kbps": dc["kbps"],
            "players": dc["players"],
            "ball": dc["mean"]["ball"],
            "wire_sha256": dc["packet_sha256"],
        },
    }
    for row in anchors.values():
        row["equal_class_fg"] = (row["players"] + row["ball"]) / 2
    out = dict(
        points=pts,
        anchors=anchors,
        delta_old_score_max=delta,
        per_tick=rows,
        metric=score.quality.dists_record(score.WEIGHTS),
        cpu_seconds=time.process_time() - start,
        estimated_cpu_seconds=est,
        script_sha256=score.sha(Path(__file__)),
        display_receipt_sha256=score.sha(run / "display.json"),
        scope="12 source-reviewed times, both player silhouettes and 64-pixel visible ball crop; numerical average of player mean and ball crop weights both classes equally. One seen match under receiver cache.",
    )
    (run / "scores.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "points": pts,
                "anchor": anchors,
                "original_dists_check": delta,
                "cpu_seconds": out["cpu_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
