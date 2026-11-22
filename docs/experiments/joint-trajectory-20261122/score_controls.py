"""Score the six real conventional searches on the joint receiver's reviewed frames."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("joint_score", HERE / "score.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
quarter = module.quarter


def main(root):
    quarter.cv2.setNumThreads(1)
    quarter.torch.set_num_threads(2)
    quarter.torch.set_num_interop_threads(1)
    assert quarter.sha(quarter.WEIGHTS) == quarter.EXPECTED
    net = quarter.quality.load_dists(quarter.WEIGHTS, "cpu")
    points, receipts = [], []
    joint = json.loads((HERE / "result.json").read_text())
    for cap, joint_stream in joint["streams"].items():
        for width in (160, 240, 480):
            directory = root / f"{cap}-{width}"
            report = json.loads((directory / "result.json").read_text())
            assert report["target_bytes"] == joint_stream["bytes"]
            assert report["geometry"][0] == width
            assert quarter.sha(directory / "crops.zip") == report["crop_sha256"]
            arms = list(report["selected"])
            start = time.process_time()
            with zipfile.ZipFile(directory / "crops.zip") as z:
                smoke = quarter.scorer.score_tick(net, z, 10, ["source", *arms])
                assert (time.process_time() - start) * 12 * 1.5 + 10 < 165
                scores = [
                    r
                    for t in joint["reviewed_ticks"]
                    for r in quarter.scorer.score_tick(net, z, t, arms)
                ]
            for name, row in report["selected"].items():
                means = {
                    r: sum(x["dists"] for x in scores if x["arm"] == name and x["region"] == r) / 12
                    for r in ("ball", "p1", "p2")
                }
                points.append(
                    dict(
                        cap=int(cap),
                        mode=row["mode"],
                        crf=row["q"],
                        width=width,
                        kbps=row["rate"],
                        bytes=row["file_bytes"],
                        decoded_sha256=row["decoded_sha256"],
                        players=(means["p1"] + means["p2"]) / 2,
                        **means,
                    )
                )
            receipts.append(
                dict(
                    cap=cap,
                    width=width,
                    source=report["source_sha256"],
                    roi_map=report["roi_map_sha256"],
                    driver_sha256=report["driver_sha256"],
                    stream_report_sha256=quarter.sha(directory / "result.json"),
                    complete_encodes=[
                        {
                            k: v[k]
                            for k in ("mode", "n", "q", "file_bytes", "rate", "decoded_sha256")
                        }
                        for v in report["all_encodes"]
                    ],
                    selected=report["selected"],
                    per_tick=scores,
                    smoke=smoke,
                    cpu=report["cpu_seconds"],
                )
            )
    output = dict(
        points=points,
        receipts=receipts,
        metric=quarter.quality.dists_record(quarter.WEIGHTS),
        torch=quarter.torch.__version__,
        opencv=quarter.cv2.__version__,
        tools=report["tools"],
        script_sha256=quarter.sha(Path(__file__)),
        python=sys.executable,
        reviewed_ticks=joint["reviewed_ticks"],
    )
    (root / "control-scores.json").write_text(json.dumps(output, indent=2) + "\n")
    print([(p["cap"], p["width"], p["mode"], round(p["kbps"], 2)) for p in points])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
