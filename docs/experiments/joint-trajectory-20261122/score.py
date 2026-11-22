"""Score both transmitted joint modes on the original twelve foreground supports."""

import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "starved-base-local-20261122"))

spec = importlib.util.spec_from_file_location(
    "quarter_scale_score", HERE.parent / "starved-base-local-20261122/score.py"
)
quarter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quarter)


def main(run):
    source = json.loads((run / "result.json").read_text())
    assert quarter.sha(run / "crops.zip") == source["crops_sha256"]
    assert quarter.sha(quarter.WEIGHTS) == quarter.EXPECTED
    quarter.cv2.setNumThreads(1)
    quarter.torch.set_num_threads(2)
    quarter.torch.set_num_interop_threads(1)
    net = quarter.quality.load_dists(quarter.WEIGHTS, "cpu")
    with zipfile.ZipFile(run / "crops.zip") as z:
        start = time.process_time()
        self_check = quarter.scorer.score_tick(net, z, 10, ["source", "20", "33"])
        forecast = (time.process_time() - start) * 12 * 1.5 + 60
        assert forecast < 740
        rows = [
            row
            for tick in source["reviewed_ticks"]
            for row in quarter.scorer.score_tick(net, z, tick, ["20", "33"])
        ]
    actual = []
    for row in source["streams"].values():
        assert (
            row["all_240_rgb_sha256"] and row["cold_bytes"] == row["bytes"] + row["reference_bytes"]
        )
    for cap in ("20", "33"):
        means = {
            region: sum(r["dists"] for r in rows if r["arm"] == cap and r["region"] == region) / 12
            for region in ("p1", "p2", "ball")
        }
        actual.append(
            dict(
                cap=int(cap),
                players=(means["p1"] + means["p2"]) / 2,
                **means,
                **source["streams"][cap],
            )
        )
    result = dict(
        points=actual,
        per_tick=rows,
        metric=quarter.quality.dists_record(quarter.WEIGHTS),
        torch=quarter.torch.__version__,
        numpy=quarter.np.__version__,
        opencv=quarter.cv2.__version__,
        self_check=self_check,
        forecast=forecast,
        cpu_seconds=time.process_time() - start,
        receipt_sha256=quarter.sha(run / "result.json"),
        script_sha256=quarter.sha(Path(__file__)),
        python=sys.executable,
    )
    (run / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            [{k: r[k] for k in ("cap", "kbps", "cold_kbps", "players", "ball")} for r in actual]
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
