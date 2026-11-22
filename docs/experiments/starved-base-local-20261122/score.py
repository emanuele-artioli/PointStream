"""Use the existing masked DISTS scorer for the six quarter-scale arms."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/private/tmp/ps-reference-check-env/lib/python3.12/site-packages")
sys.path.insert(0, "/private/tmp/ps-20261119-scorer-addon")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from src.codecs import quality  # noqa: E402

CROPS = Path("/private/tmp/agentctl-artifacts/Emanueles-MacBook-Pro.local/4a8859cb65354d3ebb3859ab96a00e4d/output/full-regional")
WEIGHTS = Path("/Users/manu/.codex/visualizations/2026/11/01/01a196fe-ab4d-70f0-a4b4-6843ca289667/atlas-metric-models/checkpoints/vgg16-397923af.pth")
EXPECTED = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
SCORER = ROOT / "docs/experiments/clean-court-regional-screen-20261118/score.py"
spec = importlib.util.spec_from_file_location("regional_score", SCORER)
assert spec is not None and spec.loader is not None
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=(1, 12), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    start = time.process_time()
    assert sha(WEIGHTS) == EXPECTED
    meta = json.loads((CROPS / "report.json").read_text())
    assert sha(CROPS / "crops.zip") == meta["crops_sha256"]
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available()
    net = quality.load_dists(WEIGHTS, "cpu")
    arms = [f"{mode}-{crf}" for mode in ("original", "clean") for crf in (36, 48, 60)]
    ticks = list(range(10, 240, 20))[:args.count]
    with zipfile.ZipFile(CROPS / "crops.zip") as archive:
        smoke = scorer.score_tick(net, archive, ticks[0], ["source", arms[0]])
        rows = []
        for tick in ticks:
            rows.extend(scorer.score_tick(net, archive, tick, arms))
    cpu = time.process_time() - start
    if args.count == 1 and cpu * 12 * 1.5 + 200 >= 900:
        raise RuntimeError("forecast exceeds CPU budget")
    means = {arm: {region: float(np.mean([r["dists"] for r in rows if r["arm"] == arm and r["region"] == region]))
             for region in ("p1", "p2", "ball")} for arm in arms}
    result = {"ticks": ticks, "arms": arms, "mean_dists": means, "rows": rows,
              "cpu_seconds": cpu, "metric": quality.dists_record(WEIGHTS),
              "crops_sha256": meta["crops_sha256"], "smoke": smoke,
              "torch": torch.__version__, "opencv": cv2.__version__}
    (args.output / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "metrics.json").write_text(json.dumps({"valid": True, "metrics": {"ticks": len(ticks)},
                                                          "cost": max(cpu, .001)}) + "\n")
    print(json.dumps({"cpu_seconds": cpu, "ticks": len(ticks), "mean_dists": means}))


if __name__ == "__main__":
    main()
