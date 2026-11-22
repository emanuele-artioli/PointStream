"""Production K4b feature worker with native-length resource and output gates."""

import argparse
import json
import os
from pathlib import Path
import tarfile
import tempfile

from experiments.tennis import k4b_features as features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sam-checkpoint")
    p.add_argument("--pose")
    p.add_argument("--source")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--limit", type=int, default=1)
    a = p.parse_args()
    stage = Path(os.environ["PS_STAGE_DIR"])
    if not a.validate:
        if a.limit != 1:
            raise ValueError("fixed one-window scope")
        a.windows = ["shanghai/match284@120"]
        a.sources = [a.source]
        a.max_windows = 1
        a.ffmpeg = "ffmpeg"
        raise SystemExit(features.run(a))
    summary = json.loads((stage / "k4b-features.json").read_text())
    with tempfile.TemporaryDirectory(
        prefix="match284-features-validate-", dir=os.environ["TMPDIR"]
    ) as temp:
        with tarfile.open(stage / "published.tar") as tar:
            tar.extractall(temp, filter="data")
        report = features.validate(Path(temp) / "publish", summary, Path(a.manifest))
        root = Path(temp) / "publish" / "shanghai__match284@120"
        result = json.loads((root / "features.json").read_text())
        report["checks"]["at_least_two_canonical_tracks"] = len(result["features"]) >= 2
        report["checks"]["native_240_tick_coverage"] = result["frame_count"] == 240
        report["checks"]["smoke_forecast"] = (
            os.environ["PS_STAGE"] != "smoke" or 1.5 * summary["seconds"] + 120 < 1200
        )
        report["passed"] = all(report["checks"].values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(report) + "\n")
    if not report["passed"]:
        raise SystemExit(str(report["checks"]))


if __name__ == "__main__":
    main()
