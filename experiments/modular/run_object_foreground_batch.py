"""Run the three fixed 48-frame object-foreground handoffs independently."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sqlite3  # noqa: F401  # host C++ runtime before pose backend imports Torch
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.jobs.monitor import publish_progress  # noqa: E402
from experiments.modular.object_foreground_campaign import OUT, run_clip  # noqa: E402


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "batch-summary.json"
    summary: dict[str, object] = {
        "kind": "three independent 48-frame foreground bitstreams",
        "clips": {},
        "started_unix": time.time(),
    }
    for index, clip in enumerate(("alcaraz000", "federer007", "perricard002"), 1):
        try:
            result = run_clip(clip, out_dir=OUT)
            summary["clips"][clip] = {
                "status": "complete",
                "rows": len(result["rows"]),
                "result": str(OUT / f"{clip}.json"),
            }
        except Exception as exc:
            summary["clips"][clip] = {
                "status": "stopped",
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        publish_progress("object-foreground", index, decision=f"{clip}: {summary['clips'][clip]['status']}")
        gc.collect()
    summary["finished_unix"] = time.time()
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if any(item["status"] != "complete" for item in summary["clips"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
