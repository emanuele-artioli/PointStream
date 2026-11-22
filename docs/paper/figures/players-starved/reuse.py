"""Price the original player bank at the highest scored DCVC rate in each window."""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "experiments/w9-inputs"
ADAPTERS = {"al": 1_488_252, "med": 1_495_592, "nad": 1_489_876}


def main():
    sources = {}
    rows = []
    for tier in (1, 2):
        file = DATA / f"w7-tier{tier}-billing-corrected.json"
        sources[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        for row in json.loads(file.read_bytes())["windows"]:
            neural = row["dcvc_points"]
            if not neural:
                continue
            name = row["window"]
            bill = row["billing_correction"]
            fee = ADAPTERS[name.rstrip("0123456789")]
            assert bill["intended_bank_bytes"] == 6 * bill["png_bytes_two_crops"]
            once = fee + bill["intended_bank_bytes"]
            running = bill["old_rate_kbps"] - (fee + bill["old_bank_bytes"]) * 0.008 / 7200
            assert math.isclose(row["rate_kbps"], running + once * 0.008 / 7200)
            best_bitrate = max(p[1] for p in neural)
            worst_advantage = min(p[-1] for p in neural) - row["hybrid"]["dists"]["foreground"]
            assert worst_advantage > 0
            seconds = once * 0.008 / (best_bitrate - running)
            assert seconds > 0 and math.isclose(running + once * 0.008 / seconds, best_bitrate)
            rows.append(dict(window=name, running_kbps=running, fixed_model_and_image_bytes=once,
                             highest_scored_neural_kbps=best_bitrate, seconds_to_fit_that_rate=seconds,
                             dists_advantage_over_best_scored_neural=worst_advantage))
    assert len(rows) == 8
    (HERE / "reuse-thresholds.json").write_text(json.dumps(dict(
        sources=sources, scope="Fixed historical bank and match-adaptation cost; active seconds at the archived running rate. Each comparator is the highest tested DCVC rate on the same window.",
        rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
