"""Measure registered 24-frame RGB crop fidelity from source and original render tensors."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from experiments.tennis.player_reference_review import resolved
from experiments.tennis.player_requalification import make_plan, sha256, validate_output
from experiments.tennis.player_requalification_preview import source_crops

ROOT = Path("/local/users/emanuele/pointstream/q78-metric-audit-20261122")
RECOVERY = Path("/local/users/emanuele/pointstream/q78-salvage-review-20261122")
BASE = Path("/local/users/emanuele/pointstream/cache/075b001a3ae9dcc9fdda9fb81e4240aaf3a9da228d707ac58733db094d983285/tree/publish/tennis-base@fit-100.pt")
SOURCE = Path("/home/itec/emanuele/Datasets/pointstream-data/tennis/w7-tier2-codex-20261031/eval/sources/nadal_kyrgios__000.mp4")
CASES = {
    "lossless": {
        "config": "/home/itec/emanuele/Datasets/pointstream-data/tennis/player-requalification-20261116/config.json",
        "hash": "eb67e4a972cde36895addef8db306c963423f814a31be7273a31c696ce5e98c6",
        "conditioning": ROOT / "lossless-cond", "full": ROOT / "lossless/publish",
        "archive": "/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261116T124347Z-407d92df/full/published.tar",
        "archive_sha256": "04518b719632329c1164263c9231f77740838cfef9f6dccfe6d5bfe34c409a21",
    },
    "avif_q65": {
        "config": "/home/itec/emanuele/Datasets/pointstream-data/tennis/player-reference-common-avif-q65-20261122/config.json",
        "hash": "0cd63fe70ce618be8ee9457ed69c266e19a50fa9ade751fbcdd7d74995c33869",
        "conditioning": ROOT / "q65-cond", "full": ROOT / "q65/publish",
        "archive": "/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261121T183911Z-a992fc53/full/published.tar",
        "archive_sha256": "94fc07b916792b8c3e7c50b1745d45e3050a04fbd02c8928407cdbe27a027ec0",
    },
    "avif_q78": {
        "config": "/home/itec/emanuele/Datasets/pointstream-data/tennis/player-reference-common-avif-q78-20261122/config.json",
        "hash": "ad389004d8f08e3428afa317c3e32aeaeb00fbd9955d333be501619b60df7768",
        "conditioning": Path("/local/users/emanuele/pointstream/cache/44ceb171ffbafb0a31261d11544e8b234f68c20170d06c2360e3169113405bfb/tree"),
        "full": RECOVERY / "full/publish",
        "archive": "/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261121T214127Z-a11b5627/full/partial.tar",
        "archive_sha256": "9b26151cd2e92f233ececb3f7f3a1c2627f1cbd65841be0ed3ca4ecaf0c39b9b",
    },
}


def main(include_q90=False):
    if include_q90:
        staging = json.loads(Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261121T225211Z-d22b94ba/full/staging.json").read_text())
        inputs = {v["name"]: v["path"] for v in staging["inputs"]}
        CASES["avif_q90"] = {
            "config": inputs["recipe"], "hash": "6d2327a4f3a61414202dc5de90ff517161a489ffecee9ac7ea359f18a0c4f976",
            "conditioning": Path(inputs["conditioning"]), "full": Path("/local/users/emanuele/pointstream/q90-paired-review-20261122/full"),
            "archive": "/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261121T225211Z-d22b94ba/full/published.tar",
            "archive_sha256": "c3121bb56227374c20a2de9274bd096b745947a6f77cc82895919073e9918c01",
        }
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This is a CPU-only audit")
    assert sha256(SOURCE) == "b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d"
    configs, receipts, plans, arrays = {}, {}, {}, {}
    for name, case in CASES.items():
        assert sha256(Path(case["archive"])) == case["archive_sha256"]
        configs[name] = resolved(Path(case["config"]), case["hash"], case["conditioning"], BASE)
        receipts[name] = validate_output(case["full"], configs[name], case["hash"], "tennis_only", 2)
        plans[name], arrays[name] = make_plan(configs[name], case["hash"], 2)
    control = plans["lossless"]
    for name in (name for name in CASES if name != "lossless"):
        assert plans[name]["settings"] == control["settings"]
        assert plans[name]["source"]["source_video_sha256"] == control["source"]["source_video_sha256"]
        by_unit = {unit["unit"]: unit for unit in plans[name]["units"]}
        assert set(by_unit) == {unit["unit"] for unit in control["units"]}
        for a in control["units"]:
            b = by_unit[a["unit"]]
            different = [key for key in ("unit", "run", "seed", "t", "pose_sha256", "box_sha256") if a[key] != b[key]]
            if different:
                raise ValueError(f"Conditioning mismatch for {name}, {a['unit']}: {different}")
    source, deltas = source_crops(SOURCE, control, arrays["lossless"],
                                  Path(CASES["lossless"]["config"]),
                                  [["conditioning", str(CASES["lossless"]["conditioning"])]])
    assert deltas == [0, 0]
    result = {"source_sha256": sha256(SOURCE), "reference_alignment_max_rgb_delta": deltas,
              "support": "all RGB pixels of each 512x784 source-aligned crop, 24 ticks",
              "metrics": {}, "archives": {name: case["archive_sha256"] for name, case in CASES.items()},
              "config_sha256": {name: case["hash"] for name, case in CASES.items()}}
    for index, unit in enumerate(control["units"]):
        result["metrics"][unit["unit"]] = {}
        truth = source[index]
        for name, case in CASES.items():
            with np.load(case["full"] / f'{unit["unit"]}@tennis_only.npz', allow_pickle=False) as output:
                generated = output["crops"]
                assert generated.shape == truth.shape == (24, 784, 512, 3)
                total_abs, total_sq = 0, 0
                for tick in range(24):
                    error = generated[tick].astype(np.int32) - truth[tick].astype(np.int32)
                    total_abs += int(np.abs(error).sum(dtype=np.int64))
                    total_sq += int(np.square(error).sum(dtype=np.int64))
                count = generated.size
                mse = total_sq / count
                result["metrics"][unit["unit"]][name] = {
                    "rgb_mae": total_abs / count, "rgb_mse": mse,
                    "rgb_psnr_db": 10 * math.log10(255**2 / mse),
                    "samples": count, "render_sha256": sha256(case["full"] / f'{unit["unit"]}@tennis_only.npz'),
                }
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q90", action="store_true")
    main(parser.parse_args().q90)
