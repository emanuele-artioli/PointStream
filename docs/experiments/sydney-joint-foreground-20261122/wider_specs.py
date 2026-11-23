"""Build bounded calibration and paid-replay jobs from the frozen Sydney receipts."""

import argparse
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "test-joint-extension-20261122/native/full-confirmation-20261123"
CAMPUS = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")
CLEAN_SHA = "810783b90f05fd6f7413a6ff2d9a1db4a8903c2d5bdce05a89ddbe81e07238e8"


def produced(name, job, filename, digest):
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", job)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    return dict(name=name, path=str(CAMPUS / job / "full" / filename), sha256=digest, extract=False)


def build(a):
    template = json.loads((PREVIOUS / "match127-rung-70.json").read_text())
    staged = {r["name"]: r for r in template["staged_inputs"]}
    template["hosts"] = a.hosts
    template["deadline"] = a.deadline
    template.pop("cpu_affinity_index", None)
    if a.mode == "calibration":
        retained = [
            produced("clean", a.job, "clean.yuv", CLEAN_SHA),
            staged["original"],
            staged["cache"],
        ]
        module = "experiments.tennis.sydney_wider_base"
        native_hash = json.loads((PREVIOUS / "match127-finish-summary.json").read_text())[
            "native_yuv_sha256"
        ]
        template.update(
            entrypoint=["-m", module],
            validator=["{python}", "-m", module, "--validate"],
            arguments=[
                "--clean",
                "{staged:clean}",
                "--original",
                "{staged:original}",
                "--cache",
                "{staged:cache}",
                "--original-native-sha",
                native_hash,
                "--width",
                str(a.width),
                "--frames",
                "{frames}",
            ],
            smoke={
                "seconds": 150,
                "representative_basis": "Authenticated 24-frame wider native CPU encode and pinned decode",
            },
            full={"seconds": 1300},
            budget_seconds=1650,
            scale={"frames": {"smoke": 24, "full": 240}},
            local_storage_gib=6,
        )
    else:
        if a.reference:
            # Re-run the exact archived proof under the revised decoder.
            retained = list(staged.values())
        else:
            report = json.loads(a.calibration.read_text())
            assert report["frames"] == 240 and report["original_native_parity"]
            rate = next(r for r in report["rates"] if r["crf"] == a.crf)
            names = {
                "carrier": "players.pscp",
                "manifest": "players.manifest",
                "joint": "rally.psj",
                "cache": "initial.psr",
            }
            retained = [
                produced(name, a.job, f"crf{a.crf}-{file}", rate["files"][file])
                for name, file in names.items()
            ]
            retained += [staged["original"], staged["witness"]]
            args = template["arguments"]
            args[args.index("--carrier-sha256") + 1] = rate["files"]["players.pscp"]
            args[args.index("--native-sha") + 1] = rate["native_sha256"]
            args.remove("--reference")
    template["staged_inputs"] = retained
    template["inputs"] = [{k: item[k] for k in ("path", "sha256")} for item in retained]
    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibration", "replay"))
    parser.add_argument("--job")
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--hosts", nargs="+", required=True)
    parser.add_argument("--width", type=int, choices=(320, 640), default=320)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--crf", type=int)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parsed = parser.parse_args()
    parsed.out.write_text(json.dumps(build(parsed), indent=2) + "\n")
