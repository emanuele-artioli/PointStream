"""Bind CPU native-base replays to the finished, source-denied full-display witness."""

import argparse
import json
from pathlib import Path

from build import HERE
from specs import DATA, INBOX


def main(a):
    r = json.loads((a.build / "result.json").read_text())
    original = next(x for x in r["rates"] if x["crf"] == r["selected_crf"])
    original_path = (
        DATA
        / "tennis"
        / f"med{a.clip}-mixed-20261122-{original['joint_sha256'][:12]}"
        / "rally.psj"
    )
    gpu = json.loads((HERE / f"{a.clip}-generate-0.json").read_text())
    template = json.loads((HERE / "000-ball-recost.json").read_text())
    denied = [gpu["arguments"][i + 1] for i, x in enumerate(gpu["arguments"]) if x == "--deny"]
    for cost in r["rates"]:
        reference = cost["crf"] == r["selected_crf"]
        folder = DATA / "tennis" / f"med{a.clip}-mixed-20261122-{cost['joint_sha256'][:12]}"
        inputs = []
        for name, filename, key in [
            ("carrier", "players.pscp", "carrier_sha256"),
            ("manifest", "players.manifest", "manifest_sha256"),
            ("joint", "rally.psj", "joint_sha256"),
            ("cache", "initial.psr", "init_sha256"),
        ]:
            inputs.append(
                dict(name=name, path=str(folder / filename), sha256=cost[key], extract=False)
            )
        if not reference:
            inputs.append(
                dict(
                    name="original",
                    path=str(original_path),
                    sha256=original["joint_sha256"],
                    extract=False,
                )
            )
        inputs.append(
            dict(
                name="witness",
                path=str(INBOX / a.witness / "full/published.tar"),
                sha256=a.witness_sha256,
                extract=True,
            )
        )
        args = [
            "--carrier",
            "{staged:carrier}",
            "--carrier-sha256",
            cost["carrier_sha256"],
            "--manifest",
            "{staged:manifest}",
            "--joint",
            "{staged:joint}",
            "--cache",
            "{staged:cache}",
            "--original-joint",
            "{staged:joint}" if reference else "{staged:original}",
            "--witness",
            "{staged:witness}",
            "--native-sha",
            cost["decoded_sha256"],
            "--frames",
            "{frames}",
        ]
        if reference:
            args.append("--reference")
        for p in denied:
            args.extend(["--deny", p])
        sp = dict(
            schema=1,
            entrypoint=["-m", "experiments.tennis.med_mixed_rung"],
            validator=["{python}", "-m", "experiments.tennis.med_mixed_rung", "--validate"],
            arguments=args,
            staged_inputs=inputs,
            inputs=[{k: r[k] for k in ("path", "sha256")} for r in inputs],
            environment=template["environment"],
            required_commands=[],
            deadline=template["deadline"],
            verify_staged_inputs=True,
            stall_seconds=1380,
            contention=template["contention"],
            device="cpu",
            hosts=template["hosts"],
            gpu_models=[],
            gpu_memory_mib=0,
            cpu_threads=1,
            single_cpu_campaign=True,
            local_storage_gib=30,
            smoke=dict(
                seconds=500,
                representative_basis="Original paid/native stream, 3 real active ticks and 1 real fallback observed under source denial, real frame hashes, full working set and timing.",
            ),
            full=dict(seconds=1320),
            validator_seconds=170,
            budget_seconds=2080,
            scale=dict(frames=dict(smoke=4, full=240)),
        )
        (a.out / f"{a.clip}-rung-{cost['crf']}.json").write_text(json.dumps(sp, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--witness", required=True)
    p.add_argument("--witness-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
