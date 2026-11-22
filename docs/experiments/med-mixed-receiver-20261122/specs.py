"""Materialize campus receiver specs for the selected native mixed packet."""

import argparse
import json
from pathlib import Path

from build import HERE, SOURCES, sha

OLD = HERE.parent / "joint-replication-feasibility-20261122"
DATA = Path("/home/itec/emanuele/Datasets/pointstream-data")
INBOX = DATA / "jobs/fleet/inbox"


def update_inputs(spec, rows):
    by_name = {x["name"]: x for x in rows}
    assert len(rows) == len(by_name)
    spec["staged_inputs"] = rows
    spec["inputs"] = [{k: r[k] for k in ("path", "sha256")} for r in rows]


def main(a):
    result = json.loads((a.folder / "result.json").read_text())
    assert result["source_sha256"] == SOURCES[a.clip] and result["native_fps"] == 50
    (chosen,) = [
        x
        for x in result["rates"]
        if x["crf"] == (a.crf if a.crf is not None else result["selected_crf"])
    ]
    local = a.folder / f"crf{chosen['crf']}"
    prefix = DATA / "tennis" / f"med{a.clip}-mixed-20261122-{chosen['joint_sha256'][:12]}"
    files = [
        ("carrier", "players.pscp", "carrier_sha256"),
        ("manifest", "players.manifest", "manifest_sha256"),
        ("joint", "rally.psj", "joint_sha256"),
        ("cache", "initial.psr", "init_sha256"),
    ]
    paid = {}
    for name, filename, key in files:
        assert sha(local / filename) == chosen[key]
        paid[name] = dict(name=name, path=str(prefix / filename), sha256=chosen[key], extract=False)
    names = [p["unit"] for p in result["foreground"]]
    denied = [
        str(DATA / f"tennis/w7-tier1-20261030/sources/medvedev_struff__{a.clip}.mp4"),
        str(INBOX / "20261031T091038Z-e8d1cefb/full/published.tar"),
        str(INBOX / "20261031T091524Z-7d2f62ea/full/published.tar"),
        str(DATA / "tennis/k5b-2026-10-20/inputs/balltrack_best.state_dict.pt"),
    ]
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage_only:
        pass
    elif a.generation is None:
        for i, n in enumerate(names):
            sp = json.loads((OLD / "al028-receiver-4-a6000-2680-spec.json").read_text())
            args = sp["arguments"]
            args[args.index("--carrier-sha256") + 1] = chosen["carrier_sha256"]
            args[args.index("--unit") + 1] = n
            args[args.index("--full-cap") + 1] = "2150"
            args = args[: args.index("--deny")] + ["--run-length", "120"]
            for p in denied:
                args += ["--deny", p]
            sp.update(
                arguments=args,
                deadline="2026-11-22T21:01:00Z",
                budget_seconds=3000,
                gpu_models=["RTX A6000", "RTX 6000 Ada"],
                hosts=["gpu2", "gpu3", "gpu5", "gpu6"],
                full=dict(seconds=2150),
                scale=dict(ticks=dict(smoke=44, full=120)),
            )
            sp["smoke"]["representative_basis"] = (
                "44 consecutive paid native source ticks, 30 generation steps and same seed as the full 120-tick interval; actual CUDA kernels, source denial, repeated A6000 test and 1.5x processing forecast required."
            )
            update_inputs(
                sp,
                [r for r in sp["staged_inputs"] if r["name"] not in ("carrier", "manifest")]
                + [paid["carrier"], paid["manifest"]],
            )
            (a.out / f"{a.clip}-generate-{i}.json").write_text(json.dumps(sp, indent=2) + "\n")
    else:
        sp = json.loads((OLD / "al028-finisher-full-spec.json").read_text())
        sp.update(
            deadline="2026-11-22T21:01:00Z",
            budget_seconds=1800,
            hosts=["gpu2", "gpu3", "gpu5", "gpu6"],
            gpu_sharing={"mode": "exclusive"},
            contention={"policy": "stop", "resume_attempts": 0},
            gpu_models=["RTX A6000", "RTX 6000 Ada"],
            full=dict(seconds=1400),
        )
        sp["smoke"].update(
            seconds=330,
            representative_basis="Three real active ticks, one conventional fallback, and both authenticated 120-tick learned player banks; complete native decode, exact mixed remux, independent raster and per-frame validation.",
        )
        auth = json.loads(a.generation.read_text())
        assert set(auth) == set(names)
        sam = next(r for r in sp["staged_inputs"] if r["name"] == "sam")
        staged = [sam, *paid.values()]
        args = [
            "run",
            "--carrier",
            "{staged:carrier}",
            "--manifest",
            "{staged:manifest}",
            "--carrier-sha256",
            chosen["carrier_sha256"],
            "--sam",
            "{staged:sam}",
            "--full-cap",
            "1400",
            "--ticks",
            "{ticks}",
            "--joint",
            "{staged:joint}",
            "--cache",
            "{staged:cache}",
            "--expected-yuv",
            chosen["decoded_sha256"],
        ]
        for i, n in enumerate(names):
            r = auth[n]
            name = f"g{i}"
            staged.append(dict(name=name, extract=True, path=r["path"], sha256=r["archive_sha256"]))
            args += ["--generation", n, "{staged:" + name + "}", r["receipt_sha256"]]
        for pp in denied:
            args += ["--deny", pp]
        sp["arguments"] = args
        update_inputs(sp, staged)
        (a.out / f"{a.clip}-finish.json").write_text(json.dumps(sp, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                local=str(local),
                remote=str(prefix),
                objects=names,
                roles={k: v["sha256"] for k, v in paid.items()},
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True, choices=SOURCES)
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--generation", type=Path)
    p.add_argument("--crf", type=int)
    p.add_argument("--stage-only", action="store_true")
    main(p.parse_args())
