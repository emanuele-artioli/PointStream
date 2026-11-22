"""Rebind the qualified generation and finish dispatch to the sealed Shanghai data."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MED = HERE.parent / "med-mixed-receiver-20261122"
FLEET = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make(local, generation):
    data = json.loads((local / "result.json").read_text())
    row = next(r for r in data["rates"] if r["crf"] == data["selected_crf"])
    chosen = local / f"crf{row['crf']}"
    object_root = FLEET.parent.parent.parent / "tennis" / ("rv102-paid-" + row["joint_sha256"][:16])
    assert str(object_root).startswith("/home/itec/emanuele/Datasets/pointstream-data/tennis/")
    paths = [
        ("carrier", "players.pscp", "carrier_sha256"),
        ("manifest", "players.manifest", "manifest_sha256"),
        ("joint", "rally.psj", "joint_sha256"),
        ("cache", "initial.psr", "init_sha256"),
    ]
    paid = {}
    for key, name, field in paths:
        assert digest(chosen / name) == row[field]
        paid[key] = dict(name=key, path=str(object_root / name), sha256=row[field], extract=False)
    names = [r["unit"] for r in data["foreground"]]
    original = json.loads((HERE / "match102-spec.json").read_text())
    explicit = {r["name"]: r["path"] for r in original["staged_inputs"]}
    deny = [
        explicit["source"],
        explicit["features"],
        explicit["yolo11x_pose"],
        str(FLEET / "20261122T102553Z-a34de622/full/published.tar"),
        str(FLEET / "20261122T094101Z-ad793d52/full/ball.payload"),
    ]
    specs = []
    if generation is None:
        for idx, unit in enumerate(names):
            spec = json.loads((MED / "000-generate-0.json").read_text())
            args = spec["arguments"]
            args = args[: args.index("--deny")]
            for k, v in (
                ("--unit", unit),
                ("--carrier-sha256", row["carrier_sha256"]),
                ("--run-length", "240"),
                ("--full-cap", "2680"),
            ):
                args[args.index(k) + 1] = v
            spec.update(
                arguments=args,
                full={"seconds": 2680},
                budget_seconds=3450,
                scale={"ticks": {"smoke": 44, "full": 240}},
            )
            spec["smoke"]["representative_basis"] = (
                "44 consecutive native ticks with the same true two-player conditioning, 30 denoising steps, 1.5x full-stage forecast and source isolation."
            )
            spec["staged_inputs"] = [
                r for r in spec["staged_inputs"] if r["name"] not in ("carrier", "manifest")
            ] + [paid["carrier"], paid["manifest"]]
            specs.append((f"match102-generation-{idx}.json", spec))
    else:
        spec = json.loads((MED / "000-finish.json").read_text())
        seals = json.loads(generation.read_text())
        assert set(seals) == set(names)
        args = spec["arguments"][: spec["arguments"].index("--generation")]
        for k, value in (
            ("--carrier-sha256", row["carrier_sha256"]),
            ("--expected-yuv", row["decoded_sha256"]),
        ):
            args[args.index(k) + 1] = value
        sam = next(r for r in spec["staged_inputs"] if r["name"] == "sam")
        spec["staged_inputs"] = [sam, *paid.values()]
        for idx, n in enumerate(names):
            seal = seals[n]
            spec["staged_inputs"].append(
                dict(name=f"g{idx}", path=seal["path"], extract=True, sha256=seal["archive_sha256"])
            )
            args += ["--generation", n, "{staged:g" + str(idx) + "}", seal["receipt_sha256"]]
        spec["arguments"] = args
        spec["smoke"]["representative_basis"] = (
            "Four paid active ticks from first through last, two 240-tick generated banks, native decoded YUV verification, two-player SAM and independent full-output pixel checks."
        )
        specs.append(("match102-finish.json", spec))
    for filename, spec in specs:
        for p in deny:
            spec["arguments"] += ["--deny", p]
        spec["inputs"] = [{k: r[k] for k in ("path", "sha256")} for r in spec["staged_inputs"]]
        (HERE / filename).write_text(json.dumps(spec, indent=2) + "\n")
    print(object_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--generation", type=Path)
    a = parser.parse_args()
    make(a.local, a.generation)
