"""Bind the two additional native Shanghai AV1 base rungs to a full isolated witness."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MED = HERE.parent / "med-mixed-receiver-20261122"
BASE = Path("/home/itec/emanuele/Datasets/pointstream-data")


def main(a):
    s = json.loads((HERE / "match102-sender.json").read_text())
    chosen = next(r for r in s["rates"] if r["crf"] == s["selected_crf"])
    gpu = json.loads((HERE / "match102-generation-0-ada.json").read_text())
    denies = [gpu["arguments"][i + 1] for i, v in enumerate(gpu["arguments"]) if v == "--deny"]
    for row in s["rates"]:
        reference = row["crf"] == s["selected_crf"]
        spec = json.loads(
            (MED / ("000-rung-60.json" if reference else "000-rung-70.json")).read_text()
        )
        original = BASE / "tennis" / ("rv102-paid-" + chosen["joint_sha256"][:16]) / "rally.psj"
        folder = BASE / "tennis" / ("rv102-paid-" + row["joint_sha256"][:16])
        keys = [
            ("carrier", "players.pscp", "carrier_sha256"),
            ("manifest", "players.manifest", "manifest_sha256"),
            ("joint", "rally.psj", "joint_sha256"),
            ("cache", "initial.psr", "init_sha256"),
        ]
        inputs = [
            dict(name=key, path=str(folder / f), sha256=row[h], extract=False) for key, f, h in keys
        ]
        if not reference:
            inputs.append(
                dict(
                    name="original",
                    path=str(original),
                    sha256=chosen["joint_sha256"],
                    extract=False,
                )
            )
        inputs.append(
            dict(
                name="witness",
                path=str(BASE / "jobs/fleet/inbox" / a.job / "full/published.tar"),
                sha256=a.witness_sha256,
                extract=True,
            )
        )
        args = spec["arguments"][: spec["arguments"].index("--deny")]
        for k, v in (
            ("--carrier-sha256", row["carrier_sha256"]),
            ("--native-sha", row["decoded_sha256"]),
        ):
            args[args.index(k) + 1] = v
        assert ("--reference" in args) == reference
        for p in denies:
            args += ["--deny", p]
        spec["arguments"] = args
        spec["staged_inputs"] = inputs
        spec["inputs"] = [{k: d[k] for k in ("path", "sha256")} for d in inputs]
        spec["smoke"]["representative_basis"] = (
            "Four true active ticks across the 240-frame native Shanghai path; independent native decode plus opaque pixel identity validated; fixed 1.5x whole-stage forecast."
        )
        (HERE / f"match102-rung-{row['crf']}.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True)
    p.add_argument("--witness-sha256", required=True)
    main(p.parse_args())
