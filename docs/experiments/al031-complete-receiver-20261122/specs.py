"""Bind the previously qualified workers to the new immutable 25-fps assets."""

import argparse
import json
from pathlib import Path

from experiments.tennis.player_requalification import sha256

HERE = Path(__file__).resolve().parent
MED = HERE.parent / "med-mixed-receiver-20261122"
OLD = HERE.parent / "reference-transfer-al028-20261122"
DATA = Path("/home/itec/emanuele/Datasets/pointstream-data")
FLEET = DATA / "jobs/fleet/inbox"


def get_paid(local, crf):
    book = json.loads((local / "result.json").read_text())
    row = next(q for q in book["rates"] if q["crf"] == crf)
    staged = {}
    current = local / f"crf{crf}"
    remote = DATA / "tennis" / ("al031-paid-" + row["joint_sha256"][:16])
    for k, name, sh in [
        ("carrier", "players.pscp", "carrier_sha256"),
        ("manifest", "players.manifest", "manifest_sha256"),
        ("joint", "rally.psj", "joint_sha256"),
        ("cache", "initial.psr", "init_sha256"),
    ]:
        assert sha256(current / name) == row[sh]
        staged[k] = dict(name=k, path=str(remote / name), sha256=row[sh], extract=False)
    return book, row, staged, current, remote


def bind(sp, inputs, args):
    sp["staged_inputs"] = inputs
    sp["inputs"] = [{k: r[k] for k in ("path", "sha256")} for r in inputs]
    sp["arguments"] = args
    sp["deadline"] = "2026-11-22T21:01:00Z"
    sp.pop("deadline_epoch", None)
    return sp


def main(a):
    book = json.loads((a.local / "result.json").read_text())
    crf = a.crf or book["selected_crf"]
    _, row, paid, cur, remote = get_paid(a.local, crf)
    crop = json.loads(
        (HERE.parent / "six-match-source-panel-20261122/alcaraz031-crop-spec.json").read_text()
    )
    ix = {r["name"]: r["path"] for r in crop["staged_inputs"]}
    denied = [
        ix["source"],
        ix["features"],
        ix["yolo11x_pose"],
        str(FLEET / "20261122T110952Z-4513f702/full/published.tar"),
        str(FLEET / a.sender / "full/ball.payload"),
    ]
    names = [r["unit"] for r in book["foreground"]]
    generated = json.loads(a.generation.read_text()) if a.generation else None
    if a.mode == "generate":
        for i, n in enumerate(names):
            sp = json.loads((MED / "000-generate-0.json").read_text())
            args = sp["arguments"][: sp["arguments"].index("--deny")]
            for k, v in [
                ("--unit", n),
                ("--carrier-sha256", row["carrier_sha256"]),
                ("--run-length", "240"),
                ("--full-cap", "2680"),
            ]:
                args[args.index(k) + 1] = v
            inputs = [
                r for r in sp["staged_inputs"] if r["name"] not in ("carrier", "manifest")
            ] + [paid["carrier"], paid["manifest"]]
            sp.update(
                full=dict(seconds=2680),
                budget_seconds=3450,
                scale=dict(ticks=dict(smoke=44, full=240)),
            )
            sp["smoke"]["representative_basis"] = (
                "44 consecutive actual native paid frames at 25 fps with identical settings, device kernels, seed and 1.5x full-cost check."
            )
            sp["gpu_models"] = ["RTX 6000 Ada"]
            sp["hosts"] = ["gpu5", "gpu6"]
            for d in denied:
                args += ["--deny", d]
            (HERE / f"{a.prefix}-generate-{i}.json").write_text(
                json.dumps(bind(sp, inputs, args), indent=2) + "\n"
            )
    elif a.mode == "finish":
        assert generated and set(generated) == set(names)
        sp = json.loads((MED / "000-finish.json").read_text())
        args = sp["arguments"][: sp["arguments"].index("--generation")]
        for k, v in [
            ("--carrier-sha256", row["carrier_sha256"]),
            ("--expected-yuv", row["decoded_sha256"]),
        ]:
            args[args.index(k) + 1] = v
        inputs = [q for q in sp["staged_inputs"] if q["name"] == "sam"] + list(paid.values())
        for i, n in enumerate(names):
            v = generated[n]
            inputs.append(
                dict(name=f"g{i}", path=v["path"], sha256=v["archive_sha256"], extract=True)
            )
            args += ["--generation", n, "{staged:g" + str(i) + "}", v["receipt_sha256"]]
        for d in denied:
            args += ["--deny", d]
        (HERE / f"{a.prefix}-finish.json").write_text(
            json.dumps(bind(sp, inputs, args), indent=2) + "\n"
        )
    elif a.mode == "pair":
        assert a.reused
        _, prior, old, _, _ = get_paid(a.reused, crf)
        assert (
            prior["init_sha256"]
            == "91bd3a3fd9b18db301348c98e5af1536b859f2e51dde71e7de56c3beee1c1af2"
        )
        for i, n in enumerate(names):
            sp = json.loads((OLD / "paired-track-4-spec.json").read_text())
            args = sp["arguments"][: sp["arguments"].index("--deny")]
            args[args.index("--unit") + 1] = n
            args += [
                "--comparison",
                "al031-fresh-prior",
                "--control-sha",
                row["carrier_sha256"],
                "--prior-sha",
                prior["carrier_sha256"],
            ]
            inputs = [
                q
                for q in sp["staged_inputs"]
                if q["name"] not in ("carrier", "manifest", "q90", "q90_manifest")
            ] + [
                paid["carrier"],
                paid["manifest"],
                {**old["carrier"], "name": "q90"},
                {**old["manifest"], "name": "q90_manifest"},
            ]
            for d in denied:
                args += ["--deny", d]
            (HERE / f"prefix-pair-{i}.json").write_text(
                json.dumps(bind(sp, inputs, args), indent=2) + "\n"
            )
    elif a.mode == "rung":
        assert a.witness and a.sha
        primary = next(v for v in book["rates"] if v["crf"] == book["selected_crf"])
        is_ref = crf == book["selected_crf"]
        old = DATA / "tennis" / ("al031-paid-" + primary["joint_sha256"][:16]) / "rally.psj"
        sp = json.loads((MED / ("000-rung-60.json" if is_ref else "000-rung-70.json")).read_text())
        args = sp["arguments"][: sp["arguments"].index("--deny")]
        args[args.index("--carrier-sha256") + 1] = row["carrier_sha256"]
        args[args.index("--native-sha") + 1] = row["decoded_sha256"]
        inputs = list(paid.values())
        if not is_ref:
            inputs.append(
                dict(name="original", path=str(old), sha256=primary["joint_sha256"], extract=False)
            )
        inputs.append(
            dict(
                name="witness",
                path=str(FLEET / a.witness / "full/published.tar"),
                sha256=a.sha,
                extract=True,
            )
        )
        for d in denied:
            args += ["--deny", d]
        (HERE / f"{a.prefix}-rung-{crf}.json").write_text(
            json.dumps(bind(sp, inputs, args), indent=2) + "\n"
        )
    print(
        json.dumps(
            dict(
                local=str(cur), remote=str(remote), roles={k: q["sha256"] for k, q in paid.items()}
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["stage", "generate", "finish", "pair", "rung"])
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--sender", required=True)
    p.add_argument("--generation", type=Path)
    p.add_argument("--reused", type=Path)
    p.add_argument("--crf", type=int)
    p.add_argument("--witness")
    p.add_argument("--sha")
    main(p.parse_args())
