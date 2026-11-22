"""Price ordinary, neural and paid reconstructions from immutable native-rate receipts."""

import argparse
import hashlib
import json
from pathlib import Path

from receiver_spec import INBOX, HERE, SHANGHAI, save

BANK = HERE.parents[1] / "cohort-expansion-20261122/test-physical"


def registry(event, kind):
    rows = [
        r
        for r in map(json.loads, (BANK / (kind + "-jobs.jsonl")).read_text().splitlines())
        if r["match"] == event
    ]
    assert len(rows) == 1
    return rows[0]["id"]


def write(a):
    e = a.event
    b = json.loads((HERE / (e + "-binding.json")).read_text())
    sender = json.loads(a.sender.read_text())
    assert (
        sender["source_sha256"] == b["source_sha256"]
        and sender["native_fps"] == b["fps"]
        and sender["frames"] == 240
    )
    sel = next(x for x in sender["rates"] if x["crf"] == sender["selected_crf"])
    original = json.loads((HERE / (e + "-sender.json")).read_text())
    template = json.loads(
        (SHANGHAI / "match102-campus-score-calibration-dispatch.json").read_text()
    )
    weights = next(r for r in template["staged_inputs"] if r["name"] == "weights")
    source = [r for r in original["staged_inputs"] if r["name"] in ("source", "features")]
    ordinary = json.loads((BANK / (e + "-rate.json")).read_text())
    orc = json.loads((BANK / (e + "-score.json")).read_text())
    nn = json.loads((BANK / (e + "-neuroscore-fixed.json")).read_text())
    assert orc["source_sha256"] == nn["source_sha256"] == b["source_sha256"]
    rateid = registry(e, "rate")
    nnid = registry(e, "neural")
    priced = []
    selected = {}
    for row in sender["rates"]:
        for state, total in (
            ("warm", row["joint_bytes"]),
            ("cold", row["joint_bytes"] + row["init_bytes"]),
        ):
            for w in sorted({p["width"] for p in ordinary["rows"]}):
                pts = [p for p in ordinary["rows"] if p["width"] == w and p["bytes"] <= total]
                if not pts:
                    priced.append(
                        dict(crf=row["crf"], state=state, ceiling=total, width=w, arm=None)
                    )
                    continue
                pt = max(pts, key=lambda p: p["bytes"])
                name = "av1-" + pt["native_file"].split(".")[0]
                priced.append(
                    dict(
                        crf=row["crf"],
                        state=state,
                        ceiling=total,
                        width=w,
                        arm=name,
                        bytes=pt["bytes"],
                    )
                )
                selected[name] = dict(
                    producer=rateid, sha256=orc["rate_archive_sha256"], file="published.tar"
                )
    selected["dcvc-480-q24"] = dict(
        producer=nnid, sha256=nn["neural_archive_sha256"], file="published.tar"
    )
    nn36 = BANK / (e + "-neuroscore-q36.json")
    if nn36.exists():
        additional = json.loads(nn36.read_text())
        assert additional["source_sha256"] == b["source_sha256"]
        selected["dcvc-480-q36"] = dict(
            producer=registry(e, "neural-q36"),
            sha256=additional["neural_archive_sha256"],
            file="published.tar",
        )
    save(
        HERE / (e + "-price-controls.json"),
        dict(source=b["source_sha256"], prices=priced, selected=selected),
    )
    if a.joint:
        assert sel["crf"] == a.crf or a.rung_receipt is not None
        assert a.joint_sha and len(a.joint_sha) == 64
        selected = {
            f"joint-crf{a.crf}": dict(
                producer=a.joint,
                sha256=a.joint_sha,
                file="frames.zseq" if a.rung_receipt else "published.tar",
            )
        }
    module = "experiments.tennis.test_extension_native_score"
    for arm, part in selected.items():
        j = part["producer"]
        joint = arm.startswith("joint-crf")
        inp = [
            *source,
            weights,
            dict(
                name="data",
                path=str(INBOX / j / "full" / part["file"]),
                sha256=part["sha256"],
                extract=joint and not a.rung_receipt,
            ),
        ]
        args = [
            "--event",
            e,
            "--arm",
            arm,
            "--producer",
            j,
            "--frames",
            "{frames}",
            "--full-cap",
            "1600",
        ]
        if joint:
            ss = dict(
                name="sender",
                path=str(INBOX / a.sender_job / "full/sender-result.json"),
                sha256=hashlib.sha256(a.sender.read_bytes()).hexdigest(),
                extract=False,
            )
            inp.append(ss)
            if a.rung_receipt:
                assert len(a.rung_receipt) == 64
                inp.append(
                    dict(
                        name="receipt",
                        path=str(INBOX / j / "full/rung.json"),
                        sha256=a.rung_receipt,
                        extract=False,
                    )
                )
        for item in inp:
            args += ["--" + item["name"], "{staged:" + item["name"] + "}"]
        spec = dict(
            template,
            entrypoint=["-m", module],
            validator=["{python}", "-m", module, "--validate"],
            arguments=args,
            staged_inputs=inp,
            inputs=[{k: r[k] for k in ("path", "sha256")} for r in inp],
            smoke=dict(
                seconds=420,
                representative_basis="First 24 real native source ticks. Verify the full original feature mask digest, two body supports on each tick, genuine codec pixels, and timing with 1.5x scaled margin.",
            ),
            full=dict(seconds=1600),
            validator_seconds=90,
            budget_seconds=2280,
            single_cpu_campaign=True,
        )
        save(HERE / (e + "-score-" + arm + ".json"), spec)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", choices=("match127", "match105", "match116"), required=True)
    p.add_argument("--sender", type=Path, required=True)
    p.add_argument("--sender-job")
    p.add_argument("--joint")
    p.add_argument("--joint-sha")
    p.add_argument("--crf", type=int, choices=(50, 60, 70))
    p.add_argument("--rung-receipt")
    write(p.parse_args())
