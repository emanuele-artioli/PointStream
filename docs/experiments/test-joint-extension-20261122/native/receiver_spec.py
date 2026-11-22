"""Bind an authenticated external sender to the fixed, source-denied native receiver."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHANGHAI = HERE.parents[1] / "test-crop-joint-20261122"
INBOX = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")


def save(path, data):
    text = json.dumps(data, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == text
    else:
        path.write_text(text)


def make(a):
    e = a.event
    source = json.loads((HERE / (e + "-binding.json")).read_text())
    request = json.loads((HERE / (e + "-sender.json")).read_text())
    result = json.loads(a.sender.read_text())
    validation = json.loads(a.validation.read_text())
    receipts = json.loads(a.sender_receipt.read_text())
    assert source["source_sha256"] == result["source_sha256"] == receipts["source"]
    assert receipts["event"] == e and receipts["source_verified"] and receipts["steps"] == 240
    assert validation["passed"] and validation["checks"]["output_steps"] == 240
    assert receipts["selected_crf"] == result["selected_crf"]
    assert result["native_fps"] == source["fps"] and result["frames"] == 240
    assert [r["unit"] for r in result["foreground"]] == [e + "_far", e + "_near"]
    allrows = sorted(result["rates"], key=lambda r: r["kbps_cached"])
    selected = min(allrows, key=lambda r: (abs(r["kbps_cached"] - 20), r["kbps_cached"]))
    assert selected["crf"] == result["selected_crf"]
    verified = next(r for r in receipts["rungs"] if r["crf"] == selected["crf"])
    assert verified["covered_ticks"] == 240
    inputs = {}
    for key, fn, h in (
        ("carrier", "players.pscp", "carrier_sha256"),
        ("manifest", "players.manifest", "manifest_sha256"),
        ("joint", "rally.psj", "joint_sha256"),
        ("cache", "initial.psr", "init_sha256"),
    ):
        file = f"crf{selected['crf']}-{fn}"
        assert verified["published"][file] == selected[h]
        inputs[key] = dict(
            name=key, path=str(INBOX / a.job / "full" / file), sha256=selected[h], extract=False
        )
    deny = [s["path"] for s in request["staged_inputs"]]
    # The source pose detector's weights and source-only masks are not receiver inputs.
    pose = next(
        s
        for s in json.loads((HERE.parent / (e + "-crop.json")).read_text())["staged_inputs"]
        if s["name"] == "yolo11x_pose"
    )
    deny.append(pose["path"])
    if a.generation is None:
        for role in ("far", "near"):
            template = json.loads((SHANGHAI / "match102-generation-0-ada-64.json").read_text())
            args = template["arguments"][: template["arguments"].index("--deny")]
            args[args.index("--unit") + 1] = e + "_" + role
            args[args.index("--carrier-sha256") + 1] = selected["carrier_sha256"]
            names = {"carrier", "manifest"}
            template["staged_inputs"] = [
                s for s in template["staged_inputs"] if s["name"] not in names
            ] + [inputs[k] for k in sorted(names)]
            template["smoke"]["representative_basis"] = (
                "64 consecutive original native ticks with the fixed paid reference and poses, 30 denoising steps, 1.5x full-stage projection, 90 s setup guard, 60 s teardown and 2680 s cap."
            )
            template["arguments"] = args + ["--smoke-frames", "64"]
            finish(template, deny, HERE / (e + "-generate-" + role + ".json"))
    else:
        seals = json.loads(a.generation.read_text())
        assert set(seals) == {e + "_far", e + "_near"}
        template = json.loads((SHANGHAI / "match102-finish.json").read_text())
        template["arguments"] = template["arguments"][: template["arguments"].index("--generation")]
        args = template["arguments"]
        for k, v in (
            ("--carrier-sha256", selected["carrier_sha256"]),
            ("--expected-yuv", selected["decoded_sha256"]),
        ):
            args[args.index(k) + 1] = v
        sam = next(s for s in template["staged_inputs"] if s["name"] == "sam")
        template["staged_inputs"] = [sam, *inputs.values()]
        for i, role in enumerate(("far", "near")):
            unit = e + "_" + role
            seal = seals[unit]
            template["staged_inputs"].append(
                dict(name=f"g{i}", path=seal["path"], sha256=seal["archive_sha256"], extract=True)
            )
            args += ["--generation", unit, "{staged:g" + str(i) + "}", seal["receipt_sha256"]]
        finish(template, deny, HERE / (e + "-finish.json"))


def finish(spec, deny, path):
    paths = {r["path"] for r in spec["staged_inputs"]}
    assert len(paths) == len(spec["staged_inputs"]) and not paths.intersection(deny)
    for p in deny:
        spec["arguments"] += ["--deny", p]
    spec["inputs"] = [{k: r[k] for k in ("path", "sha256")} for r in spec["staged_inputs"]]
    save(path, spec)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", choices=("match127", "match105", "match116"), required=True)
    p.add_argument("--job", required=True)
    for key in ("sender", "sender-receipt", "validation"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--generation", type=Path)
    make(p.parse_args())
