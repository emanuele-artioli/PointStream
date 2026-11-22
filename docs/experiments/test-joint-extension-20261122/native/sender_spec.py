"""Create a campus sender specification only after the pre-ordered source-coverage gate passes."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE.parents[1] / "cohort-expansion-20261122/test-physical"
FLEET = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox")
ORDER = ("match127", "match105", "match116")
ROLES = {"match127": (13, 8), "match105": (7, 0), "match116": (5, 2)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    content = json.dumps(data, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == content
    else:
        path.write_text(content)


def make(a):
    e = a.event
    crop = json.loads(a.crop_receipt.read_text())
    assert crop["event"] == e and crop["frames"] == 240 and crop["fitted_both"]
    prior = [json.loads(p.read_text()) for p in a.prior]
    assert [r["event"] for r in prior] == list(ORDER[: ORDER.index(e)])
    assert all(r["frames"] == 240 and not r["fitted_both"] for r in prior)
    roles_file = HERE.parent / (e + "-source-roles.json")
    source_roles = json.loads(roles_file.read_text())
    roles = {k: f"person:{i:012d}" for k, i in zip(("far", "near"), ROLES[e], strict=True)}
    assert set(roles.values()) == set(crop["selected_ids"]) == set(source_roles["selected"])
    assert all(crop["tracks"][v]["fitted"] == 240 for v in roles.values())
    assert source_roles["source_sha256"] == crop["source_sha256"]
    sender = json.loads((OFFICIAL / (e + "-sender-retry.json")).read_text())
    official = json.loads((OFFICIAL / (e + "-neural.json")).read_text())
    assert (
        sender["frames"] == 240
        and sender["source_sha256"] == crop["source_sha256"] == official["source_sha256"]
    )
    assert official["native_fps"] in ([60, 1], [25, 1]) and sender["fps"] == str(
        official["native_fps"][0]
    )
    reg = [
        r
        for r in map(json.loads, (OFFICIAL / "sender-retry-jobs.jsonl").read_text().splitlines())
        if r["match"] == e
    ]
    assert len(reg) == 1
    basis = json.loads((HERE.parent / (e + "-crop.json")).read_text())
    poses = {
        "name": "poses",
        "path": str(FLEET / a.crop_job / "full/published.tar"),
        "sha256": crop["output_archive_sha256"],
        "extract": False,
    }
    ball = {
        "name": "ball",
        "path": str(FLEET / reg[0]["id"] / "full/ball.payload"),
        "sha256": sender["payload_sha256"],
        "extract": False,
    }
    inputs = [
        dict(r, extract=False)
        for r in basis["staged_inputs"]
        if r["name"] in ("source", "features")
    ] + [poses, ball]
    assert {r["name"]: r["sha256"] for r in inputs} == dict(
        source=crop["source_sha256"],
        features=crop["prior_feature_archive_sha256"],
        poses=poses["sha256"],
        ball=ball["sha256"],
    )
    target_crop = HERE / (e + "-crop-receipt.json")
    if target_crop.exists():
        assert digest(target_crop) == digest(a.crop_receipt)
    else:
        target_crop.write_bytes(a.crop_receipt.read_bytes())
    binding = dict(
        event=e,
        fps=sender["fps"],
        native_width=1920,
        native_height=1080,
        source_sha256=crop["source_sha256"],
        source_features_sha256=crop["prior_feature_archive_sha256"],
        pose_archive_sha256=poses["sha256"],
        crop_receipt_sha256=digest(target_crop),
        source_role_receipt_sha256=digest(roles_file),
        roles=roles,
        paired_source_coverage={role: 240 for role in roles},
        prior_ranks_failed=[r["event"] for r in prior],
        ball_sha256=ball["sha256"],
        sender_job_id=reg[0]["id"],
        crop_job_id=a.crop_job,
    )
    save(HERE / (e + "-binding.json"), binding)
    template = json.loads(
        (
            HERE.parents[1]
            / "test-crop-joint-20261122/match102-campus-score-calibration-dispatch.json"
        ).read_text()
    )
    module = "experiments.tennis.test_extension_joint_sender"
    args = ["--event", e, "--steps", "{steps}"]
    for r in inputs:
        args += ["--" + r["name"], "{staged:" + r["name"] + "}"]
    spec = dict(
        template,
        entrypoint=["-m", module],
        validator=["{python}", "-m", module, "--validate"],
        arguments=args,
        staged_inputs=inputs,
        inputs=[{k: r[k] for k in ("path", "sha256")} for r in inputs],
        required_commands=[],
        scale={"steps": {"smoke": 3, "full": 240}},
        smoke={
            "seconds": 500,
            "representative_basis": "Seal both physical pose and appearance channels, decode the paid ball stream, then test the actual source-to-donor registration at first, middle and final native ticks.",
        },
        full={"seconds": 1600},
        validator_seconds=90,
        budget_seconds=2430,
        local_storage_gib=18,
    )
    save(HERE / (e + "-sender.json"), spec)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", choices=ORDER, required=True)
    p.add_argument("--crop-receipt", type=Path, required=True)
    p.add_argument("--crop-job", required=True)
    p.add_argument("--prior", type=Path, nargs="*", default=[])
    make(p.parse_args())
