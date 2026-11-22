"""Score native foreground receivers on campus using lossless, signed evaluator pixels."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
import sys
from pathlib import Path
import time

from experiments.tennis.player_requalification import sha256

HERE = Path(__file__).resolve().parents[2]
SCORE_PATH = HERE / "docs/experiments/med-mixed-receiver-20261122/score.py"


def read(p):
    return json.loads(p.read_text())


def arm(a):
    if a.kind == "neural":
        pub = a.data / "publish"
        native = read(pub / "result.json")
        proof = read(pub / "reports.json")
        r = native["native"]
        w, h = r["size"]
        route = pub / "decoded.yuv"
        passes = [v for d in proof["decode"].values() for v in d["passes"]]
        assert len(passes) == 4 and all(
            v["frame_sha256"] == native["yuv_frames"] == v["yuv420_sha256"] for v in passes
        )
        assert (
            not r["dpb_reachable"] and r["accepted"] and r["qp"] == a.qp and native["frames"] == 240
        )
        assert native["source_sha256"] == a.source_sha and native["native_fps"] == [a.fps, 1]
        assert (
            sha256(pub / "stream.psdc") == r["container_sha256"]
            and (pub / "stream.psdc").stat().st_size == r["bytes"]
        )
        assert (
            r["max_decode_diff"] == 0
            and r["identical_frame_hashes"]
            and r["within_process_deterministic"] == [True, True]
        )
        return dict(
            name=a.label,
            kind="native_yuv420p",
            path=str(route),
            sha256=native["raw_sha256"],
            width=w,
            height=h,
            bytes=r["bytes"],
            kbps=r["kbps"],
            container_sha256=r["container_sha256"],
            receipt_sha256=sha256(pub / "result.json"),
            producer_job=a.producer,
            validated_native_passes=4,
        )
    if a.kind == "finish":
        pub = a.data / "publish"
        receipt = pub / "finish.json"
        route = pub / "frames.npy"
        r = read(receipt)
        dig = r["frames_file_sha256"]
        kind = "rgb_npy"
    else:
        receipt = a.receipt
        route = a.data
        r = read(receipt)
        dig = r["frames_sha256"]
        kind = "rgb_zseq"
    assert r["ticks"] == list(range(240)) and [z["tick"] for z in r["rows"]] == list(range(240))
    assert r["isolation"]["interfaces"] == ["lo"] and min(r["isolation"]["denials"].values()) >= 1
    wire = r["wire"]
    assert (
        wire["joint_sha256"] == a.expected_joint
        and wire["joint_bytes"] == a.paid
        and wire["cache_bytes"] == a.cache
    )
    return dict(
        name=a.label,
        kind=kind,
        path=str(route),
        sha256=dig,
        receipt=str(receipt),
        receipt_sha256=sha256(receipt),
        width=1920,
        height=1080,
        bytes=a.paid,
        kbps=a.paid * 8 * a.fps / 240000,
        cold_kbps=(a.cache + a.paid) * 8 * a.fps / 240000,
        joint_sha256=a.expected_joint,
        receiver_job=a.producer,
    )


def verify(a, report, location):
    n = a.frames
    score = report["score"]
    needed = 2 * n if a.clip == "al031" else 2 * sum(0 <= tick < n for tick in range(10, 240, 20))
    assert n in (24, 240)
    identity = report["arm"]
    assert identity["name"] == a.label and score["arm"] == identity
    checks = dict(
        real_native=score["frames"] == n and score["native_fps"] == a.fps,
        signed_source=score["source_sha256"] == a.source_sha,
        body_count=len(score["body_rows"]) == needed,
        source_self=score["source_self_abs"] < 1e-5,
        full_frame=0 < score["rgb_psnr"] < 80,
        exact_wire=identity["bytes"] == a.paid
        and abs(identity["kbps"] - a.paid * 8 * a.fps / 240000) < 1e-8,
        cost=report["worker_seconds"] < a.full_cap
        and (n == 240 or 1.5 * report["worker_seconds"] * 240 / n + 70 < a.full_cap),
    )
    if a.clip == "al031":
        checks["ball_locations"] = score["ball"]["total_ticks"] == n and score["ball"][
            "present"
        ] == len(score["ball"]["rows"])
    if a.expected_dists is not None and n == 240:
        checks["cross_platform_calibration"] = abs(score["mean_dists"] - a.expected_dists) < 0.0015
    result = dict(passed=all(checks.values()), checks=checks)
    location.write_text(json.dumps(result) + "\n")
    assert result["passed"], result


def run(a):
    stamp = time.monotonic()
    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    assert sha256(a.source) == a.source_sha
    spec = spec_from_file_location("masked_native_scorer", SCORE_PATH)
    assert spec and spec.loader
    scorer = module_from_spec(spec)
    spec.loader.exec_module(scorer)
    s = arm(a)
    f = scratch / "score-physical-arm.json"
    f.write_text(json.dumps([s]))
    out = scratch / "scored"
    grid = read(a.canonical_receipt)
    assert grid["source_sha256"] == a.source_sha and grid["fps"] == a.fps
    args = argparse.Namespace(
        source=a.source,
        clip=a.clip,
        weights=a.weights,
        supports=a.supports / "supports.json",
        arms=f,
        out=out,
        canonical=a.canonical,
        canonical_receipt=a.canonical_receipt,
        ball_payload=a.ball_payload,
        ball_receipt=a.ball_receipt,
        frame_count=a.frames,
        threads=1,
        sample_dir=stage / "sampled-crops"
        if a.kind in ("finish", "rung") and a.frames == 240
        else None,
    )
    scorer.main(args)
    report = dict(
        job=a.producer,
        source_sha=sha256(a.source),
        canonical_sha=sha256(a.canonical),
        support_bundle=grid["supports_tar_sha256"],
        arm=s,
        score=read(out / (a.label + ".json")),
        worker_seconds=time.monotonic() - stamp,
    )
    (stage / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    verify(a, report, stage / "native_score_validation.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ["source", "weights", "data", "supports", "canonical", "canonical-receipt"]:
        p.add_argument("--" + k, required=True, type=Path)
    for k in ["ball-payload", "ball-receipt", "receipt"]:
        p.add_argument("--" + k, type=Path)
    for k in ["source-sha", "label", "producer", "clip"]:
        p.add_argument("--" + k, required=True)
    for k in ["fps", "frames", "paid"]:
        p.add_argument("--" + k, type=int, required=True)
    p.add_argument("--kind", choices=["neural", "finish", "rung"], required=True)
    p.add_argument("--expected-joint")
    p.add_argument("--cache", type=int)
    p.add_argument("--qp", type=int)
    p.add_argument("--expected-dists", type=float)
    p.add_argument("--full-cap", type=int, default=1600)
    p.add_argument("--validate", action="store_true")
    if sys.argv[1:] == ["--validate"]:
        stage = Path(os.environ["PS_STAGE_DIR"])
        command = read(stage / "dispatch.json")["command"]
        assert command[1:3] == ["-m", "experiments.tennis.foreground_campus_score"]
        verify(
            p.parse_args(command[3:]),
            read(stage / "result.json"),
            Path(os.environ["PS_VALIDATION_PATH"]),
        )
    else:
        run(p.parse_args())
