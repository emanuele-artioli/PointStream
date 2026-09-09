"""Bounded pilot -> confirmation -> final paired codec ladders.

Policy JSON supplies explicit QP/JPEG bounds, measurement bands, source scenes,
frame counts and wall-time budgets. No defaults pretend to be calibrated bounds.
Each pair runs in a bounded subprocess and is saved before another starts.
"""

from __future__ import annotations

import sqlite3  # noqa: F401
import argparse
import hashlib
import fcntl
import json
import math
import signal
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from experiments.jobs.monitor import publish_progress, read_json, write_json, stop_child


def validate(policy: dict[str, Any]) -> None:
    for key in ("budget_seconds", "pair_timeout_seconds", "min_gap_db", "qp_step"):
        if not math.isfinite(policy[key]) or policy[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if policy["pair_timeout_seconds"] > 3500:
        raise ValueError("pair timeout must be <=3500s to save stage state at least hourly")
    frames = [policy[k] for k in ("pilot_frames", "confirmation_frames", "final_frames")]
    if not 2 <= frames[0] < frames[1] <= frames[2]:
        raise ValueError("require 2 <= pilot < confirmation <= final frames")
    if not policy["scenes"] or len(set(map(tuple, policy["scenes"]))) != len(policy["scenes"]):
        raise ValueError("provide unique [video, scene] pairs")
    for key in (
        "max_adjustments",
        "pilot_frames",
        "confirmation_frames",
        "final_frames",
        "qp_step",
    ):
        if type(policy[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if not all(type(q) is int for q in policy["qps"]):
        raise ValueError("QPs must be integers")
    if policy["max_adjustments"] < 0:
        raise ValueError("max_adjustments must be nonnegative")
    if policy["sweep"] not in {"qp", "payload"}:
        raise ValueError("adaptive control supports qp and payload sweeps")
    qps = policy["qps"]
    if len(qps) < 3 or sorted(set(qps)) != qps:
        raise ValueError("provide at least three strictly increasing QPs")
    if not policy["qp_min"] <= qps[0] < qps[-1] <= policy["qp_max"]:
        raise ValueError("QPs exceed approved bounds")
    if policy["sweep"] == "payload":
        jpegs = policy["jpegs"]
        if type(policy["jpeg_step"]) is not int or policy["jpeg_step"] <= 0:
            raise ValueError("jpeg_step must be a positive integer")
        if not all(type(q) is int for q in jpegs):
            raise ValueError("JPEG qualities must be integers")
        if len(jpegs) != len(qps) or sorted(set(jpegs), reverse=True) != jpegs:
            raise ValueError("payload needs one decreasing JPEG quality per QP")
        if not 1 <= policy["jpeg_min"] <= min(jpegs) <= max(jpegs) <= policy["jpeg_max"] <= 100:
            raise ValueError("JPEG bounds invalid")
    for key in ("psnr_dB", "coded_bytes", "seconds"):
        low, high = policy["bands"][key]
        if not all(math.isfinite(x) for x in (low, high)) or low >= high:
            raise ValueError(f"invalid two-sided band for {key}")
    if not policy.get("dataset_revision"):
        raise ValueError("dataset_revision is required for reproducible selection")
    if not policy.get("bounds_basis") or not policy.get("controls_evidence"):
        raise ValueError("bounds rationale and control evidence path required")


def assess(pair: dict[str, Any], policy: dict[str, Any]) -> str:
    """Return pass, clustered, or attention. Missing evidence never passes."""
    if pair.get("failures") or pair.get("bound_alarms") or pair.get("rungs_excluded_not_a_rate"):
        return "attention"
    clustered = False
    for arm in ("anchor_rungs", "pointstream_rungs"):
        rows = pair.get(arm, [])
        if len(rows) != len(policy["qps"]):
            return "attention"
        rates = sorted(row.get("residual_rate", row.get("rate_value", -1)) for row in rows)
        if rates != policy["qps"]:
            return "attention"
        if arm == "pointstream_rungs" and policy["sweep"] == "payload":
            actual = sorted(
                (row.get("residual_rate"), row.get("background_jpeg_quality")) for row in rows
            )
            if actual != list(zip(policy["qps"], policy["jpegs"])):
                return "attention"
        for row in rows:
            for key, (low, high) in policy["bands"].items():
                value = row.get(key)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not low <= value <= high
                ):
                    return "attention"
        if len({row["coded_bytes"] for row in rows}) == 1:
            return "attention"
        # The planned order is increasing QP: quality and bytes should decrease.
        # Equal quality with meaningfully different bytes still needs review;
        # never call it an inferior model or discard a valid RD tradeoff.
        ordered = sorted(rows, key=lambda row: row["rate_value"])
        for left, right in zip(ordered, ordered[1:]):
            if left["psnr_dB"] < right["psnr_dB"] or left["coded_bytes"] < right["coded_bytes"]:
                return "attention"
            if left["psnr_dB"] - right["psnr_dB"] < policy["min_gap_db"]:
                clustered = True
    return "clustered" if clustered else "pass"


def widen(policy: dict[str, Any]) -> dict[str, Any] | None:
    """Spread all points across a wider bounded interval, keeping point count."""
    result = dict(policy)
    qps = policy["qps"]
    lo = max(policy["qp_min"], qps[0] - policy["qp_step"])
    hi = min(policy["qp_max"], qps[-1] + policy["qp_step"])
    result["qps"] = [round(lo + i * (hi - lo) / (len(qps) - 1)) for i in range(len(qps))]
    if policy["sweep"] == "payload":
        jpegs = policy["jpegs"]
        hi_j = min(policy["jpeg_max"], jpegs[0] + policy["jpeg_step"])
        lo_j = max(policy["jpeg_min"], jpegs[-1] - policy["jpeg_step"])
        result["jpegs"] = [
            round(hi_j - i * (hi_j - lo_j) / (len(qps) - 1)) for i in range(len(qps))
        ]
    return None if result == policy else result


def worker(request_path: Path, output: Path) -> None:
    from experiments.tier.clip import load_tier_clip
    from experiments.tier.ladder import pair_for_codec
    from src.runner.config_io import load_tier

    request = read_json(request_path)
    policy = request["policy"]
    clip = load_tier_clip(
        video=request["video"], scene=request["scene"], n_frames=request["frames"]
    )
    result = pair_for_codec(
        clip,
        load_tier(policy["tier"]),
        codec_name=policy["codec"],
        rungs=tuple(policy["qps"]),
        sweep=policy["sweep"],
        payload_rungs=tuple(zip(policy.get("jpegs", []), policy["qps"])) or None,
    )
    write_json(output, result)


def run(policy_path: Path, directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(policy_path, directory)


def _run_locked(policy_path: Path, directory: Path) -> int:
    policy = read_json(policy_path)
    validate(policy)
    # Control evidence is inspected before opening measured ladder results.
    controls = read_json(Path(policy["controls_evidence"]))
    if (
        not controls
        or controls.get("calibrated") is not True
        or controls.get("null_checked") is not True
    ):
        raise ValueError("control record must attest calibrated and null_checked; see protocol")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source = (
        Path(__file__).read_bytes()
        + (Path(__file__).parents[1] / "tier" / "ladder.py").read_bytes()
    )
    digest = hashlib.sha256(
        json.dumps([policy, controls, revision], sort_keys=True).encode() + source
    ).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    saved = read_json(directory / "state.json")
    if saved and saved["policy_hash"] != digest:
        raise ValueError("policy changed: use a new campaign directory")
    state = saved or {
        "policy_hash": digest,
        "policy": policy,
        "phase": "pilot",
        "adjustments": 0,
        "spent_seconds": 0.0,
        "completed": [],
        "decisions": [],
    }
    if state.get("status") in {"attention", "complete"}:
        print(f"campaign {state['status']}; inspect {directory / 'state.json'}")
        return 0 if state["status"] == "complete" else 2
    active = state["policy"]

    def save(reason: str | None = None) -> None:
        write_json(directory / "state.json", state)
        publish_progress(state["phase"], len(state["completed"]), decision=reason)

    def pause(reason: str) -> int:
        state.update(status="attention", reason=reason)
        state["decisions"].append({"action": "pause", "reason": reason})
        save(reason)
        return 2

    save()
    while True:
        verdicts = []
        for index, (video, scene) in enumerate(active["scenes"]):
            name = f"{state['phase']}-{state['adjustments']}-{index}"
            output = directory / f"{name}.json"
            request_path = directory / f"{name}.request.json"
            if name not in state["completed"]:
                remaining = active["budget_seconds"] - state["spent_seconds"]
                if remaining < active["pair_timeout_seconds"]:
                    return pause("remaining budget cannot cover another bounded pair")
                if state.get("inflight"):
                    return pause("previous worker interrupted; inspect outputs before resuming")
                write_json(
                    request_path,
                    {
                        "policy": active,
                        "video": video,
                        "scene": scene,
                        "frames": active[f"{state['phase']}_frames"],
                    },
                )
                state["inflight"] = name
                save()
                started = time.monotonic()
                with (directory / f"{name}.log").open("a") as log:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "experiments.jobs.codec",
                            "worker",
                            str(request_path),
                            str(output),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        code = child.wait(timeout=active["pair_timeout_seconds"])
                    except subprocess.TimeoutExpired:
                        code = -1
                    finally:
                        stop_child(child)
                state["spent_seconds"] += time.monotonic() - started
                state.pop("inflight")
                if code != 0 or not output.exists():
                    return pause(f"{name}: worker failed/timed out; no promotion")
                state["completed"].append(name)
                save()
            verdicts.append(assess(read_json(output), active))
        if "attention" in verdicts:
            return pause("invalid, nonmonotone or out-of-band result; investigate instrument")
        if "clustered" in verdicts:
            if state["phase"] != "pilot":
                return pause("quality spacing did not transfer to longer clips")
            updated = widen(active)
            if state["adjustments"] >= active["max_adjustments"] or updated is None:
                return pause("quality spacing unresolved within approved adjustment limits")
            active = updated
            state.update(policy=active, adjustments=state["adjustments"] + 1)
            state["decisions"].append(
                {"action": "widen", "qps": active["qps"], "jpegs": active.get("jpegs")}
            )
        elif state["phase"] == "final":
            state.update(
                status="complete",
                citable=False,
                reason="exploratory selection; frozen full-metric held-out confirmation still required",
            )
            save()
            return 0
        else:
            state["phase"] = "confirmation" if state["phase"] == "pilot" else "final"
            state["decisions"].append({"action": "advance", "phase": state["phase"]})
        save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError("campaign interrupted; preserve inflight state")

    signal.signal(signal.SIGTERM, interrupted)
    if args.action == "worker":
        worker(args.input, args.output)
        return 0
    return run(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
