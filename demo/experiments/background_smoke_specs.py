"""Schema-1 fleet specs for the background smoke, checked against plan caps.

Specs are written outside the code tree. ``check_plan`` enforces this plan's
limits on top of ``experiments.jobs.inbox.validate_spec``: at most 480 s per
job, at most 2400 s of submitted budget in total, stalls inside the budget,
and a substantive validator for every request.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any

from demo.experiments.background_smoke_core import DATA_ROOT, is_sha256

JOB_BUDGET_SECONDS = 480
# User approved the continuation amendment on 4 October 2026. Prior attempts
# remain charged; model-stage and per-job caps are unchanged.
PLAN_BUDGET_SECONDS = 2400
PREPARATION_BUDGET_SECONDS = 1500
HOSTS = ["gpu3", "gpu5"]
GPU_MODELS = ["RTX A6000", "RTX 6000 Ada"]
MODEL_MEMORY_MIB = 24000
OVERHEAD_SECONDS = 20

PLANS: dict[str, dict[str, Any]] = {
    "inventory": {
        "smoke_parts": "preview,frames-f001c3,ckpt-image,ckpt-htl,env",
        "full_parts": "ckpt-ld,ckpt-hts,frames-f002,ckpt-f002-htl,ckpt-hnerv,training",
        "smoke_seconds": 170, "full_seconds": 120, "memory": 1024,
        "basis": "B0/B1 on the primary cut: DCVC/HNeRV source identity, the saved preview, hold-out 120..151 and the HT-L checkpoints the first codec case loads",
        "required_commands": ["git"],
    },
    "codec": {
        "smoke_parts": "f001c3-htl-pre,f001c3-htl-ft",
        "full_parts": "f001c3-ld-pre,f001c3-ld-ft,f001c3-hts-pre,f001c3-hts-ft,f001c3-av1",
        "smoke_seconds": 170, "full_seconds": 230, "memory": MODEL_MEMORY_MIB,
        "basis": "B2 primary HT-L pretrained then fine-tuned at QP 21 on hold-out 120..127: real bitstreams, independent decode, RGB/LPIPS/temporal metrics",
        "required_commands": [],
    },
    "latent": {
        "smoke_parts": "f001c3-b6",
        "full_parts": "f002-b6",
        "smoke_seconds": 200, "full_seconds": 170, "memory": MODEL_MEMORY_MIB,
        "basis": "B4 primary cut 120..151: frozen HNeRV encode once, 6-bit packets in 1/8/32-frame segments, decode from packets alone",
        "required_commands": [],
    },
    "drift": {
        "smoke_parts": "f001c3-ld-pre",
        "full_parts": "f001c3-ld-ft",
        "smoke_seconds": 110, "full_seconds": 140, "memory": MODEL_MEMORY_MIB,
        "basis": "B3 LD pretrained, 120..151: one 32-frame stream versus four reset 8-frame streams",
        "required_commands": [],
    },
}


def build_spec(kind: str, *, inputs: list[dict[str, str]], manifests: list[str], deadline: str, plan: dict[str, Any] | None = None, codec_evidence: str | None = None) -> dict[str, Any]:
    plan = plan or PLANS[kind]
    arguments = [kind, "--parts", "{parts}", "--stage-seconds", "{stage_seconds}"]
    for manifest in manifests:
        arguments += ["--manifest", manifest]
    if kind == "drift":
        if not codec_evidence or not any(item["path"] == codec_evidence for item in inputs):
            raise ValueError("drift requires hash-pinned B2 codec evidence in inputs")
        arguments += ["--codec-evidence", codec_evidence]
    validator_seconds = 30
    budget = plan["smoke_seconds"] + plan["full_seconds"] + validator_seconds + OVERHEAD_SECONDS
    return {
        "schema": 1, "hosts": HOSTS, "gpu_models": GPU_MODELS, "gpu_memory_mib": plan["memory"], "cpu_threads": 4,
        "entrypoint": ["-m", "demo.experiments.background_smoke"], "arguments": arguments,
        "scale": {
            "parts": {"smoke": plan["smoke_parts"], "full": plan["full_parts"]},
            "stage_seconds": {"smoke": plan["smoke_seconds"], "full": plan["full_seconds"]},
        },
        "inputs": inputs,
        "smoke": {"seconds": plan["smoke_seconds"], "representative_basis": plan["basis"]},
        "full": {"seconds": plan["full_seconds"]},
        "validator": ["{python}", "-m", "demo.experiments.background_smoke", "validate", "--kind", kind],
        "validator_seconds": validator_seconds,
        "required_commands": plan["required_commands"],
        "budget_seconds": budget, "stall_seconds": budget, "deadline": deadline,
    }


def check_plan(specs: list[dict[str, Any]], *, spent_seconds: float = 0.0, spent_inventory_seconds: float = 0.0, spent_by_kind: dict[str, float] | None = None) -> dict[str, Any]:
    """Plan caps beyond the dispatcher schema. Raises ValueError on violation."""
    if not math.isfinite(spent_seconds) or spent_seconds < 0 or not math.isfinite(spent_inventory_seconds) or spent_inventory_seconds < 0:
        raise ValueError("spent budgets must be finite and nonnegative")
    if spent_inventory_seconds > spent_seconds:
        raise ValueError("inventory spend cannot exceed total spend")
    total = spent_seconds
    per_kind = {"inventory": spent_inventory_seconds, "codec": 0, "drift": 0, "latent": 0}
    limits = {"inventory": PREPARATION_BUDGET_SECONDS, "codec": 720, "drift": 300, "latent": 480}
    for kind, seconds in (spent_by_kind or {}).items():
        if kind not in limits or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("prior stage spend must have planned kinds and finite nonnegative values")
        per_kind[kind] += seconds
    attributed = sum(per_kind.values())
    for spec in specs:
        from experiments.jobs.inbox import validate_spec
        try:
            validate_spec(spec)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        kind = spec["arguments"][0]
        if kind not in limits:
            raise ValueError("unplanned diagnostic kind")
        budget = spec["budget_seconds"]
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("job budget must be finite and positive")
        per_kind[kind] += budget
        if budget > JOB_BUDGET_SECONDS:
            raise ValueError(f"job budget {budget}s exceeds the {JOB_BUDGET_SECONDS}s per-job cap")
        reserved = spec["smoke"]["seconds"] + spec["full"]["seconds"] + spec.get("validator_seconds", 60)
        if reserved > budget:
            raise ValueError("budget does not reserve smoke, full and validation")
        if spec.get("stall_seconds", 1800) > budget:
            raise ValueError("stall_seconds exceeds the task budget")
        for item in spec["inputs"]:
            if not is_sha256(item.get("sha256")) or not str(item.get("path", "")).startswith(str(DATA_ROOT) + "/"):
                raise ValueError("every input needs an absolute data-root path and a real SHA-256")
        validator = spec["validator"]
        if "validate" not in validator or "--kind" not in validator or "-c" in validator:
            raise ValueError("validator must be the workload's substantive gate, not inline code")
        if "-c" in spec["entrypoint"] or "-c" in spec["arguments"]:
            raise ValueError("inline code is not allowed")
        for name, values in spec["scale"].items():
            if "{" + name + "}" not in spec["arguments"]:
                raise ValueError(f"scale parameter {name} does not occupy an argument")
            if name == "parts" and values["smoke"] == values["full"]:
                raise ValueError("full must be the remainder of the diagnostic, not a repeat of the smoke")
        total += budget
    if total > PLAN_BUDGET_SECONDS:
        raise ValueError(f"submitted budgets total {total:.0f}s, above the {PLAN_BUDGET_SECONDS}s plan ceiling")
    if not math.isclose(attributed, spent_seconds):
        raise ValueError("all prior spend must be attributed to diagnostic kinds")
    for kind, seconds in per_kind.items():
        if seconds > limits[kind]:
            raise ValueError(f"{kind} budgets total {seconds:g}s, above the {limits[kind]}s stage ceiling")
    return {"jobs": len(specs), "budget_seconds_total": total, "remaining_seconds": PLAN_BUDGET_SECONDS - total}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(PLANS))
    parser.add_argument("--input", action="append", required=True, help="PATH=SHA256 under the data root")
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--deadline-minutes", type=int, default=90)
    parser.add_argument("--codec-evidence", help="hash-pinned B2 result.json required for drift")
    parser.add_argument("--spent-by-kind", type=json.loads, default={}, help="JSON of prior diagnostic reservations by kind; together with inventory spend must sum to --spent-seconds")
    parser.add_argument("--spent-inventory-seconds", type=float, default=0.0)
    parser.add_argument("--spent-seconds", type=float, default=0.0, help="budget already submitted by earlier jobs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from experiments.jobs.inbox import validate_spec

    inputs = [{"path": item.split("=", 1)[0], "sha256": item.split("=", 1)[1]} for item in args.input]
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=args.deadline_minutes)).isoformat(timespec="seconds")
    spec = build_spec(args.kind, inputs=inputs, manifests=args.manifest, deadline=deadline, codec_evidence=args.codec_evidence)
    validate_spec(spec, now=datetime.now(timezone.utc).timestamp())
    caps = check_plan([spec], spent_seconds=args.spent_seconds, spent_inventory_seconds=args.spent_inventory_seconds, spent_by_kind=args.spent_by_kind)
    if Path.cwd() in args.output.resolve().parents:
        raise SystemExit("specs live outside the code tree")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"spec": str(args.output), **caps}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
