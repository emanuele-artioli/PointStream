"""Paired player context development renderer; not a full codec or isolation claim.

A hash-pinned config selects the same 24 native-frame inputs for each profile.
The source and conditioning provenance are declarations, not an independent
source decoder or a complete transmitted packet. This worker grants no compute authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import resource
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from experiments.tennis.k6b_generate import CFG, CONTEXT_FRAMES, CONTEXT_OVERLAP, GEN_H, GEN_W, Renderer, UNUSED_REFERENCE_HEAD, sha256
from experiments.tennis.w8_receiver_generate import BASE_SHA, PINNED_WEIGHTS

PROFILES = ("original", "tennis_only", "match_adapted")
SETTINGS = {"width": GEN_W, "height": GEN_H, "steps": 30, "cfg": CFG,
            "context_frames": CONTEXT_FRAMES, "context_overlap": CONTEXT_OVERLAP}
SHA = re.compile(r"^[0-9a-f]{64}$")
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_sha(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{a.dtype.str}:{a.shape}:".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def pinned_file(item: Any, label: str, expected: str | None = None) -> Path:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not SHA.fullmatch(str(item.get("sha256", ""))):
        raise ValueError(f"{label}: path and SHA256 required")
    if expected is not None and item["sha256"] != expected:
        raise ValueError(f"{label}: wrong pinned identity")
    path = Path(item["path"])
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise ValueError(f"{label}: file missing or hash mismatch")
    return path


def resolve_staged(config: dict[str, Any], bindings: list[list[str]]) -> dict[str, Any]:
    """Resolve only explicitly bound roots, rejecting traversal and symlink escapes."""
    roots: dict[str, Path] = {}
    for name, raw_root in bindings:
        if not SAFE.fullmatch(name) or name in roots:
            raise ValueError("invalid or duplicate staged binding")
        roots[name] = Path(raw_root).resolve(strict=True)
    result = copy.deepcopy(config)
    items = [*result.get("units", []), *result.get("weights", {}).values(),
             result.get("tennis_checkpoint", {})]
    if "match_checkpoint" in result:
        items.append(result["match_checkpoint"])
    used: set[str] = set()
    for item in items:
        value = item.get("path")
        if not isinstance(value, str):
            raise ValueError("file path required")
        if not value.startswith("staged:"):
            if not Path(value).is_absolute():
                raise ValueError("direct paths must be absolute")
            continue
        logical = value.removeprefix("staged:")
        parts = logical.split("/")
        if not parts or not SAFE.fullmatch(parts[0]) or any(not x or x in (".", "..") for x in parts):
            raise ValueError("invalid staged path")
        name = parts[0]
        if name not in roots:
            raise ValueError(f"missing staged binding: {name}")
        root = roots[name]
        if len(parts) > 1:
            if not root.is_dir():
                raise ValueError("staged member root must be a directory")
            target = (root / PurePosixPath(*parts[1:])).resolve(strict=True)
            if not target.is_relative_to(root):
                raise ValueError("staged member escapes root")
        else:
            target = root
        item["path"] = str(target)
        used.add(name)
    if used != set(roots):
        raise ValueError("extra staged binding")
    return result


def load_config(path: Path, expected_sha: str, bindings: list[list[str]] | None = None) -> dict[str, Any]:
    if not SHA.fullmatch(expected_sha) or not path.is_file() or sha256(path) != expected_sha:
        raise ValueError("config hash mismatch")
    config = resolve_staged(json.loads(path.read_text()), bindings or [])
    if (config.get("schema"), config.get("scope")) not in ((1, "paired-player-context-development"), (2, "player-motion-development")):
        raise ValueError("invalid config schema or scope")
    if config["schema"] == 2:
        motion_policy(config)
    if config["schema"] == 2 and (config.get("native_frames") != {"smoke": 44, "full": 240}
                                   or "match_checkpoint" in config):
        raise ValueError("invalid motion lengths or unexpected match checkpoint")
    if config.get("settings") != SETTINGS or not math.isfinite(float(config["settings"]["cfg"])):
        raise ValueError("invalid rendering settings")
    source = config.get("source", {})
    if (not all(isinstance(source.get(k), str) and source[k].strip() for k in ("group", "window", "provenance"))
            or not SHA.fullmatch(str(source.get("source_video_sha256", "")))
            or source.get("exposure") != "opened-development"
            or not isinstance(source.get("native_fps"), list) or len(source["native_fps"]) != 2
            or any(type(n) is not int or n <= 0 for n in source["native_fps"])
            or any(type(source.get(k)) is not int or source[k] <= 0 for k in ("width", "height"))):
        raise ValueError("invalid source provenance, fps, or geometry")
    conditioning = config.get("conditioning", {})
    if (not isinstance(conditioning.get("provenance"), str) or not conditioning["provenance"].strip()
            or conditioning.get("not_complete_packet") is not True
            or conditioning.get("not_independent_source_decoder") is not True):
        raise ValueError("conditioning provenance and boundaries required")
    units = config.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("ordered units required")
    if config["schema"] == 2 and len(units) != 1:
        raise ValueError("motion development requires exactly one configured unit")
    names = [u.get("unit") for u in units]
    if len(set(names)) != len(names) or any(not isinstance(n, str) or not SAFE.fullmatch(n) for n in names):
        raise ValueError("unit names must be unique and file-safe")
    if set(config.get("weights", {})) != set(PINNED_WEIGHTS):
        raise ValueError("all eight original weights required")
    for name, digest in PINNED_WEIGHTS.items():
        pinned_file(config["weights"][name], name, digest)
    pinned_file(config.get("tennis_checkpoint"), "tennis checkpoint", BASE_SHA)
    if config["schema"] == 1:
        match = config.get("match_checkpoint", {})
        pinned_file(match, "match checkpoint")
        if match.get("base_sha256") != BASE_SHA or match.get("match") != source["group"] or match.get("bits") != 4:
            raise ValueError("match checkpoint base lineage and match required")
    return config


def context(unit: dict[str, Any], native_frames: int = CONTEXT_FRAMES) -> dict[str, Any]:
    path = pinned_file(unit, f"unit {unit['unit']}")
    if type(unit.get("run")) is not int or type(unit.get("start_t")) is not int or unit["run"] < 0 or unit["start_t"] < 0:
        raise ValueError("integer run and start_t required")
    with np.load(path, allow_pickle=False) as f:
        if not {"reference", "pose", "t", "run", "box"} <= set(f.files):
            raise ValueError("unit is missing conditioning arrays")
        data = {k: f[k] for k in ("reference", "pose", "t", "run", "box")}
    t, runs, boxes = data["t"], data["run"], data["box"]
    n = len(t)
    if (t.ndim != 1 or t.dtype.kind not in "iu" or runs.shape != (n,) or runs.dtype.kind not in "iu"
            or n < native_frames or np.any(np.diff(t) <= 0) or np.any(runs < 0)
            or boxes.shape != (n, 4) or boxes.dtype.kind not in "fiu" or not np.isfinite(boxes).all()
            or np.any(boxes[:, 2:] <= 0)
            or data["reference"].shape != (GEN_H, GEN_W, 3) or data["reference"].dtype != np.uint8
            or data["pose"].shape != (n, GEN_H, GEN_W, 3) or data["pose"].dtype != np.uint8):
        raise ValueError("invalid unit array shape, dtype, ticks, or boxes")
    rows = np.flatnonzero((t >= unit["start_t"]) & (t < unit["start_t"] + native_frames))
    if (len(rows) != native_frames or not np.array_equal(t[rows], np.arange(unit["start_t"], unit["start_t"] + native_frames))
            or not np.all(runs[rows] == unit["run"])):
        raise ValueError("selected context must be consecutive native ticks in one run")
    return {"reference": data["reference"], "pose": data["pose"][rows], "t": t[rows], "box": boxes[rows]}


def make_plan(config: dict[str, Any], config_sha: str, max_units: int, native_frames: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if type(max_units) is not int or max_units < 1 or max_units > len(config["units"]):
        raise ValueError("max-units must select an available ordered prefix")
    frames = selected_frames(config, native_frames)
    all_arrays = [context(unit, frames) for unit in config["units"]]
    arrays = all_arrays[:max_units]
    units = []
    for unit, data in zip(config["units"], arrays):
        seed = int(hashlib.sha256(unit["unit"].encode()).hexdigest()[:7], 16) * 1009 + unit["run"]
        entry = {"unit": unit["unit"], "npz_sha256": unit["sha256"], "run": unit["run"], "seed": seed,
                 "t": data["t"].tolist(), "reference_sha256": array_sha(data["reference"]),
                 "pose_sha256": array_sha(data["pose"]), "box_sha256": array_sha(data["box"])}
        if config.get("schema") == 2:
            label = unit.get("review_label")
            if not isinstance(label, str) or not label.strip() or len(label) > 80 or not label.isprintable():
                raise ValueError("invalid motion review label")
            entry["review_label"] = label
        units.append(entry)
    plan = {"config_sha256": config_sha, "scope": config["scope"], "source": config["source"],
            "conditioning": config["conditioning"], "settings": SETTINGS, "units": units}
    if config.get("schema") == 2:
        plan["native_frames"] = frames
    return plan, arrays


def complete_crops(crops: np.ndarray, native_frames: int = 24) -> bool:
    """Reject incomplete or flat crops; repeated static frames remain valid."""
    return (crops.shape == (native_frames, GEN_H, GEN_W, 3) and crops.dtype == np.uint8
            and all(float(c.std()) > 5 for c in crops))


def profile_inputs(config: dict[str, Any], profile: str) -> tuple[Path | None, Path | None]:
    if profile == "original":
        return None, None
    if profile == "tennis_only":
        return Path(config["tennis_checkpoint"]["path"]), None
    if profile == "match_adapted":
        return Path(config["match_checkpoint"]["path"]), Path(config["tennis_checkpoint"]["path"])
    raise ValueError("unknown profile")


def motion_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve a frozen admission policy; selection does not authorize compute."""
    name = config.get("motion_admission_policy", "original-1500-v1")
    policies = {
        "original-1500-v1": {"smoke": 450, "full": 1500, "job": 2100, "validator": 60},
        "extended-1800-v1": {"smoke": 450, "full": 1800, "job": 2400, "validator": 60},
        "extended-2500-v1": {"smoke": 600, "full": 2500, "job": 3200, "validator": 60},
    }
    if not isinstance(name, str) or name not in policies:
        raise ValueError("unknown motion admission policy")
    return {"name": name, **policies[name]}


def selected_frames(config: dict[str, Any], native_frames: int | None, *, required: bool = False) -> int:
    if config.get("schema") == 1:
        if native_frames is not None:
            raise ValueError("schema 1 forbids --native-frames")
        return 24
    if config.get("schema") != 2 or config.get("scope") != "player-motion-development" or config.get("native_frames") != {"smoke": 44, "full": 240}:
        raise ValueError("invalid motion schema or lengths")
    motion_policy(config)
    if native_frames is None:
        if required:
            raise ValueError("schema 2 requires --native-frames")
        return 240
    if type(native_frames) is not int or native_frames not in config["native_frames"].values():
        raise ValueError("native frames must be a declared length")
    return native_frames


def require_profile(config: dict[str, Any], profile: str) -> None:
    if config.get("schema") == 2 and profile != "tennis_only":
        raise ValueError("motion development permits only tennis_only")


def run(args: argparse.Namespace) -> None:
    worker_start, cpu_start = time.monotonic(), time.process_time()
    config = load_config(args.config, args.config_sha256, getattr(args, "staged", None))
    frames = selected_frames(config, getattr(args, "native_frames", None), required=True)
    require_profile(config, args.profile)
    plan, arrays = make_plan(config, args.config_sha256, args.max_units, frames if config["schema"] == 2 else None)
    fit, base = profile_inputs(config, args.profile)
    exclusive_limit = "18000" if os.environ.get("PS_CLAIMED_GPU_UUID") and "PS_GPU_SHARING" not in os.environ else "0"
    memory = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", exclusive_limit))
    if memory <= 0:
        raise ValueError("positive PS_GPU_MEMORY_LIMIT_MIB required")
    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    output = scratch / "publish"
    # Never resume or overwrite a previous partial run.
    if (stage / "published.tar").exists() or (stage / "partial.tar").exists():
        raise FileExistsError("stale stage archive")
    output.mkdir(parents=False, exist_ok=False)
    model_scratch = scratch / "player-requalification-model"
    model_scratch.mkdir(parents=False, exist_ok=False)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1"})
    import torch

    from experiments.audit.env_smoke import model_on_cuda, profile_cuda
    from experiments.jobs.monitor import publish_progress

    torch.set_num_threads(1)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, memory * 2**20 / properties.total_memory))
    start = time.monotonic()
    weights = {key: Path(value["path"]) for key, value in config["weights"].items()}
    renderer = Renderer(weights, "original", "cuda", model_scratch, fit=fit, base=base,
                        bits=4 if args.profile == "match_adapted" else 0, cfg=CFG)
    load_seconds = time.monotonic() - start
    check_model_record(renderer.record, args.profile, fit)
    on_cuda = {name: model_on_cuda(model) for name, model in renderer.modules.items()}
    if not all(on_cuda.values()):
        raise RuntimeError("a renderer module is not on CUDA")
    plan_sha = digest_json(plan)
    records = []
    for entry, data in zip(plan["units"], arrays):
        started = time.monotonic()
        crops, kernels = profile_cuda(lambda: renderer.render(data["reference"], data["pose"], 30, entry["seed"]))
        elapsed = time.monotonic() - started
        if not complete_crops(crops, frames):
            raise ValueError("renderer output incomplete, invalid or flat")
        path = output / f"{entry['unit']}@{args.profile}.npz"
        temporary = output / f".{path.name}.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, t=data["t"], box=data["box"], crops=crops)
        temporary.replace(path)
        sidecar = {"plan_sha256": plan_sha, "profile": args.profile, "unit": entry["unit"],
                   "reference_sha256": entry["reference_sha256"], "pose_sha256": entry["pose_sha256"],
                   "output_sha256": sha256(path), "profiled_render_seconds": elapsed,
                   "distinct_frame_count": len({array_sha(c) for c in crops}), "kernels": kernels}
        path.with_suffix(".json").write_text(json.dumps(sidecar, sort_keys=True) + "\n")
        records.append(sidecar)
        publish_progress(os.environ.get("PS_STAGE", "unknown"), len(records))
    receipt = {"schema": 1, "scope": "24-native-frame-context-only", "plan": plan, "plan_sha256": plan_sha,
               "profile": args.profile, "fit_sha256": sha256(fit) if fit else None,
               "base_sha256": sha256(base) if base else None, "base_role": "installed common base; match state content-specific",
               "match_lineage": "config declaration; checkpoint semantic provenance not independently verified",
               "peft_unexpected_keys": "not recorded by apply_fit; unverified",
               "device": {"name": properties.name, "capability": [properties.major, properties.minor],
                          "torch": str(torch.__version__), "cuda": str(torch.version.cuda)},
               "on_cuda": on_cuda, "model_record": renderer.record, "gpu_memory_limit_mib": memory,
               "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20, "load_seconds": load_seconds, "outputs": records}
    if config["schema"] == 2:
        receipt.update({"schema": 2, "scope": "player-motion-development", "native_frames": frames,
                        "base_role": "installed common tennis checkpoint", "match_lineage": None})
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, default=str) + "\n")
    validate_output(output, config, args.config_sha256, args.profile, args.max_units, getattr(args, "native_frames", None))
    temporary_archive = stage / ".published.tar.tmp"
    with tarfile.open(temporary_archive, "w") as archive:
        archive.add(output, arcname="publish")
    temporary_archive.replace(stage / "published.tar")
    archive_digest = sha256(stage / "published.tar")
    timing = {"scope": "run entry through archive hash; excludes interpreter startup and timing receipt write",
              "worker_wall_seconds": time.monotonic() - worker_start, "worker_cpu_seconds": time.process_time() - cpu_start,
              "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "archive_sha256": archive_digest}
    (stage / "worker_timing.json").write_text(json.dumps(timing, sort_keys=True) + "\n")


def compare_receipts(receipts: list[dict[str, Any]]) -> None:
    """Require all three profiles to refer to the very same complete context plan."""
    if len(receipts) != 3 or {r.get("profile") for r in receipts} != set(PROFILES):
        raise ValueError("exactly one receipt for each profile required")
    plan = receipts[0].get("plan")
    if not isinstance(plan, dict) or any(r.get("plan") != plan or r.get("plan_sha256") != digest_json(plan) for r in receipts):
        raise ValueError("paired profiles have unequal conditioning or plan")


def check_model_record(model: dict[str, Any], profile: str, fit: Path | None) -> None:
    if model.get("profile") != "original" or model.get("cfg") != CFG:
        raise ValueError("model profile or guidance mismatch")
    for key in ("vae", "image_encoder", "reference_unet", "denoising_unet", "pose_guider"):
        record = model.get(key, {})
        allowed = set(UNUSED_REFERENCE_HEAD) if key == "reference_unet" else set()
        if (record.get("missing") != 0 or record.get("unexpected") != 0 or record.get("mismatched", 0) != 0
                or not set(record.get("missing_allowed", [])) <= allowed):
            raise ValueError(f"unqualified model loading keys: {key}")
    if model.get("motion_module_keys", 0) <= 100:
        raise ValueError("motion module is not substantively loaded")
    loaded_fit = model.get("fit")
    if profile == "original" and loaded_fit is not None:
        raise ValueError("original unexpectedly loaded a fit")
    if profile != "original":
        if fit is None or not isinstance(loaded_fit, dict) or loaded_fit.get("checkpoint_sha256") != sha256(fit):
            raise ValueError("loaded fit identity mismatch")
    if isinstance(loaded_fit, dict) and isinstance(loaded_fit.get("pose_guider"), dict):
        guider = loaded_fit["pose_guider"]
        if guider.get("missing") != 0 or guider.get("unexpected") != 0:
            raise ValueError("unqualified fitted pose guider keys")
    if profile == "match_adapted" and (not isinstance(loaded_fit, dict) or loaded_fit.get("base", {}).get("sha256") != BASE_SHA
                                       or loaded_fit.get("billing", {}).get("bits") != 4
                                       or loaded_fit.get("billing", {}).get("bytes", 0) <= 0):
        raise ValueError("loaded base or quantized match identity mismatch")
    if profile == "tennis_only" and (not isinstance(loaded_fit, dict) or loaded_fit.get("base") is not None):
        raise ValueError("tennis profile unexpectedly loaded a base twice")


def validate_output(output: Path, config: dict[str, Any], config_sha: str, profile: str, max_units: int, native_frames: int | None = None) -> dict[str, Any]:
    frames = selected_frames(config, native_frames, required=True)
    require_profile(config, profile)
    plan, arrays = make_plan(config, config_sha, max_units, native_frames)
    receipt = json.loads((output / "receipt.json").read_text())
    fit, base = profile_inputs(config, profile)
    if (receipt.get("schema") != config["schema"] or receipt.get("scope") != ("24-native-frame-context-only" if config["schema"] == 1 else "player-motion-development")
            or (config["schema"] == 2 and receipt.get("native_frames") != frames)
            or receipt.get("profile") != profile or receipt.get("plan") != plan or receipt.get("plan_sha256") != digest_json(plan)
            or receipt.get("fit_sha256") != (sha256(fit) if fit else None)
            or receipt.get("base_sha256") != (sha256(base) if base else None)):
        raise ValueError("receipt identity, settings or profile mismatch")
    device = receipt.get("device", {})
    on_cuda = receipt.get("on_cuda", {})
    memory, peak = receipt.get("gpu_memory_limit_mib"), receipt.get("gpu_peak_mib")
    if (not device.get("name") or not device.get("cuda") or len(device.get("capability", [])) != 2
            or set(on_cuda) != {"vae", "image_encoder", "reference_unet", "denoising_unet", "pose_guider"}
            or not all(v is True for v in on_cuda.values()) or not isinstance(memory, (int, float)) or memory <= 0
            or not isinstance(peak, (int, float)) or not math.isfinite(peak) or not 0 < peak <= memory):
        raise ValueError("invalid CUDA or memory receipt")
    check_model_record(receipt.get("model_record", {}), profile, fit)
    records = receipt.get("outputs")
    if not isinstance(records, list) or len(records) != len(plan["units"]):
        raise ValueError("missing output records")
    expected_names = {"receipt.json"}
    for entry, data, record in zip(plan["units"], arrays, records):
        path = output / f"{entry['unit']}@{profile}.npz"
        sidepath = path.with_suffix(".json")
        expected_names.update((path.name, sidepath.name))
        sidecar = json.loads(sidepath.read_text())
        if (record != sidecar or sidecar.get("unit") != entry["unit"] or sidecar.get("profile") != profile
                or sidecar.get("plan_sha256") != digest_json(plan) or sidecar.get("reference_sha256") != entry["reference_sha256"]
                or sidecar.get("pose_sha256") != entry["pose_sha256"] or sidecar.get("output_sha256") != sha256(path)
                or sidecar.get("kernels", {}).get("kernel_launches", 0) <= 0
                or not isinstance(sidecar.get("profiled_render_seconds"), (int, float))
                or not math.isfinite(sidecar["profiled_render_seconds"]) or sidecar["profiled_render_seconds"] <= 0):
            raise ValueError("output sidecar identity or kernel evidence mismatch")
        with np.load(path, allow_pickle=False) as result:
            if (set(result.files) != {"t", "box", "crops"} or not np.array_equal(result["t"], data["t"])
                    or not np.array_equal(result["box"], data["box"]) or result["crops"].shape != (frames, GEN_H, GEN_W, 3)
                    or not complete_crops(result["crops"], frames)
                    or sidecar.get("distinct_frame_count") != len({array_sha(c) for c in result["crops"]})):
                raise ValueError("output ticks, boxes, crops or coverage mismatch")
    if {p.name for p in output.iterdir()} != expected_names:
        raise ValueError("unexpected or stale output files")
    return receipt


def smoke_cost_gate(config: dict[str, Any], receipt: dict[str, Any], timing: dict[str, Any],
                    dispatch: dict[str, Any], now: float) -> dict[str, Any]:
    """Conservative admission heuristic; hard fleet timeouts remain authoritative."""
    if config.get("schema") != 2 or len(config.get("units", [])) != 1 or receipt.get("native_frames") != 44:
        raise ValueError("cost gate requires one-unit schema 2 smoke")
    def positive(value: Any) -> float:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("missing or invalid cost timing or campaign clock")
        return float(value)

    policy = motion_policy(config)
    load = positive(receipt.get("load_seconds"))
    wall = positive(timing.get("worker_wall_seconds"))
    clock = dispatch.get("campaign_clock")
    end = positive(clock.get("budget_end") if isinstance(clock, dict) else None)
    now = positive(now)
    if wall < load:
        raise ValueError("worker wall shorter than model load")
    remaining = end - now
    available = remaining - policy["validator"]
    if available <= 0:
        raise ValueError("campaign clock expired or lacks validator reserve")
    forecast = 1.25 * (load + max(0, wall - load) * (240 / 44)) + 90
    shutdown_reserve = 60 if config.get("motion_admission_policy") == "extended-2500-v1" else 0
    if not math.isfinite(forecast) or forecast + shutdown_reserve > policy["full"] or forecast + shutdown_reserve > available:
        raise ValueError("full-stage forecast exceeds cap or remaining campaign time")
    return {"forecast_seconds": forecast, "campaign_remaining_seconds": remaining,
            "available_after_validator_reserve_seconds": available, "full_stage_cap_seconds": policy["full"],
            "policy": policy["name"],
            "basis": "1.25*(load + (worker wall - load)*(240/44)) + 90 seconds; admission heuristic, not a completion guarantee"}


def reference_smoke_cost_gate(receipt: dict[str, Any], timing: dict[str, Any],
                              dispatch: dict[str, Any], now: float) -> dict[str, Any]:
    """Bound two 24-tick contexts using the real one-context worker clock."""
    def positive(value: Any) -> float:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid timing or campaign clock")
        return float(value)

    clock = dispatch.get("campaign_clock")
    load = positive(receipt.get("load_seconds"))
    wall = positive(timing.get("worker_wall_seconds"))
    current = positive(now)
    end = positive(clock.get("budget_end") if isinstance(clock, dict) else None)
    if wall < load:
        raise ValueError("worker wall shorter than load")
    remaining = end - current
    forecast = 1.25 * (load + 2 * (wall - load)) + 90
    if forecast > 450 or forecast > remaining - 60:
        raise ValueError("reference pilot forecast exceeds cap or remaining time")
    return {"forecast_seconds": forecast, "campaign_remaining_seconds": remaining,
            "full_stage_cap_seconds": 450, "validator_reserve_seconds": 60}


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        p = subs.add_parser(command)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--config-sha256", required=True)
        p.add_argument("--profile", choices=PROFILES, required=True)
        p.add_argument("--max-units", type=int, required=True)
        p.add_argument("--native-frames", type=int)
        p.add_argument("--staged", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    plan = subs.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--config-sha256", required=True)
    plan.add_argument("--max-units", type=int, required=True)
    plan.add_argument("--native-frames", type=int)
    plan.add_argument("--staged", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    subs.add_parser("validate-stage")
    return parser


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    stage_validation = args.command == "validate-stage"
    if stage_validation:
        dispatch = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "dispatch.json").read_text())
        command = dispatch.get("command")
        if not isinstance(command, list) or len(command) < 4 or command[1:4] != ["-m", "experiments.tennis.player_requalification", "run"]:
            raise ValueError("dispatch is not the player requalification run entrypoint")
        args = parser.parse_args(["validate", *command[4:]])
    if args.command == "run":
        run(args)
    elif args.command == "plan":
        config = load_config(args.config, args.config_sha256, args.staged)
        plan, _ = make_plan(config, args.config_sha256, args.max_units, args.native_frames)
        print(json.dumps(plan, sort_keys=True))
    else:
        config = load_config(args.config, args.config_sha256, getattr(args, "staged", None))
        scratch = Path(os.environ["PS_SCRATCH_DIR"])
        scratch.mkdir(parents=True, exist_ok=True)
        stage = Path(os.environ["PS_STAGE_DIR"])
        timing = json.loads((stage / "worker_timing.json").read_text())
        if (timing.get("archive_sha256") != sha256(stage / "published.tar")
                or any(not isinstance(timing.get(k), (int, float)) or not math.isfinite(timing[k]) or timing[k] <= 0
                       for k in ("worker_wall_seconds", "worker_cpu_seconds"))):
            raise ValueError("worker timing receipt or archive hash mismatch")
        with tempfile.TemporaryDirectory(prefix="player-validate-", dir=scratch) as tmp:
            with tarfile.open(stage / "published.tar") as archive:
                archive.extractall(tmp, filter="data")
            receipt = validate_output(Path(tmp) / "publish", config, args.config_sha256, args.profile, args.max_units, args.native_frames)
        report: dict[str, Any] = {"passed": True, "checks": {"complete_conditioning_and_ticks": True, "profile_lineage_and_cuda": True,
                                             "output_hash_shape_and_nonflat": True}}
        if stage_validation and config["schema"] == 2:
            stage_name = os.environ.get("PS_STAGE", "")
            expected_frames = {"smoke": 44, "full": 240}.get(stage_name)
            if expected_frames is None or args.native_frames != expected_frames:
                raise ValueError("motion stage and native frame count mismatch")
            if stage_name == "smoke":
                report["smoke_cost_gate"] = smoke_cost_gate(config, receipt, timing, dispatch, time.monotonic())
                report["checks"]["smoke_cost_admission"] = True
        if stage_validation and config.get("reference_pilot"):
            stage_name = os.environ.get("PS_STAGE", "")
            if (config["schema"] != 1 or args.profile != "tennis_only"
                    or args.max_units != {"smoke": 1, "full": 2}.get(stage_name)):
                raise ValueError("invalid reference pilot stage")
            if stage_name == "smoke":
                report["smoke_cost_gate"] = reference_smoke_cost_gate(receipt, timing, dispatch, time.monotonic())
                report["checks"]["smoke_cost_admission"] = True
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(report) + "\n")
        print("validated complete paired-context profile output")


if __name__ == "__main__":
    main()
