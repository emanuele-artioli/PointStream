"""Identity, hashing, and provenance helpers for the diagnostic matrix.

The matrix JSON has to prove what ran: code revision, source-frame hashes,
resolved config, checkpoint SHA, byte parts, timings, and whether a generator
result is actually a generator result. Reuse of a prior corner requires this
whole identity, not just video/scene/frame-count/generator name.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

REQUIRED_REPORT_KEYS: tuple[str, ...] = (
    "doc_role",
    "video",
    "scene",
    "frames",
    "identity",
    "source_manifest",
    "resolved_configuration",
    "generator_backend",
    "checkpoint_sha256",
    "seed",
    "device",
    "inference_parameters",
    "model_invocation_count",
    "controls",
    "generation_effect",
    "generator_comparison_valid",
    "matrix",
    "timestamp_unix",
)

REQUIRED_CORNER_KEYS: tuple[str, ...] = (
    "corner",
    "generation_on",
    "residual_on",
    "delivered_frame_hashes",
    "base_frame_hashes",
    "parts",
    "byte_subledger",
    "wire_reconciliation",
    "timing",
    "failure",
    "model_invocation_count",
)

REQUIRED_IDENTITY_KEYS: tuple[str, ...] = (
    "code_revision",
    "checkpoint_sha256",
    "source_frame_hashes",
    "config",
)

SIZE_PART_KEYS: tuple[str, ...] = (
    "residual",
    "panorama",
    "actor_reference",
    "metadata",
    "transport_total",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_rgb_frame(frame: np.ndarray) -> str:
    """SHA-256 of one source/delivered RGB frame (contiguous uint8 HWC)."""
    arr = np.ascontiguousarray(frame, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB uint8 frame, got shape {arr.shape}")
    return sha256_bytes(arr.tobytes())


def per_frame_sha256(frames: np.ndarray) -> list[str]:
    stack = np.ascontiguousarray(frames, dtype=np.uint8)
    if stack.ndim != 4 or stack.shape[-1] != 3:
        raise ValueError(f"expected THWC RGB uint8 stack, got shape {stack.shape}")
    return [sha256_rgb_frame(stack[index]) for index in range(int(stack.shape[0]))]


def sha256_path(path: Path) -> str | None:
    """Immutable SHA-256 of a checkpoint file or directory. None if missing."""
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    return None


def git_revision(repo: Path) -> dict[str, Any]:
    """`git rev-parse HEAD` plus a dirty flag. Missing git is missing identity."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"commit": commit, "dirty": bool(porcelain.strip())}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": True, "error": str(exc)}


def jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        try:
            return jsonable(value.value)
        except Exception:
            return str(value)
    return str(value)


def serialize_config(config: Any) -> dict[str, Any]:
    return jsonable(config) if config is not None else {}


def resolved_configuration(config: Any) -> dict[str, Any]:
    """Lattice stages, residual, generator backend, appearance, and seed."""
    lattice = getattr(config, "lattice", None)
    stages = getattr(config, "stages", None)
    enabled: list[str] = []
    if stages is not None and hasattr(stages, "enabled"):
        enabled = sorted(str(name) for name in stages.enabled)
    generator = getattr(config, "generator", None)
    residual = getattr(config, "residual", None)
    appearance = getattr(config, "appearance", None)
    run_cfg = getattr(config, "run", None)
    return {
        "lattice_stages": enabled,
        "lattice": jsonable(lattice),
        "residual": jsonable(residual),
        "generator_backend": getattr(generator, "backend", None),
        "generator": jsonable(generator),
        "appearance": jsonable(appearance),
        "seed": getattr(run_cfg, "seed", None),
    }


def inference_parameters(config: Any) -> dict[str, Any]:
    generator = getattr(config, "generator", None)
    if generator is None:
        return {}
    return {
        "steps": getattr(generator, "steps", None),
        "strength": getattr(generator, "strength", None),
        "guidance": getattr(generator, "guidance", None),
        "width": getattr(generator, "width", None),
        "height": getattr(generator, "height", None),
        "variant": getattr(generator, "variant", None),
        "checkpoint": getattr(generator, "checkpoint", None),
    }


def source_manifest(
    *,
    video: str,
    scene: str,
    frames: np.ndarray,
    context_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stack = np.ascontiguousarray(frames, dtype=np.uint8)
    hashes = per_frame_sha256(stack)
    record = {
        "video": video,
        "scene": scene,
        "context_id": context_id,
        "n_frames": int(stack.shape[0]),
        "shape": [int(dim) for dim in stack.shape],
        "frame_hashes": hashes,
        "stack_sha256": sha256_bytes(stack.tobytes()),
    }
    if extra:
        record.update(extra)
    return record


def build_run_identity(
    *,
    code_revision: dict[str, Any],
    checkpoint_sha256: str | None,
    source_frame_hashes: list[str],
    config: dict[str, Any],
    video: str,
    scene: str,
    frames: int,
    generator: str,
    residual_qp: int,
) -> dict[str, Any]:
    return {
        "code_revision": code_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "source_frame_hashes": list(source_frame_hashes),
        "config": config,
        "video": video,
        "scene": scene,
        "frames": int(frames),
        "generator": generator,
        "residual_qp": int(residual_qp),
    }


def identity_matches(current: dict[str, Any], prior: dict[str, Any] | None) -> bool:
    """Reuse requires complete identity, including checkpoint SHA and revision."""
    if not prior:
        return False
    for key in REQUIRED_IDENTITY_KEYS:
        if key not in prior or key not in current:
            return False
        if current[key] != prior[key]:
            return False
    for key in ("video", "scene", "frames", "generator", "residual_qp"):
        if key in current and current.get(key) != prior.get(key):
            return False
    return True


def extract_size_parts(sizes: Any) -> dict[str, int]:
    parts: dict[str, int] = {}
    as_dict: dict[str, Any] = {}
    if hasattr(sizes, "as_dict"):
        try:
            loaded = sizes.as_dict()
            if isinstance(loaded, dict):
                as_dict = loaded
        except Exception:
            as_dict = {}
    for key in SIZE_PART_KEYS:
        if hasattr(sizes, key):
            parts[key] = int(getattr(sizes, key) or 0)
        elif key in as_dict:
            parts[key] = int(as_dict[key] or 0)
        else:
            parts[key] = 0
    return parts


def extract_byte_subledger(sizes: Any) -> Any:
    """Worker B subledger if present; otherwise None (caller records current parts)."""
    subledger = getattr(sizes, "subledger", None)
    if subledger is not None:
        if hasattr(subledger, "as_dict"):
            return subledger.as_dict()
        return jsonable(subledger)
    if hasattr(sizes, "as_dict"):
        try:
            loaded = sizes.as_dict()
        except Exception:
            loaded = None
        if isinstance(loaded, dict):
            if "metadata_subledger" in loaded:
                return loaded["metadata_subledger"]
            if "subledger" in loaded:
                return loaded["subledger"]
    return None


def wire_reconciliation(result: Any, sizes: Any | None = None) -> dict[str, Any]:
    """`len(wire_request) == sizes.transport_total` when the bag carries the wire."""
    ledger = sizes if sizes is not None else getattr(result, "sizes", None)
    transport_total = int(getattr(ledger, "transport_total", 0) or 0)
    chunks = getattr(result, "chunks", ()) or ()
    wire_lengths: list[int] = []
    present = False
    for chunk in chunks:
        bag = getattr(chunk, "bag", None) or {}
        wire = bag.get("wire_request") if isinstance(bag, dict) else None
        if wire is None:
            continue
        present = True
        wire_lengths.append(len(wire))
    if not present:
        return {
            "wire_request_present": False,
            "transport_total": transport_total,
            "wire_bytes": None,
            "matched": None,
            "verdict": "wire_request_absent",
        }
    wire_bytes = int(sum(wire_lengths))
    matched = wire_bytes == transport_total
    return {
        "wire_request_present": True,
        "transport_total": transport_total,
        "wire_bytes": wire_bytes,
        "matched": matched,
        "verdict": "matched" if matched else "mismatch",
    }


def actual_timing(result: Any, wall_seconds: float | None = None) -> dict[str, Any]:
    """Keep encoder/client/evaluation timings. Do not substitute wall time for them."""
    timing = dict(getattr(result, "timing", None) or {})
    encoder = getattr(result, "encoder_seconds", None)
    client = getattr(result, "client_seconds", None)
    evaluation = getattr(result, "evaluation_seconds", None)
    if encoder is None:
        encoder = timing.get("encoder_seconds")
    if client is None:
        client = timing.get("client_seconds")
    if evaluation is None:
        evaluation = timing.get("evaluation_seconds")
    record = {
        "encoder_seconds": encoder,
        "client_seconds": client,
        "evaluation_seconds": evaluation,
    }
    if wall_seconds is not None:
        record["wall_seconds"] = round(float(wall_seconds), 2)
    return record


def failure_record(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def wrap_generator_with_counter(ref: Any) -> tuple[Any, list[int]]:
    """Count `generate` / `generate_sequence` calls without changing dispatch."""
    counter = [0]
    if ref is None:
        return None, counter
    inner = getattr(ref, "backend", ref)

    class CountingBackend:
        def generate(self, *args: Any, **kwargs: Any) -> Any:
            counter[0] += 1
            return inner.generate(*args, **kwargs)

        def generate_sequence(self, *args: Any, **kwargs: Any) -> Any:
            counter[0] += 1
            return inner.generate_sequence(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(inner, name)

    counted = CountingBackend()
    if is_dataclass(ref) and not isinstance(ref, type) and hasattr(ref, "backend"):
        return replace(ref, backend=counted), counter
    if hasattr(ref, "backend"):
        try:
            object.__setattr__(ref, "backend", counted)
            return ref, counter
        except Exception:
            try:
                ref.backend = counted
                return ref, counter
            except Exception:
                return ref, counter
    return counted, counter


def assess_generator_comparison(
    *,
    checkpoint_sha256: str | None,
    generator_backend: str | None,
    matrix: list[dict[str, Any]],
) -> dict[str, Any]:
    """A generator result is only claimable with identity and a non-no-op model."""
    reasons: list[str] = []
    sha_ok = isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64
    if not sha_ok:
        reasons.append("missing checkpoint SHA-256")
    backend = (generator_backend or "").strip().lower()
    if backend in ("", "none", "none (pasted_reference_control)"):
        reasons.append("no generator backend")

    paste_hashes: list[str] | None = None
    gen_hashes: list[str] | None = None
    gen_invocations = 0
    gen_failures = 0
    for row in matrix:
        if row.get("generation_on"):
            gen_invocations += int(row.get("model_invocation_count") or 0)
        if row.get("failure"):
            if row.get("generation_on"):
                gen_failures += 1
            continue
        if row.get("shuffled_conditioning"):
            continue
        if not row.get("generation_on"):
            if paste_hashes is None:
                paste_hashes = list(row.get("delivered_frame_hashes") or [])
            continue
        if gen_hashes is None:
            gen_hashes = list(row.get("delivered_frame_hashes") or [])

    pixels_changed = (
        paste_hashes is not None
        and gen_hashes is not None
        and paste_hashes != gen_hashes
        and bool(paste_hashes)
        and bool(gen_hashes)
    )
    if gen_failures:
        reasons.append("generation corner failed")
    if gen_invocations <= 0:
        reasons.append("no-op generator (invocation count is 0)")
    if paste_hashes is not None and gen_hashes is not None and not pixels_changed:
        reasons.append("generation did not change delivered pixels versus paste")

    valid = not reasons
    return {
        "generator_comparison_valid": valid,
        "reasons": reasons,
        "delivered_pixels_changed": pixels_changed,
        "model_invocation_count": gen_invocations,
    }


def generation_effect(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare gen-on vs paste hashes, and residual bytes when residual is on."""
    paste_off = _corner(matrix, generation_on=False, residual_on=False)
    paste_on = _corner(matrix, generation_on=False, residual_on=True)
    gen_off = _corner(matrix, generation_on=True, residual_on=False)
    gen_on = _corner(matrix, generation_on=True, residual_on=True)
    shuffled = next((row for row in matrix if row.get("shuffled_conditioning")), None)

    paste_hashes = (paste_off or paste_on or {}).get("delivered_frame_hashes")
    gen_hashes = (gen_off or gen_on or {}).get("delivered_frame_hashes")
    pixels_changed = (
        isinstance(paste_hashes, list)
        and isinstance(gen_hashes, list)
        and paste_hashes != gen_hashes
    )

    res_gen = _residual_bytes(gen_on)
    res_paste = _residual_bytes(paste_on)
    residual_changed: bool | None
    if res_gen is None or res_paste is None:
        residual_changed = None
    else:
        residual_changed = res_gen != res_paste

    shuffled_differs = None
    if shuffled and gen_hashes is not None:
        shuffled_hashes = shuffled.get("delivered_frame_hashes")
        if isinstance(shuffled_hashes, list):
            shuffled_differs = shuffled_hashes != gen_hashes

    return {
        "delivered_pixels_changed": pixels_changed,
        "gen_on_vs_paste_hash_match": (
            isinstance(paste_hashes, list)
            and isinstance(gen_hashes, list)
            and paste_hashes == gen_hashes
        ),
        "residual_bytes_gen_on": res_gen,
        "residual_bytes_gen_off": res_paste,
        "residual_demand_changed": residual_changed,
        "shuffled_conditioning_changed_pixels": shuffled_differs,
    }


def reusable_corners(
    prior: dict[str, Any] | None,
    current_identity: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not prior:
        return {}
    prior_identity = prior.get("identity")
    if not identity_matches(current_identity, prior_identity):
        return {}
    found: dict[str, dict[str, Any]] = {}
    for row in prior.get("matrix") or []:
        name = row.get("corner")
        if name:
            found[str(name)] = row
    return found


def _corner(
    matrix: list[dict[str, Any]],
    *,
    generation_on: bool,
    residual_on: bool,
) -> dict[str, Any] | None:
    for row in matrix:
        if row.get("shuffled_conditioning"):
            continue
        if bool(row.get("generation_on")) == generation_on and bool(
            row.get("residual_on")
        ) == residual_on:
            if row.get("failure"):
                continue
            return row
    return None


def _residual_bytes(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    parts = row.get("parts") or {}
    if "residual" in parts:
        return int(parts["residual"])
    return None


def fingerprint_identity(identity: dict[str, Any]) -> str:
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(canonical.encode("utf-8"))


__all__ = [
    "REQUIRED_CORNER_KEYS",
    "REQUIRED_IDENTITY_KEYS",
    "REQUIRED_REPORT_KEYS",
    "SIZE_PART_KEYS",
    "actual_timing",
    "assess_generator_comparison",
    "build_run_identity",
    "extract_byte_subledger",
    "extract_size_parts",
    "failure_record",
    "fingerprint_identity",
    "generation_effect",
    "git_revision",
    "identity_matches",
    "inference_parameters",
    "jsonable",
    "per_frame_sha256",
    "resolved_configuration",
    "reusable_corners",
    "serialize_config",
    "sha256_bytes",
    "sha256_path",
    "sha256_rgb_frame",
    "source_manifest",
    "wire_reconciliation",
    "wrap_generator_with_counter",
]
