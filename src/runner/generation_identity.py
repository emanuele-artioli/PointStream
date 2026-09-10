"""Checkpoint and configuration identity for client generator reconstruction.

Encoder residual may reuse generated crops as ``supplied_crop``. The client
wire must not infer generation from that field. Identity on the envelope is
how a fresh client process reconstructs the same backend, or fails closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import hashlib
import json

IDENTITY_KEYS = (
    "checkpoint_id",
    "checkpoint_sha256",
    "config_identity",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def params_as_dict(params: Any | None) -> dict[str, Any]:
    """The generation knobs that travel with the client envelope."""
    out: dict[str, Any] = {}
    if params is None:
        return out
    for key in ("steps", "strength", "guidance_scale", "width", "height"):
        value = getattr(params, key, None)
        if value is not None:
            out[key] = value
    return out


def canonical_identity_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    capabilities = payload.get("capabilities") or ()
    requires = payload.get("requires") or ()
    return {
        "name": payload.get("name"),
        "seed": payload.get("seed"),
        "params": dict(payload.get("params") or {}),
        "capabilities": sorted(str(item) for item in capabilities),
        "requires": sorted(str(item) for item in requires),
        "checkpoint_id": payload.get("checkpoint_id"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
    }


def config_identity_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(canonical_identity_body(payload), sort_keys=True, default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_file(ref: Any, checkpoint: str | Path | None) -> Path | None:
    """Weight file on the live ref, walking nested ``.backend`` wrappers.

    Encoder identity gets an explicit CLI path. Client reconstruct gets the same
    object after ``as_runner_ref`` and the diagnostic call-counter wrap, neither
    of which copies ``checkpoint`` onto the outer object.
    """
    candidates: list[Any] = [checkpoint]
    seen: set[int] = set()
    node: Any = ref
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        candidates.append(getattr(node, "checkpoint", None))
        candidates.append(getattr(node, "loaded_checkpoint", None))
        node = getattr(node, "backend", None)
    for item in candidates:
        if item is None or item == "":
            continue
        path = Path(str(item))
        if path.is_file():
            return path
    return None


def identity_from_ref(
    ref: Any,
    *,
    seed: int,
    params: Mapping[str, Any] | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Identity block stored in ``generator_meta``.

    Injected test backends have no weight file; their digest is
    ``injected:<name>``. A file-backed backend (pix2pix and friends) is
    hashed from the weights on the object, not from a missing CLI path.
    """
    name = str(getattr(ref, "name", "injected"))
    checkpoint_path = _checkpoint_file(ref, checkpoint)
    if checkpoint_path is not None:
        digest = sha256_file(checkpoint_path)
        checkpoint_id = f"{checkpoint_path.name}:{digest}"
    else:
        digest = f"injected:{name}"
        checkpoint_id = digest
    meta = {
        "name": name,
        "seed": int(seed),
        "params": dict(params or {}),
        "capabilities": sorted(str(item) for item in (getattr(ref, "capabilities", ()) or ())),
        "requires": sorted(str(item) for item in (getattr(ref, "requires", ()) or ())),
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": digest,
    }
    meta["config_identity"] = config_identity_digest(meta)
    return meta


def missing_identity_fields(meta: Mapping[str, Any]) -> tuple[str, ...]:
    missing: list[str] = []
    for key in IDENTITY_KEYS:
        value = meta.get(key)
        if value is None or value == "":
            missing.append(key)
    return tuple(missing)


def _assert_checkpoint_matches(injected: Any, gen_meta: Mapping[str, Any]) -> None:
    claimed = identity_from_ref(
        injected,
        seed=int(gen_meta.get("seed") or 0),
        params=gen_meta.get("params") or {},
    )
    payload_sha = gen_meta.get("checkpoint_sha256")
    if claimed["checkpoint_sha256"] == payload_sha:
        return
    injected_name = getattr(injected, "name", None)
    if injected_name == "injected" and payload_sha == f"injected:{gen_meta.get('name')}":
        return
    raise ValueError(
        f"Mismatched checkpoint identity: requested {claimed['checkpoint_sha256']}, "
        f"payload has {payload_sha}"
    )


def _from_registry(name: str) -> Any:
    from src.components.generation import REGISTRY
    from src.contracts.conditioning import FrameGenerator
    from src.pipeline.reconstruction.dispatch import from_spec

    if not REGISTRY.has(name):
        return None
    spec = REGISTRY.spec(name)
    backend = REGISTRY.build(name)
    if isinstance(backend, FrameGenerator):
        return from_spec(spec, backend)
    return None


def resolve_client_generator(
    gen_meta: Mapping[str, Any],
    *,
    injected: Any = None,
    require_identity: bool = False,
) -> Any:
    """Resolve the generator the client will run, or raise.

    When generated placements are present, missing or unknown checkpoint
    identity fails closed. Empty envelopes that only check seed/name may pass
    ``require_identity=False`` so older tests keep working.
    """
    if require_identity:
        missing = missing_identity_fields(gen_meta)
        if missing:
            raise ValueError(f"Missing generator identity fields: {', '.join(missing)}")
        expected = config_identity_digest(gen_meta)
        if gen_meta.get("config_identity") != expected:
            raise ValueError("Mismatched generator configuration identity")

    if injected is not None:
        if require_identity or not missing_identity_fields(gen_meta):
            if missing_identity_fields(gen_meta):
                raise ValueError(
                    "Missing generator identity fields: "
                    + ", ".join(missing_identity_fields(gen_meta))
                )
            expected = config_identity_digest(gen_meta)
            if gen_meta.get("config_identity") != expected:
                raise ValueError("Mismatched generator configuration identity")
            _assert_checkpoint_matches(injected, gen_meta)
        return injected

    if not require_identity:
        name = gen_meta.get("name")
        return _from_registry(str(name)) if name else None

    name = gen_meta.get("name")
    digest = gen_meta.get("checkpoint_sha256")
    if not name:
        raise ValueError("Payload requires generation but generator name is missing")
    if isinstance(digest, str) and digest.startswith("injected:"):
        raise ValueError(
            f"Cannot reconstruct injected generator {name!r}; "
            "the client was given no matching backend"
        )
    resolved = _from_registry(str(name))
    if resolved is None:
        raise ValueError(f"Unknown generator checkpoint identity: {digest}")
    return resolved
