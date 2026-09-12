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


def _backend_checkpoint_file(ref: Any) -> Path | None:
    seen: set[int] = set()
    node: Any = ref
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        for attr in ("loaded_checkpoint", "checkpoint"):
            val = getattr(node, attr, None)
            if val is not None and val != "":
                path = Path(str(val))
                if path.is_file():
                    return path
        node = getattr(node, "backend", None)
    return None


def _backend_weight_digest(ref: Any) -> str | None:
    seen: set[int] = set()
    node: Any = ref
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        for attr in ("checkpoint_sha256", "weight_digest", "weight_hash"):
            val = getattr(node, attr, None)
            if val is not None and val != "":
                return str(val)
        node = getattr(node, "backend", None)
    return None


def _checkpoint_file(ref: Any, checkpoint: str | Path | None = None) -> Path | None:
    """Weight file on the live ref, walking nested ``.backend`` wrappers.

    The live backend's actual loaded checkpoint or checkpoint attribute takes
    precedence. An explicit checkpoint parameter is only used as fallback if
    the backend does not expose its own checkpoint.
    """
    ref_file = _backend_checkpoint_file(ref)
    if ref_file is not None:
        return ref_file
    if checkpoint is not None and checkpoint != "":
        path = Path(str(checkpoint))
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


def resolve_client_checkpoint(
    name: str,
    gen_meta: Mapping[str, Any],
    *,
    checkpoint: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_registry: Mapping[str, str | Path] | None = None,
) -> Path | None:
    """Resolve a client checkpoint path explicitly, without trusting encoder paths."""
    from src.contracts import paths

    target_sha = gen_meta.get("checkpoint_sha256")
    target_id = gen_meta.get("checkpoint_id")

    if checkpoint_registry is not None:
        for key in (target_sha, target_id, name):
            if key and key in checkpoint_registry:
                candidate = Path(checkpoint_registry[key])
                if candidate.is_file():
                    return candidate

    if checkpoint is not None:
        candidate = Path(checkpoint)
        if candidate.is_file():
            return candidate

    search_dirs: list[Path] = []
    if checkpoint_dir is not None:
        cdir = Path(checkpoint_dir)
        if cdir.is_dir():
            search_dirs.append(cdir)
    try:
        w_dir = paths.assets() / "weights"
        if w_dir.is_dir() and w_dir not in search_dirs:
            search_dirs.append(w_dir)
    except Exception:
        pass

    candidate_names: list[str] = []
    if target_id and ":" in str(target_id):
        fname = str(target_id).split(":", 1)[0]
        if fname and fname not in candidate_names:
            candidate_names.append(fname)
    default_names = {
        "pix2pix": "pix2pix_generator.pt",
        "spade4tennis": "spade4tennis_lite_generator.pt",
    }
    if name in default_names and default_names[name] not in candidate_names:
        candidate_names.append(default_names[name])

    for sdir in search_dirs:
        for cname in candidate_names:
            candidate = sdir / cname
            if candidate.is_file():
                if target_sha is not None and sha256_file(candidate) == target_sha:
                    return candidate
        if target_sha is not None and not target_sha.startswith("injected:"):
            try:
                for child in sdir.iterdir():
                    if child.is_file() and sha256_file(child) == target_sha:
                        return child
            except OSError:
                pass

    for sdir in search_dirs:
        for cname in candidate_names:
            candidate = sdir / cname
            if candidate.is_file():
                return candidate

    return None


def _from_registry(name: str, checkpoint: Path | str | None = None) -> Any:
    from src.components.generation import REGISTRY
    from src.contracts.conditioning import FrameGenerator
    from src.pipeline.reconstruction.dispatch import from_spec

    if not REGISTRY.has(name):
        return None
    spec = REGISTRY.spec(name)
    kwargs: dict[str, Any] = {}
    if checkpoint is not None:
        kwargs["checkpoint"] = str(checkpoint)
    try:
        backend = REGISTRY.build(name, **kwargs)
    except TypeError:
        if checkpoint is not None:
            raise ValueError(f"Generator {name!r} does not accept a checkpoint parameter")
        backend = REGISTRY.build(name)
    if isinstance(backend, FrameGenerator):
        return from_spec(spec, backend)
    return None


def resolve_client_generator(
    gen_meta: Mapping[str, Any],
    *,
    injected: Any = None,
    require_identity: bool = False,
    checkpoint: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_registry: Mapping[str, str | Path] | None = None,
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

    name = gen_meta.get("name")
    digest = gen_meta.get("checkpoint_sha256")

    if not require_identity and not digest:
        return _from_registry(str(name)) if name else None

    if not name:
        raise ValueError("Payload requires generation but generator name is missing")
    if isinstance(digest, str) and digest.startswith("injected:"):
        raise ValueError(
            f"Cannot reconstruct injected generator {name!r}; "
            "the client was given no matching backend"
        )

    resolved_checkpoint = resolve_client_checkpoint(
        str(name),
        gen_meta,
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        checkpoint_registry=checkpoint_registry,
    )
    resolved = _from_registry(str(name), checkpoint=resolved_checkpoint)
    if resolved is None:
        raise ValueError(f"Unknown generator checkpoint identity: {digest}")

    backend_ckpt = _backend_checkpoint_file(resolved)
    backend_digest = _backend_weight_digest(resolved)
    if backend_ckpt is not None:
        actual_sha = sha256_file(backend_ckpt)
        if actual_sha != digest:
            raise ValueError(
                f"Mismatched checkpoint identity: requested {digest}, "
                f"backend loaded {actual_sha} from {backend_ckpt}"
            )
        if resolved_checkpoint is not None and sha256_file(backend_ckpt) != sha256_file(
            Path(resolved_checkpoint)
        ):
            raise ValueError(
                f"Resolved generator loaded checkpoint {backend_ckpt} "
                f"does not match requested checkpoint {resolved_checkpoint}"
            )
    elif backend_digest is not None:
        if backend_digest != digest:
            raise ValueError(
                f"Mismatched checkpoint digest: requested {digest}, backend has {backend_digest}"
            )

    claimed = identity_from_ref(
        resolved,
        seed=int(gen_meta.get("seed") or 0),
        params=gen_meta.get("params") or {},
        checkpoint=resolved_checkpoint if backend_ckpt is None else None,
    )
    resolved_sha = claimed.get("checkpoint_sha256")
    if resolved_sha != digest:
        raise ValueError(
            f"Mismatched checkpoint identity: requested {digest}, resolved {resolved_sha}"
        )
    return resolved
