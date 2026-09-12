"""Client reconstruct must hash the live weight file, not invent injected:arch."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any
import pytest

from src.pipeline.reconstruction import GeneratorRef
from src.runner.generation_identity import (
    identity_from_ref,
    resolve_client_generator,
    sha256_file,
)


def test_file_backed_ref_identity_matches_without_cli_checkpoint(tmp_path: Path) -> None:
    weights = tmp_path / "pix2pix_generator.pt"
    weights.write_bytes(b"dummy-weights")
    ref = GeneratorRef(
        backend=SimpleNamespace(checkpoint=str(weights)),
        name="pix2pix",
    )
    encoded = identity_from_ref(ref, seed=1, params={}, checkpoint=weights)
    reconstructed = identity_from_ref(ref, seed=1, params={})
    assert encoded["checkpoint_sha256"] == sha256_file(weights)
    assert reconstructed["checkpoint_sha256"] == encoded["checkpoint_sha256"]
    resolve_client_generator(encoded, injected=ref, require_identity=True)


def test_nested_adapter_and_counter_still_hash_the_weight_file(tmp_path: Path) -> None:
    """Production path: as_runner_ref + wrap_generator_with_counter."""
    weights = tmp_path / "pix2pix_generator.pt"
    weights.write_bytes(b"dummy-weights")

    class Adapter:
        def __init__(self, inner: Any) -> None:
            self.backend = inner

    class Counter:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    pix = SimpleNamespace(checkpoint=str(weights))
    ref = GeneratorRef(backend=Counter(Adapter(pix)), name="pix2pix")
    encoded = identity_from_ref(ref, seed=1, params={}, checkpoint=weights)
    reconstructed = identity_from_ref(ref, seed=1, params={})
    assert (
        reconstructed["checkpoint_sha256"] == encoded["checkpoint_sha256"] == sha256_file(weights)
    )
    resolve_client_generator(encoded, injected=ref, require_identity=True)


def test_file_backed_mismatch_still_fails_closed(tmp_path: Path) -> None:
    weights = tmp_path / "pix2pix_generator.pt"
    other = tmp_path / "other.pt"
    weights.write_bytes(b"dummy-weights")
    other.write_bytes(b"other-weights")
    ref = GeneratorRef(
        backend=SimpleNamespace(checkpoint=str(other)),
        name="pix2pix",
    )
    payload = identity_from_ref(
        GeneratorRef(backend=SimpleNamespace(checkpoint=str(weights)), name="pix2pix"),
        seed=1,
        params={},
    )
    try:
        resolve_client_generator(payload, injected=ref, require_identity=True)
    except ValueError as exc:
        assert "Mismatched checkpoint identity" in str(exc)
    else:
        raise AssertionError("expected mismatch")


def test_client_rejects_requested_a_when_b_available(tmp_path: Path) -> None:
    weights_a = tmp_path / "weights_a.pt"
    weights_b = tmp_path / "weights_b.pt"
    weights_a.write_bytes(b"checkpoint-A-bytes")
    weights_b.write_bytes(b"checkpoint-B-bytes")

    ref_a = GeneratorRef(backend=SimpleNamespace(checkpoint=str(weights_a)), name="pix2pix")
    payload = identity_from_ref(ref_a, seed=42, params={})
    assert payload["checkpoint_sha256"] == sha256_file(weights_a)

    # Rejection when explicit checkpoint B is provided to a client without injected generator
    try:
        resolve_client_generator(
            payload,
            injected=None,
            checkpoint=weights_b,
            require_identity=True,
        )
    except ValueError as exc:
        assert "Mismatched checkpoint identity" in str(exc)
    else:
        raise AssertionError("expected rejection when checkpoint B provided but A requested")

    # Rejection when checkpoint_dir has only B under candidate name
    client_dir = tmp_path / "client_weights"
    client_dir.mkdir()
    (client_dir / "pix2pix_generator.pt").write_bytes(b"checkpoint-B-bytes")
    try:
        resolve_client_generator(
            payload,
            injected=None,
            checkpoint_dir=client_dir,
            require_identity=True,
        )
    except ValueError as exc:
        assert "Mismatched checkpoint identity" in str(exc)
    else:
        raise AssertionError("expected rejection when checkpoint B in client_dir but A requested")


def test_valid_separate_process_client_without_injected_generator(tmp_path: Path) -> None:
    weights_a = tmp_path / "weights_a.pt"
    weights_a.write_bytes(b"checkpoint-A-bytes")

    ref_a = GeneratorRef(backend=SimpleNamespace(checkpoint=str(weights_a)), name="pix2pix")
    payload = identity_from_ref(ref_a, seed=42, params={})

    # Separate-process client without injected generator resolves with explicit checkpoint
    resolved = resolve_client_generator(
        payload,
        injected=None,
        checkpoint=weights_a,
        require_identity=True,
    )
    assert resolved is not None
    assert resolved.name == "pix2pix"
    assert getattr(resolved.backend, "checkpoint") == str(weights_a)

    # Separate-process client without injected generator resolves via checkpoint_dir
    client_dir = tmp_path / "client_weights"
    client_dir.mkdir(exist_ok=True)
    (client_dir / "pix2pix_generator.pt").write_bytes(b"checkpoint-A-bytes")
    resolved_from_dir = resolve_client_generator(
        payload,
        injected=None,
        checkpoint_dir=client_dir,
        require_identity=True,
    )
    assert resolved_from_dir is not None
    assert resolved_from_dir.name == "pix2pix"
    assert getattr(resolved_from_dir.backend, "checkpoint") == str(
        client_dir / "pix2pix_generator.pt"
    )


def test_resolve_client_generator_rejects_mismatched_backend_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A factory returning backend B when checkpoint A was requested must fail closed."""
    weights_a = tmp_path / "weights_a.pt"
    weights_b = tmp_path / "weights_b.pt"
    weights_a.write_bytes(b"checkpoint-A-content-bytes")
    weights_b.write_bytes(b"checkpoint-B-content-bytes")

    ref_a = GeneratorRef(
        backend=SimpleNamespace(checkpoint=str(weights_a)),
        name="pix2pix",
    )
    ref_b = GeneratorRef(
        backend=SimpleNamespace(checkpoint=str(weights_b)),
        name="pix2pix",
    )

    meta = identity_from_ref(ref_a, seed=42, params={})
    assert meta["checkpoint_sha256"] == sha256_file(weights_a)

    monkeypatch.setattr(
        "src.runner.generation_identity._from_registry",
        lambda name, checkpoint=None: ref_b,
    )

    with pytest.raises(ValueError) as exc_info:
        resolve_client_generator(meta, checkpoint=weights_a, require_identity=True)

    assert "Mismatched checkpoint identity" in str(exc_info.value)
