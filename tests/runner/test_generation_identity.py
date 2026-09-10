"""Client reconstruct must hash the live weight file, not invent injected:arch."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any

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
    assert reconstructed["checkpoint_sha256"] == encoded["checkpoint_sha256"] == sha256_file(weights)
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
