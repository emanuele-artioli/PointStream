"""Client reconstruct must hash the live weight file, not invent injected:arch."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

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
