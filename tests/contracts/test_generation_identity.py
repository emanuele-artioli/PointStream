"""Contract tests for client generator checkpoint and configuration identity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest

from src.pipeline.reconstruction.dispatch import GeneratorRef
from src.runner.generation_identity import (
    identity_from_ref,
    resolve_client_generator,
    sha256_file,
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
