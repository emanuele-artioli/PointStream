"""Protect existing artifacts and authenticate saved inputs before use."""
import hashlib

import pytest

from experiments.tennis import paper_saved_evidence as evidence


def test_output_must_be_fresh_and_outside_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(evidence, "ROOT", root)
    with pytest.raises(ValueError, match="outside"):
        evidence.fresh_output(root / "new")
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep"
    sentinel.write_text("evidence")
    with pytest.raises(FileExistsError):
        evidence.fresh_output(existing)
    assert sentinel.read_text() == "evidence"
    link = tmp_path / "checkout-link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        evidence.fresh_output(link / "new")


def test_authentication_precedes_json_parsing(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"value": 3}')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert evidence.authenticated_json(source, expected) == {"value": 3}
    source.write_text("not json")
    with pytest.raises(ValueError, match="SHA256"):
        evidence.authenticated_json(source, expected)
