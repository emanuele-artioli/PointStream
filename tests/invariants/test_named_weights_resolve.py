"""Every weight a shipped config names resolves to a real file or directory."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.probe.weights import (
    collect_shipped_weights,
    names_from_yaml,
    resolve_named_artifact,
    unresolved,
)


def test_a_missing_weight_is_refused_rather_than_downloaded(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-backend.pt"
    with pytest.raises(FileNotFoundError):
        resolve_named_artifact(str(missing), root=tmp_path)


def test_a_dangling_symlink_is_its_own_error(tmp_path: Path) -> None:
    link = tmp_path / "ghost.pt"
    link.symlink_to(tmp_path / "nowhere.pt")
    with pytest.raises(FileNotFoundError, match="dangling"):
        resolve_named_artifact(str(link), root=tmp_path)


def test_a_yaml_that_names_a_missing_file_is_unresolved(tmp_path: Path) -> None:
    path = tmp_path / "tier.yaml"
    path.write_text(yaml.safe_dump({"detector": "definitely-missing-yolo.pt"}))
    found = names_from_yaml(path)
    assert len(found) == 1
    assert found[0].ok is False
    assert found[0].error is not None


def test_every_shipped_config_weight_resolves() -> None:
    found = collect_shipped_weights()
    assert found, "no named weights were discovered in shipped configs"
    broken = unresolved(found)
    assert not broken, (
        "named weight(s) do not resolve: "
        + "; ".join(f"{item.source}:{item.slot}={item.name!r} ({item.error})" for item in broken)
    )
