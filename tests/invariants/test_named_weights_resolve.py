"""Named weights in shipped configs are well-formed, and exist where the files do.

Structural checks run everywhere, including CI (no files on disk). Existence
checks run only where ``assets/weights/`` is present: they are marked
``integration`` so the default runner deselects them, and they **fail** rather
than skip when the tree is present but a named file is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.probe.weights import (
    already_under_weights,
    collect_shipped_weights,
    collect_structural_weights,
    names_from_yaml,
    resolve_named_artifact,
    shipped_yaml_paths,
    structural_error,
    structural_names_from_yaml,
    unresolved,
    weights_are_present,
)
from src.components.detection.weights import intended_weight_path, repo_root


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


def test_a_doubled_weights_prefix_is_not_well_formed() -> None:
    name = "assets/weights/custom-controlnet"
    assert already_under_weights(name)
    err = structural_error(name, "controlnet-id")
    assert err is not None
    assert "doubled prefix" in err
    planted = intended_weight_path(name, root=Path("/repo"))
    assert "assets/weights/assets/weights" not in planted.as_posix()


def test_a_yolo_typo_is_not_well_formed() -> None:
    err = structural_error("yolo26x-eg.pt", "detector")
    assert err is not None
    assert "yolo26x-eg.pt" in err
    assert structural_error("yolo26x-seg.pt", "detector") is None
    assert structural_error("yolo26n.pt", "detector") is None
    assert structural_error("yolo26n-pose.pt", "pose-estimator") is None


def test_a_prefixed_name_does_not_plant_a_doubled_path(tmp_path: Path) -> None:
    planted = tmp_path / "assets" / "weights" / "custom-controlnet"
    planted.mkdir(parents=True)
    (planted / "config.json").write_text("{}")
    resolved = resolve_named_artifact("assets/weights/custom-controlnet", root=tmp_path)
    assert resolved == planted.resolve()
    assert "assets/weights/assets/weights" not in resolved.as_posix()


def test_every_shipped_config_weight_is_well_formed() -> None:
    found = collect_structural_weights()
    assert found, "no named weights were discovered in shipped configs"
    broken = unresolved(found)
    assert not broken, (
        "named weight(s) are not well-formed: "
        + "; ".join(f"{item.source}:{item.slot}={item.name!r} ({item.error})" for item in broken)
    )


def test_quality_tier_names_yolo26x_seg_not_the_eg_typo() -> None:
    path = repo_root() / "config" / "tier_quality.yaml"
    names = {item.name for item in structural_names_from_yaml(path)}
    assert "yolo26x-eg.pt" not in names
    assert "yolo26x-seg.pt" in names


def test_controlnet_id_is_a_bare_name() -> None:
    for path in shipped_yaml_paths():
        for item in structural_names_from_yaml(path):
            if item.slot.replace("_", "-").endswith("controlnet-id"):
                assert not already_under_weights(item.name), (
                    f"{path}: controlnet-id={item.name!r} still carries a weights prefix"
                )


@pytest.mark.integration
def test_every_shipped_config_weight_exists() -> None:
    """Fail when the weights tree is present but a named file is missing.

    Deselected on CI by the ``integration`` marker (no files there). Skip only
    when the tree itself is absent — never when it is present and incomplete.
    """
    if not weights_are_present():
        pytest.skip("assets/weights/ is not present")
    found = collect_shipped_weights()
    assert found, "no named weights were discovered in shipped configs"
    broken = unresolved(found)
    assert not broken, (
        "named weight(s) do not resolve: "
        + "; ".join(f"{item.source}:{item.slot}={item.name!r} ({item.error})" for item in broken)
    )
