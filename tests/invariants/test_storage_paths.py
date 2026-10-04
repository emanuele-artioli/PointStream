from pathlib import Path

import pytest

from src.contracts import paths
from src.components.detection.weights import (
    intended_weight_path,
    resolve_weight,
    WeightResolutionError,
)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    home = tmp_path / "home"
    checkout = home / "pointstream"
    checkout.mkdir(parents=True)
    monkeypatch.setattr(paths, "_REPO_ROOT", checkout)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    for key in ("PS_DATA_ROOT", "PS_MODELS_ROOT", "POINTSTREAM_MODELS"):
        monkeypatch.delenv(key, raising=False)
    return home, checkout


def test_data_and_models_use_distinct_canonical_roots(layout):
    home, checkout = layout
    (home / "Datasets").mkdir()
    assert paths.repo_root() == checkout
    assert paths.data_root() == home / "Datasets"
    assert paths.models_root() == home / "Models"
    assert paths.describe()["models_root"] == str(home / "Models")


def test_environment_and_markers_are_authoritative_even_when_missing(layout, monkeypatch):
    home, checkout = layout
    (home / "Datasets").mkdir()
    (checkout / ".ps-data-root").write_text(str(home / "old-data"))
    (checkout / ".ps-models-root").write_text(str(home / "marked-models"))
    assert paths.data_root() == home / "old-data"
    assert paths.models_root() == home / "marked-models"
    monkeypatch.setenv("PS_DATA_ROOT", str(home / "explicit-data"))
    monkeypatch.setenv("PS_MODELS_ROOT", str(home / "explicit-models"))
    assert paths.data_root() == home / "explicit-data"
    assert paths.models_root() == home / "explicit-models"


def test_legacy_checkout_still_resolves_weights(layout):
    home, checkout = layout
    old = checkout / "assets/weights"
    old.mkdir(parents=True)
    checkpoint = old / "yolo26n.pt"
    checkpoint.write_bytes(b"legacy")
    assert paths.data_root() == checkout
    assert resolve_weight("yolo26n.pt") == checkpoint
    assert intended_weight_path("assets/weights/yolo26n.pt") == checkpoint


def test_family_paths_and_legacy_prefix_resolve_in_models(layout):
    home, _ = layout
    (home / "Datasets").mkdir()
    model = home / "Models/YOLO/yolo26n.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"checkpoint")
    assert resolve_weight("yolo26n.pt") == model
    assert resolve_weight("assets/weights/yolo26n.pt") == model


def test_explicit_model_root_cannot_silently_load_legacy_weights(layout, monkeypatch):
    home, checkout = layout
    old = checkout / "assets/weights/yolo26n.pt"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"wrong identity")
    monkeypatch.setenv("POINTSTREAM_MODELS", str(home / "absent-models"))
    with pytest.raises(WeightResolutionError):
        resolve_weight("yolo26n.pt")
    assert paths.models_root() == home / "absent-models"
    assert paths.model_asset("yolo26n.pt").parent == home / "absent-models/YOLO"


def test_dangling_canonical_alias_is_an_error_not_a_legacy_fallback(layout):
    home, checkout = layout
    old = checkout / "assets/weights/yolo26n.pt"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"legacy")
    link = home / "Models/YOLO/yolo26n.pt"
    link.parent.mkdir(parents=True)
    link.symlink_to(home / "missing.pt")
    with pytest.raises(WeightResolutionError, match="dangling"):
        resolve_weight("yolo26n.pt")


def test_model_path_traversal_is_rejected(layout):
    with pytest.raises(ValueError):
        paths.model_asset("../credentials")


def test_explicit_model_root_rejects_same_name_in_working_directory(layout, monkeypatch):
    home, checkout = layout
    monkeypatch.chdir(checkout)
    (checkout / "yolo26n.pt").write_bytes(b"unselected checkpoint")
    monkeypatch.setenv("PS_MODELS_ROOT", str(home / "Models"))
    with pytest.raises(WeightResolutionError):
        resolve_weight("yolo26n.pt")


def test_download_presence_checks_use_family_paths(layout):
    from scripts.download_weights import ensure_weights

    home, _ = layout
    (home / "Datasets").mkdir()
    checkpoint = home / "Models/YOLO/yolo26n.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    ensure_weights(paths.models_root(), ["yolo26n.pt"])


def test_empty_model_marker_does_not_disable_legacy_lookup(layout):
    home, checkout = layout
    (home / "Models").mkdir()
    (checkout / ".ps-models-root").write_text(" ")
    checkpoint = checkout / "assets/weights/yolo26n.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"legacy")
    assert paths.model_asset("yolo26n.pt") == checkpoint
