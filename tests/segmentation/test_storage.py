from pathlib import Path

import pytest

from src.segmentation import storage


def test_environment_overrides_the_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PS_MODELS_ROOT", str(tmp_path / "m"))
    monkeypatch.setenv("PS_DATASETS_ROOT", str(tmp_path / "d"))
    assert storage.models_root() == tmp_path / "m"
    assert storage.datasets_root() == tmp_path / "d"


def test_links_then_canonical_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PS_MODELS_ROOT", raising=False)
    monkeypatch.delenv("PS_DATASETS_ROOT", raising=False)
    monkeypatch.setattr(storage, "REPO_ROOT", tmp_path)
    assert storage.models_root() == storage.CANONICAL_MODELS
    (tmp_path / "store").mkdir()
    (tmp_path / "Datasets").symlink_to(tmp_path / "store")
    assert storage.datasets_root() == tmp_path / "Datasets"


def test_missing_or_dangling_weight_names_the_expected_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PS_MODELS_ROOT", str(tmp_path))
    (tmp_path / "YOLO").mkdir()
    (tmp_path / "YOLO" / "dangling.pt").symlink_to(tmp_path / "gone.pt")
    for name in ("absent.pt", "dangling.pt"):
        with pytest.raises(FileNotFoundError, match=str(tmp_path / "YOLO" / name)):
            storage.model_path("YOLO", name)
    (tmp_path / "YOLO" / "real.pt").write_bytes(b"w")
    assert storage.model_path("YOLO", "real.pt") == tmp_path / "YOLO" / "real.pt"
