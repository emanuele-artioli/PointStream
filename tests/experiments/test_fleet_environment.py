"""A packed environment must be exactly one consistent state of its source."""
import json
from pathlib import Path
import tarfile

import pytest

from experiments.jobs import environment, inbox


def fake_prefix(root: Path) -> Path:
    prefix = root / "envs" / "demo"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("#!/bin/sh\n")
    (prefix / "bin" / "python").chmod(0o755)
    (prefix / "bin" / "python3").symlink_to("python")
    (prefix / "conda-meta").mkdir()
    (prefix / "conda-meta" / "numpy-1.0-0.json").write_text("{}")
    site = prefix / "lib" / "python3.10" / "site-packages" / "torch-2.2.2.dist-info"
    site.mkdir(parents=True)
    (site / "RECORD").write_text("torch/__init__.py\n")
    return prefix


def test_pack_publishes_identified_archive_and_manifest(tmp_path):
    prefix = fake_prefix(tmp_path)
    result = environment.pack(prefix, tmp_path / "shared", tmp_path / "local")
    archive = Path(result["path"])
    assert inbox.file_digest(archive) == result["sha256"]
    with tarfile.open(archive) as bundle:
        names = set(bundle.getnames())
        assert {"./bin/python", "./bin/python3", "./conda-meta/numpy-1.0-0.json"} <= names
        assert bundle.getmember("./bin/python3").issym()
    manifest = json.loads(archive.with_name(archive.name.replace(".tar.gz", ".json")).read_text())
    assert manifest["sha256"] == result["sha256"]
    assert manifest["packages"]["pip"] == ["torch-2.2.2.dist-info"]
    assert not list((tmp_path / "local").iterdir())
    assert not list((tmp_path / "shared").glob(".*partial"))


def test_install_during_packing_discards_the_archive(tmp_path, monkeypatch):
    prefix = fake_prefix(tmp_path)
    states = iter([{"metadata_sha256": "before"}, {"metadata_sha256": "after"}])
    monkeypatch.setattr(environment, "package_state", lambda path: next(states))
    with pytest.raises(environment.PackError):
        environment.pack(prefix, tmp_path / "shared", tmp_path / "local")
    assert not (tmp_path / "shared").exists()
    assert not list((tmp_path / "local").iterdir())


def test_non_environment_is_refused(tmp_path):
    with pytest.raises(environment.PackError):
        environment.pack(tmp_path, tmp_path / "shared", tmp_path / "local")
