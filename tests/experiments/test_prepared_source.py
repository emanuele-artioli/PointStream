import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from experiments.jobs.prepared_source import verify_prepared_source


def prepared(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    content = b"print('frozen code')\n"
    (source / "run.py").write_bytes(content)
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("run.py")
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def test_completed_unpublished_transfer_has_read_only_proof(tmp_path):
    digest = prepared(tmp_path)
    before = (tmp_path / "source/run.py").read_bytes()
    proof = verify_prepared_source(tmp_path, digest)
    assert proof["verified_files"] == 1 and proof["no_replay"]
    assert (tmp_path / "source/run.py").read_bytes() == before
    assert not (tmp_path / "ready.json").exists()


@pytest.mark.parametrize(
    "fault", ["archive", "missing", "changed", "extra", "ready", "owner", "run", "symlink"]
)
def test_ambiguous_or_incomplete_preparation_cannot_be_published(tmp_path, fault):
    digest = prepared(tmp_path)
    code = tmp_path / "source/run.py"
    if fault == "archive":
        digest = "0" * 64
    elif fault == "missing":
        code.unlink()
    elif fault == "changed":
        code.write_bytes(b"x" * code.stat().st_size)
    elif fault == "extra":
        (tmp_path / "source/extra.py").write_text("unexpected")
    elif fault == "symlink":
        code.unlink()
        code.symlink_to(Path(__file__))
    else:
        name = "ready.json" if fault == "ready" else fault
        (tmp_path / name).mkdir()
    with pytest.raises(ValueError):
        verify_prepared_source(tmp_path, digest)
