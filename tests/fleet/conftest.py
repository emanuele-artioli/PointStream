import pytest

from experiments.jobs import staging


@pytest.fixture(autouse=True)
def no_host_local_root(tmp_path, monkeypatch):
    """Fleet tests never write to a real host's /local/users cache."""
    monkeypatch.setenv(staging.LOCAL_ROOT_ENV, str(tmp_path / "no-local-root" / "pointstream"))
