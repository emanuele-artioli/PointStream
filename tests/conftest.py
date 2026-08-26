from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest


def _create_test_run_artifacts_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "outputs" / "tests" / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.fixture(scope="session")
def test_run_artifacts_dir() -> Path:
    return _create_test_run_artifacts_dir()


@pytest.fixture(scope="session", autouse=True)
def configure_test_debug_artifact_env(test_run_artifacts_dir: Path):
    previous = os.environ.get("POINTSTREAM_DEBUG_ARTIFACT_DIR")
    os.environ["POINTSTREAM_DEBUG_ARTIFACT_DIR"] = str(test_run_artifacts_dir)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("POINTSTREAM_DEBUG_ARTIFACT_DIR", None)
        else:
            os.environ["POINTSTREAM_DEBUG_ARTIFACT_DIR"] = previous
