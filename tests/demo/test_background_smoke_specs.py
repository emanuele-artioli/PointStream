from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

import pytest

from demo.experiments import background_smoke_specs as specs
from demo.experiments.background_smoke_core import DATA_ROOT
from experiments.jobs import fleet, inbox, monitor

REPO = Path(__file__).resolve().parents[2]
INPUT = [{"path": str(DATA_ROOT / "jobs/fleet/inbox/x/smoke/selected-inputs.json"), "sha256": "a" * 64}]


def deadline(minutes: int = 90) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat(timespec="seconds")


def all_specs() -> list[dict]:
    return [specs.build_spec(kind, inputs=INPUT, manifests=[INPUT[0]["path"]], deadline=deadline()) for kind in specs.PLANS]


def test_every_planned_spec_passes_the_dispatcher_schema_and_plan_caps() -> None:
    planned = all_specs()
    for spec in planned:
        inbox.validate_spec(spec, now=time.time())
        assert spec["budget_seconds"] <= specs.JOB_BUDGET_SECONDS
        assert spec["stall_seconds"] <= spec["budget_seconds"]
        assert spec["scale"]["parts"]["smoke"] != spec["scale"]["parts"]["full"]
        assert spec["entrypoint"] == ["-m", "demo.experiments.background_smoke"]
    caps = specs.check_plan(planned)
    assert caps["budget_seconds_total"] <= specs.PLAN_BUDGET_SECONDS
    stage = {s["arguments"][0]: s["budget_seconds"] for s in planned}
    assert stage["codec"] <= 12 * 60 and stage["drift"] <= 5 * 60 and stage["latent"] <= 8 * 60


def test_combined_work_above_the_ceiling_is_rejected() -> None:
    planned = all_specs()
    with pytest.raises(ValueError, match="plan ceiling"):
        specs.check_plan(planned + [max(planned, key=lambda spec: spec["budget_seconds"])])
    with pytest.raises(ValueError, match="plan ceiling"):
        specs.check_plan(planned, spent_seconds=600)
    big = dict(planned[0], budget_seconds=600, stall_seconds=600)
    with pytest.raises(ValueError, match="per-job cap"):
        specs.check_plan([big])


def test_missing_hash_or_outside_input_is_rejected() -> None:
    spec = all_specs()[0]
    with pytest.raises(fleet.FleetError, match="SHA256"):
        inbox.validate_spec({**spec, "inputs": [{"path": INPUT[0]["path"], "sha256": "REPLACE"}]}, now=time.time())
    with pytest.raises(ValueError, match="real SHA-256"):
        specs.check_plan([{**spec, "inputs": [{"path": "/tmp/x.json", "sha256": "a" * 64}]}])


def test_invalid_scale_arguments_are_rejected() -> None:
    spec = all_specs()[0]
    bad = {**spec, "arguments": [a if a != "{parts}" else "parts={parts}" for a in spec["arguments"]]}
    with pytest.raises(fleet.FleetError):
        inbox.validate_spec(bad, now=time.time())
    same = {**spec, "scale": {**spec["scale"], "parts": {"smoke": "env", "full": "env"}}}
    with pytest.raises(ValueError, match="remainder"):
        specs.check_plan([same])


def test_validator_must_be_the_substantive_gate() -> None:
    spec = all_specs()[0]
    with pytest.raises(ValueError, match="substantive"):
        specs.check_plan([{**spec, "validator": ["{python}", "-m", "json.tool"]}])
    with pytest.raises(ValueError, match="substantive|inline"):
        specs.check_plan([{**spec, "validator": ["{python}", "-c", "print(1)", "validate", "--kind"]}])


@pytest.fixture
def campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A real campaign whose validator is background_smoke's codec gate."""
    directory = tmp_path / "jobs" / "fleet" / "inbox" / "20261004T000000Z-abcdef12"
    source = directory / "source"
    source.mkdir(parents=True)
    (directory / "run").mkdir()
    identity = tmp_path / "selected-inputs.json"
    identity.write_text("{}")
    (source / "work.py").write_text('''import json, os, pathlib, sys
stage = pathlib.Path(os.environ["PS_STAGE_DIR"])
status = sys.argv[sys.argv.index("--status") + 1]
stage.joinpath("result.json").write_text(json.dumps({
    "kind": "codec", "citable": False, "parts_requested": ["f001c3-htl-pre"],
    "parts": {"f001c3-htl-pre": {"status": status}},
    "provenance": {"code_revision": "f" * 40, "gpu_uuid": "GPU-test"}, "summary": {"status": status}}))
''')
    (directory / "ready.json").write_text("{}")
    spec = {
        "schema": 1, "hosts": ["gpu5"], "gpu_models": [], "gpu_memory_mib": 100, "cpu_threads": 1,
        "entrypoint": ["work.py"], "arguments": ["--status", "{status}"],
        "scale": {"status": {"smoke": "failed", "full": "passed"}},
        "inputs": [{"path": str(identity), "sha256": inbox.file_digest(identity)}],
        "smoke": {"seconds": 20, "representative_basis": "fixture"}, "full": {"seconds": 20}, "budget_seconds": 90,
        "deadline": deadline(), "validator": ["{python}", "-m", "demo.experiments.background_smoke", "validate", "--kind", "codec"],
        "validator_seconds": 30, "required_commands": [],
    }
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setenv("PS_JOB_DIR", str(directory / "run"))

    def prepare(smoke_status: str) -> Path:
        value = inbox.validate_spec({**spec, "scale": {"status": {"smoke": smoke_status, "full": "passed"}}})
        monitor.write_json(directory / "spec.json", value)
        monitor.write_json(directory / "ready.json", {"spec_sha256": inbox.digest(value), "source_sha256": inbox.source_identity(source)})
        return directory

    return prepare


def test_validator_failure_blocks_the_full_stage(campaign) -> None:
    directory = campaign("failed")
    assert inbox.campaign(directory) == 1
    assert not (directory / "full").exists() and not (directory / "gate.json").exists()
    assert json.loads((directory / "validation.json").read_text())["passed"] is False


def test_passing_validator_promotes_with_substantive_checks(campaign) -> None:
    directory = campaign("passed")
    assert inbox.campaign(directory) == 0
    gate = json.loads((directory / "gate.json").read_text())
    assert gate["validation"]["checks"]["assertions"]["all_parts_passed"] is True
    assert (directory / "full" / "result.json").is_file()
