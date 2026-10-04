from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from demo.experiments.background_env_probe import ProbeError, run_recorded_probe
from demo.experiments import background_smoke as smoke


def test_real_timeout_kills_probe_and_preserves_prior_receipt(tmp_path):
    root = tmp_path / "receipts"
    run_recorded_probe([sys.executable, "-c", "print('ready')"], name="ready", root=root, timeout=5)
    saved = (root / "ready.json").read_bytes()
    pid = tmp_path / "pid"
    command = [sys.executable, "-c",
               "import os,time,pathlib; pathlib.Path(" + repr(str(pid)) + ").write_text(str(os.getpid())); time.sleep(10)"]
    with pytest.raises(ProbeError, match="slow exceeded"):
        run_recorded_probe(command, name="slow", root=root, timeout=0.4)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)
    receipt = json.loads((root / "slow.json").read_text())
    assert receipt["status"] == "timeout" and receipt["complete"] is False
    assert receipt["seconds"] < 3
    assert json.loads((root / "slow.started.json").read_text())["status"] == "started"
    assert (root / "ready.json").read_bytes() == saved


@pytest.mark.parametrize("code", ["raise SystemExit(3)", "print('not json')", "print('[]')"])
def test_probe_failure_never_publishes_passed_receipt(tmp_path, code):
    with pytest.raises(ProbeError):
        run_recorded_probe([sys.executable, "-c", code], name="broken", root=tmp_path, timeout=5, json_output=True)
    assert json.loads((tmp_path / "broken.json").read_text())["status"] == "failed"


def test_receipts_cannot_be_overwritten(tmp_path):
    command = [sys.executable, "-c", "print('{}')"]
    run_recorded_probe(command, name="once", root=tmp_path, timeout=5)
    with pytest.raises(FileExistsError):
        run_recorded_probe(command, name="once", root=tmp_path, timeout=5)


@pytest.mark.parametrize("timeout", [0, 91, float('inf'), float('nan')])
def test_invalid_timeout_does_not_launch(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        run_recorded_probe(["invalid"], name="bad", root=tmp_path, timeout=timeout)
    assert not list(tmp_path.iterdir())


def _environment_stage(tmp_path, monkeypatch, *, failure=None, lpips="lpips cached"):
    stage_root = tmp_path / "smoke"
    stage_root.mkdir()
    monkeypatch.setenv("PS_STAGE_DIR", str(stage_root))
    stage = smoke.Stage("inventory", 120, [])
    references = {**smoke.core.DCVC_REFERENCE_SHA256, **smoke.core.HNERV_REFERENCE_SHA256}
    monkeypatch.setattr(smoke.core, "sha256_file", lambda path, **kwargs: {"sha256": next(v for k, v in references.items() if str(path).endswith(k))})
    monkeypatch.setattr(smoke.core, "resolve_ffmpeg", lambda: None)

    def fake(command, *, name, root, timeout, **kwargs):
        from demo.experiments.background_env_probe import _write_new
        if name == failure:
            _write_new(root / (name + '.json'), {"status": "timeout", "complete": False})
            raise ProbeError(name + " exceeded allowance")
        if "-git-" in name:
            output = 'head'
        elif name == "dcvc-adapter":
            report = Path(command[command.index("--report") + 1])
            report.write_text(json.dumps({"reference": {"matches_reference": True}, "environment": {"extension": {"module": "installed"}}}))
            output = '{}'
        elif name == "hnerv-import":
            output = '{"stubbed_modules": [], "model_all": "/installed/model_all.py", "hnerv_utils": "/installed/hnerv_utils.py"}'
        else:
            output = json.dumps({"status": lpips})
        return subprocess.CompletedProcess(command, 0, output, '')

    monkeypatch.setattr("demo.experiments.background_env_probe.run_recorded_probe", fake)
    return stage


def test_environment_timeout_retains_source_and_adapter_progress(tmp_path, monkeypatch):
    stage = _environment_stage(tmp_path, monkeypatch, failure="hnerv-import")
    result = stage.run_part("env", lambda: smoke.inventory_env(stage))
    assert result["status"] == "failed" and "hnerv-import" in result["reason"]
    assert json.loads((stage.root / "environment-progress/dcvc-adapter.json").read_text())["dcvc"]["adapter_check"]["reference"]["matches_reference"]
    partial = json.loads((stage.root / "partial-inputs/env.json").read_text())
    assert partial["environment"]["stage_local"]["dcvc"]["adapter_check"]
    assert json.loads((stage.root / "environment-probes/hnerv-import.json").read_text())["status"] == "timeout"
    assert "lpips" not in stage.environment


def test_missing_cached_lpips_blocks_environment(tmp_path, monkeypatch):
    stage = _environment_stage(tmp_path, monkeypatch, lpips="blocked: weights absent")
    result = stage.run_part("env", lambda: smoke.inventory_env(stage))
    assert result["status"] == "blocked" and "weights absent" in result["reason"]
    assert stage.environment["lpips"].startswith("blocked:")


def test_complete_environment_passes_and_preserves_each_probe(tmp_path, monkeypatch):
    stage = _environment_stage(tmp_path, monkeypatch)
    result = stage.run_part("env", lambda: smoke.inventory_env(stage))
    assert result["status"] == "passed"
    progress = stage.root / "environment-progress"
    assert {p.stem for p in progress.iterdir()} == {"dcvc-source", "hnerv-source", "dcvc-adapter", "hnerv-import", "ffmpeg", "lpips"}
    assert json.loads((progress / "lpips.json").read_text())["lpips"] == "lpips cached"


def test_git_timeout_records_repository_and_stops_later_probes(tmp_path, monkeypatch):
    stage = _environment_stage(tmp_path, monkeypatch, failure="dcvc-git-diff")
    result = stage.run_part("env", lambda: smoke.inventory_env(stage))
    assert result["status"] == "failed" and "dcvc-git-diff" in result["reason"]
    assert json.loads((stage.root / "environment-probes/dcvc-git-diff.json").read_text())["status"] == "timeout"
    assert not (stage.root / "environment-progress").exists()
    assert not stage.environment
