from __future__ import annotations

import subprocess

import pytest

from scripts import check_coverage_gate as gate


def test_resolve_threshold_defaults_to_local_buffer(monkeypatch) -> None:
    monkeypatch.delenv("POINTSTREAM_COVERAGE_THRESHOLD", raising=False)
    monkeypatch.delenv("CI", raising=False)
    assert gate._resolve_threshold() == 85


def test_resolve_threshold_defaults_to_ci_gate(monkeypatch) -> None:
    monkeypatch.delenv("POINTSTREAM_COVERAGE_THRESHOLD", raising=False)
    monkeypatch.setenv("CI", "true")
    assert gate._resolve_threshold() == 80


def test_resolve_threshold_accepts_valid_override(monkeypatch) -> None:
    monkeypatch.setenv("POINTSTREAM_COVERAGE_THRESHOLD", "85")
    assert gate._resolve_threshold() == 85


def test_resolve_threshold_rejects_non_integer_override(monkeypatch) -> None:
    monkeypatch.setenv("POINTSTREAM_COVERAGE_THRESHOLD", "eighty")
    with pytest.raises(ValueError, match="must be an integer"):
        gate._resolve_threshold()


def test_resolve_threshold_rejects_out_of_range_override(monkeypatch) -> None:
    monkeypatch.setenv("POINTSTREAM_COVERAGE_THRESHOLD", "0")
    with pytest.raises(ValueError, match="must be in range 1..100"):
        gate._resolve_threshold()


def test_run_raises_system_exit_when_command_fails(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda cmd, check=False: subprocess.CompletedProcess(cmd, 2))
    with pytest.raises(SystemExit) as exc:
        gate._run(["coverage", "report"])
    assert exc.value.code == 2


def test_main_runs_coverage_commands_with_resolved_threshold(monkeypatch, capsys) -> None:
    monkeypatch.setenv("POINTSTREAM_COVERAGE_THRESHOLD", "83")

    seen: list[list[str]] = []
    monkeypatch.setattr(gate, "_run", lambda cmd: seen.append(cmd))

    exit_code = gate.main()
    assert exit_code == 0
    assert seen == [
        ["coverage", "erase"],
        ["coverage", "run", "-m", "pytest"],
        ["coverage", "report", "--fail-under=83"],
        ["coverage", "xml"],
    ]

    out = capsys.readouterr().out
    assert "Coverage gate passed at threshold 83%" in out