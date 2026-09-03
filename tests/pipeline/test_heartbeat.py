"""Heartbeat prints while a stage is blocked."""

from __future__ import annotations

import time

from src.pipeline.dag.heartbeat import Heartbeat


def test_heartbeat_emits_still_running_then_done() -> None:
    lines: list[str] = []
    with Heartbeat("work", interval_s=0.05, emit=lines.append):
        time.sleep(0.16)
    assert any("still running" in line for line in lines)
    assert lines[-1].startswith("work done in")


def test_heartbeat_without_interval_only_emits_done() -> None:
    lines: list[str] = []
    with Heartbeat("quiet", interval_s=None, emit=lines.append):
        time.sleep(0.02)
    assert lines == [line for line in lines if line.startswith("quiet done in")]
    assert len(lines) == 1
