from __future__ import annotations

import os
import subprocess
import sys


def _resolve_threshold() -> int:
    override = os.environ.get("POINTSTREAM_COVERAGE_THRESHOLD")
    if override is not None and override.strip() != "":
        try:
            value = int(override)
        except ValueError as exc:
            raise ValueError(
                "POINTSTREAM_COVERAGE_THRESHOLD must be an integer percentage"
            ) from exc
        if value <= 0 or value > 100:
            raise ValueError(
                "POINTSTREAM_COVERAGE_THRESHOLD must be in range 1..100"
            )
        return value

    # Keep CI at policy threshold while requiring a tighter local buffer.
    #
    # 2026-07-22: 80/85 → 78/82 when splitting actor_components.py un-hid
    # previously omitted code. 2026-08-23: 78/82 → 77/81 after retiring the
    # pre-rewrite encoder and the tests that only existed to drive it. Those
    # tests had been padding coverage of shared/decoder as a side effect; the
    # honest remainder is 77%. Ratchet back up as real tests land — never
    # down to accommodate new untested code, and never re-add an omit entry to
    # make the number move.
    if os.environ.get("CI"):
        return 77
    return 81


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    threshold = _resolve_threshold()
    _run(["coverage", "erase"])
    _run(["coverage", "run", "-m", "pytest"])
    _run(["coverage", "report", f"--fail-under={threshold}"])
    _run(["coverage", "xml"])
    print(f"Coverage gate passed at threshold {threshold}%")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
