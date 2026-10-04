"""Bounded environment probes; heavy dependencies are imported only in children."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class ProbeError(RuntimeError):
    pass


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def run_recorded_probe(command: list[str], *, name: str, root: Path,
                       timeout: float, cwd: Path | None = None,
                       env: dict[str, str] | None = None, check: bool = True,
                       json_output: bool = False) -> subprocess.CompletedProcess:
    """Persist admission and outcome separately, even on timeout/nonzero exit.

    Children inherit the fleet job's process group, so its supervisor can
    cancel the entire task. Probe entry points perform imports in-process;
    they do not launch another model process or modify the environment.
    """
    if not math.isfinite(timeout) or not 0 < timeout <= 90:
        raise ValueError("probe timeout must be in (0, 90] seconds")
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name):
        raise ValueError("invalid probe name")
    started = time.time()
    receipt = {"name": name, "command": command, "cwd": str(cwd) if cwd else None,
               "timeout_seconds": timeout, "started": started,
               "status": "started", "complete": False}
    _write_new(root / f"{name}.started.json", receipt)
    before = time.monotonic()
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                                text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        receipt.update(status="timeout", seconds=time.monotonic() - before,
                       reason=f"{name} exceeded {timeout:g}s")
        _write_new(root / f"{name}.json", receipt)
        raise ProbeError(receipt["reason"]) from None
    except OSError as exc:
        receipt.update(status="failed", seconds=time.monotonic() - before, reason=str(exc))
        _write_new(root / f"{name}.json", receipt)
        raise ProbeError(f"{name}: {exc}") from exc
    receipt.update(status="passed" if result.returncode == 0 else "failed",
                   complete=True, seconds=time.monotonic() - before,
                   exit_code=result.returncode, stdout_tail=result.stdout[-4000:],
                   stderr_tail=result.stderr[-4000:])
    if json_output and result.returncode == 0:
        try:
            value = json.loads(result.stdout.strip().splitlines()[-1])
            if not isinstance(value, dict) or not value:
                raise ValueError("expected a nonempty JSON object")
        except (ValueError, IndexError) as exc:
            receipt.update(status="failed", reason=f"{name} returned invalid JSON: {exc}")
            _write_new(root / f"{name}.json", receipt)
            raise ProbeError(receipt["reason"]) from exc
    _write_new(root / f"{name}.json", receipt)
    if check and result.returncode:
        raise ProbeError(f"{name} exited {result.returncode}: {result.stderr[-800:]}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("hnerv-import", "lpips"))
    args = parser.parse_args(argv)
    if args.kind == "hnerv-import":
        from demo.experiments.hnerv_frozen import enable_imports
        value = enable_imports(Path("unused-no-stubs"))
    else:
        from demo.experiments.background_smoke import _lpips_status
        value = {"status": _lpips_status()}
    print(json.dumps(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
