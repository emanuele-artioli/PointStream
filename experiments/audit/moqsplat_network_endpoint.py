"""Bounded CPU-only native QUIC endpoint for a separate-machine checker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    stage = Path(os.environ["PS_STAGE_DIR"])
    if args.validate:
        result = json.loads((stage / "summary.json").read_text())
        if not result["valid"] or not result["external_client"]["source_attributes_exact"]:
            raise ValueError("separate-machine QUIC check did not pass")
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(
            json.dumps({"passed": True, "checks": result})
        )
        return
    assert args.source and args.scene and args.python and args.port
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    checkout = scratch / "source"
    checkout.mkdir()
    with tarfile.open(args.source) as archive:
        archive.extractall(checkout, filter="data")
    temporary = scratch / "tmp"
    temporary.mkdir()
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(checkout / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(temporary),
        TMP=str(temporary),
        TEMP=str(temporary),
    )
    runner = [str(args.python), str(checkout / "scripts/run_moq_reproduction.py")]
    processes: list[subprocess.Popen[bytes]] = []
    result: dict[str, object] = {"valid": False, "device_type": "cpu", "rendering_tested": False}
    try:
        commands = [
            ("relay", runner + ["relay", "--bind", f"0.0.0.0:{args.port}", "--duration", "140"]),
            (
                "publisher",
                runner
                + [
                    "publish",
                    "--relay",
                    f"https://127.0.0.1:{args.port}",
                    "--insecure",
                    "--scene",
                    str(args.scene),
                    "--max-splats",
                    "65536",
                    "--group-splats",
                    "16384",
                    "--broadcast",
                    "qualification/external",
                    "--duration",
                    "140",
                ],
            ),
        ]
        for name, argv in commands:
            log = stage / f"{name}.log"
            with log.open("wb") as stream:
                process = subprocess.Popen(
                    argv, env=env, cwd=checkout, stdout=stream, stderr=subprocess.STDOUT
                )
            processes.append(process)
            until = time.monotonic() + 45
            while '"ready": true' not in log.read_text():
                if process.poll() is not None or time.monotonic() > until:
                    raise RuntimeError(f"{name} not ready: {log.read_text()[-2000:]}")
                time.sleep(0.2)
        (stage / "endpoint.json").write_text(json.dumps({"port": args.port, "ready": True}))
        until = time.monotonic() + 100
        proof = stage / "external-client.json"
        while not proof.exists():
            if time.monotonic() > until or any(p.poll() is not None for p in processes):
                raise RuntimeError("external client proof absent before bounded endpoint deadline")
            time.sleep(0.2)
        client = json.loads(proof.read_text())
        if not (
            client.get("valid")
            and client.get("transport") == "QUIC"
            and client.get("source_attributes_exact")
            and client.get("layers_received", 0) >= 1
            and client.get("cache_revisit_requests") == 0
        ):
            raise ValueError("independent client proof failed")
        result.update(
            valid=True,
            transport="QUIC",
            external_client=client,
            gaussian_subset=65536,
            endpoint_port=args.port,
            commands=commands,
        )
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        (stage / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
