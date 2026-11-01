"""Run an explicitly identified external MoQSplat revision through fleet limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import time


def wait_ready(process: subprocess.Popen[bytes], log: Path, seconds: float) -> None:
    """Read bounded local readiness evidence without replaying failed children."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError(f"MoQ child exited {process.returncode}: {log.read_text()[-3000:]}")
        if log.exists() and '"ready": true' in log.read_text():
            return
        time.sleep(0.1)
    raise TimeoutError(f"MoQ child readiness expired: {log}")


def native_run(checkout: Path, scratch: Path, args: argparse.Namespace,
               environment: dict[str, str]) -> None:
    """Three task-owned processes; relay and publisher are always reaped."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"127.0.0.1:{port}"
    relay_url = f"https://{address}"
    base = [str(args.python), str(checkout / "scripts" / "run_moq_reproduction.py")]
    logs = scratch / "native-logs"
    logs.mkdir()
    children: list[subprocess.Popen[bytes]] = []
    commands = [base + ["relay", "--bind", address, "--duration", "600"],
                base + ["publish", "--scene", str(args.scene), "--relay", relay_url,
                        "--insecure", "--duration", "600"]]
    try:
        for role, command in zip(("relay", "publisher"), commands):
            log = logs / f"{role}.log"
            with log.open("wb") as stream:
                child = subprocess.Popen(command, cwd=checkout, env=environment,
                                         stdout=stream, stderr=subprocess.STDOUT)
            children.append(child)
            wait_ready(child, log, 300 if role == "publisher" else 30)
        command = base + ["subscribe", "--trace", str(args.trace), "--relay", relay_url,
                          "--insecure", "--device", "cuda:0", "--frames", str(args.frames),
                          "--width", "320", "--height", "240", "--timeout", "180",
                          "--output", str(scratch / "publish")]
        commands.append(command)
        with (logs / "subscriber.log").open("wb") as stream:
            subprocess.run(command, cwd=checkout, env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=300)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
        published = scratch / "publish"
        published.mkdir(exist_ok=True)
        shutil.copytree(logs, published / "native-logs")
        (published / "commands.json").write_text(json.dumps(commands, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/miniconda3/envs/cuda"))
    parser.add_argument("--frames", type=int)
    parser.add_argument("--mode", choices=["render", "moq", "combined"], default="render")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    stage = Path(os.environ["PS_STAGE_DIR"])
    report = stage / "summary.json"
    if args.validate:
        result = json.loads(report.read_text())
        if result.get("valid") is not True or result.get("frames_rendered", 0) < 1:
            raise ValueError("MoQSplat did not produce valid real-scene frames")
        if result.get("device_type") != "cuda" or not result.get("wire_roundtrip_exact"):
            raise ValueError("CUDA and exact wire roundtrip were not established")
        if args.mode in {"moq", "combined"} and result.get("transport") != "QUIC":
            raise ValueError("Native MoQ/QUIC transport was not established")
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps({"passed": True, "checks": result}))
        print(json.dumps({"passed": True, "frames": result["frames_rendered"]}))
        return
    assert args.source and args.scene and args.trace and args.python and args.frames
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    checkout = scratch / "moqsplat"
    checkout.mkdir()
    with tarfile.open(args.source) as archive:
        archive.extractall(checkout, filter="data")
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(checkout / "src"), PYTHONDONTWRITEBYTECODE="1",
                       PATH=os.pathsep.join([str(args.python.parent), str(args.cuda_home / "bin"),
                                             environment.get("PATH", "")]),
                       CUDA_HOME=str(args.cuda_home), TORCH_CUDA_ARCH_LIST="8.9",
                       CPATH=str(args.cuda_home / "targets/x86_64-linux/include") + os.pathsep
                       + environment.get("CPATH", ""),
                       LIBRARY_PATH=str(args.cuda_home / "targets/x86_64-linux/lib") + os.pathsep
                       + environment.get("LIBRARY_PATH", ""),
                       MAX_JOBS="8",
                       TORCH_EXTENSIONS_DIR=str(Path("/local/users/emanuele/moqsplat-reproduction-20261101") / "torch-extensions"))
    with (stage / "moqsplat-environment.txt").open("w") as stream:
        subprocess.run([str(args.python), "-m", "pip", "freeze", "--all"], env=environment,
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=30)
    command = [str(args.python), str(checkout / "scripts" / "validate_reproduction.py"),
               "--scene", str(args.scene), "--trace", str(args.trace), "--frames", str(args.frames),
               "--camera-convention", "opengl",
               "--device", "cuda:0", "--output", str(scratch / "publish")]
    started = time.monotonic()
    if os.environ["PS_STAGE"] == "smoke":
        subprocess.run([str(args.python), "-m", "pytest", "tests", "-q"], cwd=checkout,
                       env={**environment, "CUDA_VISIBLE_DEVICES": ""}, check=True,
                       timeout=180)
    returncode = 0
    try:
        if args.mode == "combined":
            oracle = scratch / "oracle"
            command[-1] = str(oracle)
            subprocess.run(command, cwd=checkout, env=environment, check=True, timeout=300)
            legacy = scratch / "legacy"
            subprocess.run([str(args.python), str(checkout / "scripts" / "validate_legacy_e2e.py"),
                            "--scene", str(args.scene), "--trace", str(args.trace), "--frames",
                            str(args.frames), "--device", "cuda:0", "--output", str(legacy)],
                           cwd=checkout, env=environment, check=True, timeout=300)
            native_run(checkout, scratch, args, environment)
            shutil.copytree(oracle, scratch / "publish" / "oracle")
            shutil.copytree(legacy, scratch / "publish" / "legacy")
            primary_path = scratch / "publish" / "summary.json"
            primary = json.loads(primary_path.read_text())
            primary.update(oracle=json.loads((oracle / "summary.json").read_text()),
                           legacy=json.loads((legacy / "summary.json").read_text()))
            primary_path.write_text(json.dumps(primary, indent=2))
        elif args.mode == "moq":
            native_run(checkout, scratch, args, environment)
        else:
            returncode = subprocess.run(command, cwd=checkout, env=environment, check=False).returncode
    except Exception:
        published = scratch / "publish"
        published.mkdir(exist_ok=True)
        for name in ("oracle", "legacy"):
            saved = scratch / name
            if saved.exists() and not (published / name).exists():
                shutil.copytree(saved, published / name)
        local_report = scratch / "publish" / "summary.json"
        if local_report.exists():
            report.write_bytes(local_report.read_bytes())
        raise
    local_report = scratch / "publish" / "summary.json"
    if local_report.exists():
        result = json.loads(local_report.read_text())
        result.update(source_archive_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      fleet_elapsed_seconds=time.monotonic() - started, command=command)
        report.write_text(json.dumps(result, indent=2))
    if returncode:
        raise SystemExit(returncode)
    print(json.dumps({"status": "complete", "summary": str(report)}))


if __name__ == "__main__":
    main()
