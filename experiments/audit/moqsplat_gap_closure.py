"""Qualify a frozen external MoQSplat client using the existing gated fleet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--publisher-camera", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--soak-seconds", type=int)
    parser.add_argument("--case-set", choices=("all", "direct", "impaired"), default="all")
    parser.add_argument("--audience", type=int, default=0)
    parser.add_argument("--qualified-smoke", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    stage = Path(os.environ["PS_STAGE_DIR"])
    if args.validate:
        result = json.loads((stage / "summary.json").read_text())
        if not result.get("valid") or result.get("transport") != "QUIC":
            raise ValueError("native client qualification did not pass")
        selected = result.get("case_set", "all")
        expected_cases = {"all": 3, "direct": 1, "impaired": 2}[selected]
        if len(result.get("cases", [])) != expected_cases:
            raise ValueError("selected network cases did not all pass")
        if selected != "all" and not result.get("prior_gate_verified"):
            raise ValueError("focused follow-up requires the qualified prior smoke")
        if selected == "all" and result.get("stage") == "smoke" and not result["semantic"]["valid"]:
            raise ValueError("semantic preparation did not pass")
        for case in result["cases"]:
            for client in case["clients"]:
                frame = client["last_frame"]
                if not client["valid"] or client["nonblack_frames"] < 1:
                    raise ValueError("client did not produce a nonblack CUDA frame")
                if not 0 < frame["cuda_peak_allocated_bytes"] <= frame["gpu_budget_bytes"]:
                    raise ValueError("final measured CUDA allocation does not meet its budget")
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(
            json.dumps({"passed": True, "checks": result})
        )
        print(json.dumps({"passed": True, "frames": result["frames_rendered"]}))
        return
    assert args.source and args.scene and args.trace and args.python
    assert args.weights and args.publisher_camera and args.soak_seconds
    if hashlib.sha256(args.weights.read_bytes()).hexdigest() != (
        "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
    ):
        raise ValueError("publisher weights do not match the protocol identity")
    prior_verified = None
    if args.case_set != "all":
        if not args.qualified_smoke:
            raise ValueError("focused follow-up requires qualified prior evidence")
        prior = json.loads(args.qualified_smoke.read_text())
        if not (
            prior.get("valid")
            and prior["semantic"]["valid"]
            and len(prior["cases"]) == 3
            and prior.get("source_archive_sha256")
            == "75f399794df40944c6d905023c323ec0826b0e50766852c11fb82fe364c6399c"
        ):
            raise ValueError("prior smoke is not the qualified dedicated-worker source")
        prior_verified = hashlib.sha256(args.qualified_smoke.read_bytes()).hexdigest()
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    checkout = scratch / "source"
    checkout.mkdir()
    with tarfile.open(args.source) as archive:
        archive.extractall(checkout, filter="data")
    cache = Path("/local/users/emanuele/moqsplat-gap-closure-20261101/cache")
    toolkit = Path("/usr/local/miniconda3/envs/cuda")
    temporary = scratch / "tmp"
    temporary.mkdir()
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(checkout / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        PATH=os.pathsep.join(
            [str(args.python.parent), str(toolkit / "bin"), environment.get("PATH", "")]
        ),
        CUDA_HOME=str(toolkit),
        TORCH_CUDA_ARCH_LIST="8.9",
        MAX_JOBS="8",
        CPATH=str(toolkit / "targets/x86_64-linux/include"),
        LIBRARY_PATH=str(toolkit / "targets/x86_64-linux/lib"),
        TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"),
        TORCH_HOME=str(cache / "torch"),
        XDG_CACHE_HOME=str(cache / "xdg"),
        YOLO_CONFIG_DIR=str(cache / "yolo"),
        MPLCONFIGDIR=str(cache / "matplotlib"),
        TMPDIR=str(temporary),
        TMP=str(temporary),
        TEMP=str(temporary),
    )
    with (stage / "environment.txt").open("w") as stream:
        subprocess.run(
            [str(args.python), "-m", "pip", "freeze", "--all"],
            env=environment,
            stdout=stream,
            check=True,
            timeout=30,
        )
    if os.environ["PS_STAGE"] == "smoke":
        subprocess.run(
            [str(args.python), "-m", "pytest", "tests", "-q"],
            cwd=checkout,
            env={**environment, "CUDA_VISIBLE_DEVICES": ""},
            check=True,
            timeout=180,
        )
    output = scratch / "publish"
    command = [
        str(args.python),
        str(checkout / "scripts/validate_client_experience.py"),
        "--scene",
        str(args.scene),
        "--trace",
        str(args.trace),
        "--weights",
        str(args.weights),
        "--publisher-camera",
        str(args.publisher_camera),
        "--output",
        str(output),
        "--stage",
        os.environ["PS_STAGE"],
        "--soak-seconds",
        str(args.soak_seconds),
        "--case-set",
        args.case_set,
        "--audience",
        str(args.audience),
    ]
    try:
        subprocess.run(command, cwd=checkout, env=environment, check=True)
    finally:
        summary = output / "summary.json"
        if summary.exists():
            result = json.loads(summary.read_text())
            result["prior_gate_verified"] = prior_verified
            result["source_archive_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
            result["command"] = command
            (stage / "summary.json").write_text(json.dumps(result, indent=2))
        shutil.copyfile(checkout / "pyproject.toml", stage / "external-pyproject.toml")


if __name__ == "__main__":
    main()
