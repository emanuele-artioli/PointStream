"""Run a paired 24-frame reference comparison through the isolated native receiver."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tarfile
import time

from experiments.tennis import common_receiver_isolation as isolation
from experiments.tennis import common_receiver_job as receiver
from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import player_requalification as replay

MODULE = "experiments.tennis.reference_pair_validation"
QUANTIZERS = {
    "jpeg40-jpeg70": {
        "smoke": "d8e7d96cbf3dd3c812e245644ca15aeccc6553189fb3c5289ee78e783b5a0bfb",
        "full": "c4364a92fd5ebdcbce109f285098f1671268d714afe162d1fa8f47f1f45d940c",
    }
}
DIGESTS = {
    "smoke": "731cc34e1fcb7c0874cd5722ab63241a123d31fbcd699e199cee4163cdcbc0dc",
    "full": "6bfc32191f264e9dc98498808a20b2113e77d916ebda4ef16965934c6b871dba",
}


def select(a: argparse.Namespace) -> None:
    stage = os.environ["PS_STAGE"]
    a.frames = 24
    a.carrier_sha256 = QUANTIZERS.get(a.comparison, DIGESTS)[stage]
    a.carrier, a.manifest = (
        (a.control, a.control_manifest) if stage == "smoke" else (a.q90, a.q90_manifest)
    )
    if len(a.weight) != len(receiver.WEIGHT_PATHS) or len(set(a.deny)) < 4:
        raise ValueError("Incomplete resources or denied inputs")
    if replay.sha256(a.carrier) != a.carrier_sha256:
        raise ValueError("Changed reference comparison carrier")
    manifest.verify(a.manifest, a.carrier_sha256)


def run(a: argparse.Namespace) -> None:
    started = time.monotonic()
    select(a)
    stage = Path(os.environ["PS_STAGE_DIR"])
    receiver.bind_allocation(stage)
    output = Path(os.environ["PS_SCRATCH_DIR"]) / "reference-generation"
    fit = a.base / "publish/tennis-base@fit-100.pt"
    worker = {
        "carrier": str(a.carrier),
        "expected_sha256": a.carrier_sha256,
        "manifest": str(a.manifest),
        "weights": dict(zip(receiver.WEIGHT_PATHS, map(str, a.weight), strict=True)),
        "fit": str(fit),
        "frames": a.frames,
        "units": [a.unit],
    }
    isolation.launch(
        source_root=Path(__file__).resolve().parents[2],
        output=output,
        worker_config=worker,
        inputs=[a.carrier, a.manifest, *a.weight, fit],
        denied=a.deny,
        seconds=490 if os.environ["PS_STAGE"] == "smoke" else 690,
        kind="generation",
    )
    _, files = receiver.check_generated(a, output)
    archive = stage / "published.tar"
    temporary = stage / ".reference-pair.tar"
    with tarfile.open(temporary, "x") as tar:
        for path in files:
            tar.add(path, arcname="publish/" + path.name, recursive=False)
    if archive.exists():
        raise FileExistsError(archive)
    temporary.rename(archive)
    (stage / "worker_timing.json").write_text(
        json.dumps(
            dict(
                archive_sha256=replay.sha256(archive),
                worker_wall_seconds=time.monotonic() - started,
            )
        )
        + "\n"
    )


def validate(parser: argparse.ArgumentParser) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    dispatch = json.loads((stage / "dispatch.json").read_text())
    command = dispatch["command"]
    if command[1:4] != ["-m", MODULE, "run"]:
        raise ValueError("Wrong reference validation entry point")
    a = parser.parse_args(command[3:])
    select(a)
    timing = json.loads((stage / "worker_timing.json").read_text())
    archive = stage / "published.tar"
    if replay.sha256(archive) != timing["archive_sha256"]:
        raise ValueError("Reference comparison archive changed")
    with receiver.validation_workspace() as tmp:
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if [m.name for m in members] != [
                "publish/generation.json",
                "publish/" + a.unit + ".npz",
            ] or not all(m.isfile() and 0 < m.size < 96 * 2**20 for m in members):
                raise ValueError("Unexpected reference comparison archive")
            tar.extractall(tmp, filter="data")
        receipt, _ = receiver.check_generated(a, Path(tmp) / "publish")
    row = receipt["outputs"][0]
    remaining = dispatch["campaign_clock"]["budget_end"] - time.monotonic()
    wall = timing["worker_wall_seconds"]
    if os.environ["PS_STAGE"] == "smoke":
        forecast = 1.5 * wall + 60
        if not (0 < wall < 550 and forecast < min(720, remaining - 30)):
            raise ValueError(
                f"Paired render estimate {forecast:.1f} seconds exceeds the stage or remaining budget"
            )
    else:
        forecast = None
        if not 0 < wall < 720:
            raise ValueError("Full stage exceeded its allowance")
    checks = dict(
        matched_reference_arm=a.carrier_sha256,
        source_denied=True,
        frames=24,
        unit=a.unit,
        crop_sha256=row["crops_file_sha256"],
        seed=row["seed"],
        device=receipt["device"],
        gpu_uuid=receipt["gpu_uuid"],
        kernel_launches=row["kernels"]["kernel_launches"],
        wall_seconds=wall,
        remaining_seconds=remaining,
        full_forecast_seconds=forecast,
    )
    (stage / "result.json").write_text(json.dumps(checks, indent=2) + "\n")
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=True, checks=checks)) + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    run_command = sub.add_parser("run")
    for name in ("base", "control", "control-manifest", "q90", "q90-manifest"):
        run_command.add_argument("--" + name, required=True, type=Path)
    for name in ("weight", "deny"):
        run_command.add_argument("--" + name, required=True, type=Path, action="append")
    run_command.add_argument(
        "--comparison", choices=["lossless-q90", "jpeg40-jpeg70"], default="lossless-q90"
    )
    run_command.add_argument("--unit", required=True)
    run_command.add_argument("--pair-stage", type=int, required=True, choices=[1, 2])
    sub.add_parser("validate-stage")
    a = p.parse_args()
    if a.action == "run":
        if a.pair_stage != {"smoke": 1, "full": 2}[os.environ["PS_STAGE"]]:
            raise ValueError("Unexpected comparison arm")
        run(a)
    else:
        validate(p)


if __name__ == "__main__":
    main()
