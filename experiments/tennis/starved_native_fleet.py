"""Fleet adapter and substantive validator for the W7 plain AV1 control."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from src.codecs import svtav1

WORKER = Path("docs/experiments/starved-joint-20261122/native_control.py")
SOURCE_SHA = "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934"
TOOL_SHA = {
    "SvtAv1EncApp": "4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4",
    "dav1d": "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6",
    "ffmpeg": "7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2",
}


def load_worker():
    spec = importlib.util.spec_from_file_location("starved_native_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args):
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Single-CPU, GPU-disabled fleet isolation is required")
    child = subprocess.check_output(
        [sys.executable, "-c", "import os; print(sorted(os.sched_getaffinity(0)))"],
        text=True, timeout=10,
    ).strip()
    if child != repr(cpus):
        raise RuntimeError("Child process did not inherit the CPU affinity")
    observed = {name: svtav1.tool(name) for name in TOOL_SHA}
    if any(observed[name]["sha256"] != expected for name, expected in TOOL_SHA.items()):
        raise ValueError("Native encoder, decoder or ffmpeg differs from pinned W7 tools")
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    stage = Path(os.environ["PS_STAGE_DIR"])
    work, publish = scratch / "native", scratch / "publish"
    crfs = ["48"] if args.frames == 24 else ["36", "48", "60"]
    command = [sys.executable, str(WORKER), "--source", str(args.source),
               "--output", str(work), "--frames", str(args.frames),
               "--scale", str(args.scale), "--crfs", *crfs]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path.cwd()), env.get("PYTHONPATH", "")]))
    subprocess.run(command, check=True, timeout=850, env=env)
    report = json.loads((work / "report.json").read_text())
    report["cpu_affinity"] = cpus
    report["stage"] = os.environ["PS_STAGE"]
    publish.mkdir()
    (publish / "streams").mkdir()
    for row in report["rows"]:
        src = Path(row["stream"])
        target = publish / "streams" / f'plain-{row["crf"]}.ivf'
        shutil.copyfile(src, target)
        row["published_stream"] = str(target.relative_to(publish))
    summary = json.dumps(report, indent=2) + "\n"
    (publish / "report.json").write_text(summary)
    (stage / "native-report.json").write_text(summary)


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    report = json.loads((stage / "native-report.json").read_text())
    spec = json.loads((stage.parent / "spec.json").read_text())
    worker = load_worker()
    n = 24 if os.environ["PS_STAGE"] == "smoke" else 240
    crfs = [48] if n == 24 else [36, 48, 60]
    checks = {
        "source": report["source_sha256"] == SOURCE_SHA,
        "frozen_worker": report["script_sha256"] == worker.sha(WORKER),
        "staged_source": any(item["name"] == "source" and item["sha256"] == SOURCE_SHA
                             for item in spec["staged_inputs"]),
        "frames_cadence": report["window_frames"] == n and report["source_fps"] == 25,
        "window_geometry": report["window_start"] == 0 and report["converted_dimensions"] == [480, 272],
        "cpu_isolation": len(report["cpu_affinity"]) == 1,
        "tools": all(report["tools"][name]["sha256"] == expected for name, expected in TOOL_SHA.items()),
        "crfs": [row["crf"] for row in report["rows"]] == crfs,
        "stage": report["stage"] == os.environ["PS_STAGE"],
        "rate_support": report["target_brackets"] == worker.brackets(
            [row["complete_kbps"] for row in report["rows"]]),
    }
    stream_ok = True
    with tempfile.TemporaryDirectory(prefix="starved-validator-", dir=os.environ["TMPDIR"]) as temp, \
            tarfile.open(stage / "published.tar") as tar:
        for row in report["rows"]:
            member = tar.extractfile("publish/" + row["published_stream"])
            if member is None:
                raise ValueError("Missing published IVF")
            data = member.read()
            stream_ok &= hashlib.sha256(data).hexdigest() == row["stream_sha256"]
            stream_ok &= len(data) == row["file_bytes"] == row["payload_bytes"] + 32 + n * 12
            stream_ok &= abs(row["complete_kbps"] - len(data) * 8 * 25 / n / 1000) < 1e-9
            out = Path(temp) / f'validate-{row["crf"]}.ivf'
            out.write_bytes(data)
            stream_ok &= worker.verify_ivf(out, 480, 272, n) == row["payload_bytes"]
            decoded = out.with_suffix(".yuv")
            subprocess.run(svtav1.decode_command(report["tools"]["dav1d"]["path"], out, decoded,
                                                threads=1), check=True, timeout=90)
            stream_ok &= decoded.stat().st_size == n * 480 * 272 * 3 // 2
            stream_ok &= worker.sha(decoded) == row["decoded_sha256"]
    checks["independently_decoded_streams"] = stream_ok
    projection = None
    if n == 24:
        projection = report["elapsed_seconds"] * (240 / 24) * 3 * 1.5 + 120
        checks["full_forecast_fits"] = projection <= spec["full"]["seconds"]
    result = {"passed": all(checks.values()), "checks": checks,
              "projected_full_seconds": projection}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--frames", type=int, choices=(24, 240))
    parser.add_argument("--scale", type=int, choices=(4,))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate()
    elif args.source is not None and args.frames is not None and args.scale is not None:
        run(args)
    else:
        parser.error("source, frames and scale are required")


if __name__ == "__main__":
    main()
