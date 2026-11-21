"""Score Al028 native streams with the saved W7 source, mask and metric procedure."""
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np

from experiments.jobs.monitor import publish_progress
from experiments.tennis import w1_roi
from experiments.tennis.k4b_features import convert_command
from experiments.tennis.w1_score import SERIES, control_errors, qualified_gpu, score_point
from src.codecs import svtav1

FRAMES = 240
TICKS = list(range(0, FRAMES, 3))
SAVED = json.loads(Path("docs/experiments/w9-inputs/saved-trial-scoring/al028.json").read_text())
AUDIT = json.loads(Path("docs/experiments/starved-joint-20261122/al028-native-full-audit.json").read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_member(archive, name, destination):
    with tarfile.open(archive) as source:
        member = source.extractfile(name)
        if member is None:
            raise ValueError(f"Missing archive member {name}")
        with destination.open("xb") as output:
            shutil.copyfileobj(member, output)
    return destination


def run(args):
    import torch
    from experiments.visor.h3b import load_dists
    from src.codecs.quality import load_lpips, yuv420_to_rgb

    begun = time.monotonic()
    if not torch.cuda.is_available() or not qualified_gpu(torch.cuda.get_device_name(0), True):
        raise ValueError("A qualified CUDA GPU is required")
    torch.set_num_threads(4)
    limit = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "8000"))
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    stage = Path(os.environ["PS_STAGE_DIR"])
    spec = json.loads((stage.parent / "spec.json").read_text())
    inputs = {item["name"]: item for item in spec["staged_inputs"]}
    paths = {name: getattr(args, name) for name in inputs}
    if any(digest(path) != inputs[name]["sha256"] for name, path in paths.items()):
        raise ValueError("Input identity mismatch")
    tools = {name: svtav1.tool(name) for name in w1_roi.TOOLS}
    if any(tools[name]["sha256"] != value for name, value in w1_roi.TOOLS.items()):
        raise ValueError("Native tool mismatch")
    w1_roi.preload_windows(["--windows-file", str(args.windows), "--window-state", str(args.state)])
    window = w1_roi.WINDOWS["al028"]
    if window.source_sha != digest(args.source):
        raise ValueError("Incorrect window source")
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir()
    masks_file = extract_member(args.masks, "publish/masks.npz", scratch / "masks.npz")
    masks_meta = json.loads(extract_member(args.masks, "publish/masks.json", scratch / "masks.json").read_text())
    if digest(masks_file) != masks_meta["masks_sha256"]:
        raise ValueError("Mask identity mismatch")
    with np.load(masks_file, allow_pickle=False) as values:
        masks = {key: values[key] for key in ("ticks", "r", "s")}
    ticks = TICKS[:args.judged] if args.judged else TICKS
    if masks["ticks"].tolist()[:len(ticks)] != ticks:
        raise ValueError("Mask ticks differ from W7")
    hybrid_path = extract_member(args.hybrid, f"publish/{window.hybrid_dir}/scores.npz", scratch / "hybrid.npz")
    with np.load(hybrid_path, allow_pickle=False) as values:
        hybrid = {key: values[key] for key in SERIES}
        if values["ticks"].tolist()[:len(ticks)] != ticks:
            raise ValueError("Hybrid ticks differ from W7")
    control = next(row for row in SAVED["controls"] if row["scale"] == 4)
    control_path = extract_member(args.control, control["member"], scratch / "control.ivf")
    if digest(control_path) != control["stream_sha256"] or control_path.stat().st_size != control["bytes"]:
        raise ValueError("Historical control changed")
    points = [dict(name="control_s4_crf63", size=[480, 272], path=control_path,
                   decoded_sha256=control["decoded_sha256"], rate_kbps=control["rate_kbps"])]
    for row in AUDIT["streams"]:
        path = extract_member(args.native, f'publish/streams/plain-{row["crf"]}.ivf',
                              scratch / f'native-{row["crf"]}.ivf')
        if digest(path) != row["stream_sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Native stream changed")
        points.append(dict(name=f'plain_crf{row["crf"]}', size=[480, 272], path=path,
                           decoded_sha256=row["decoded_sha256"], rate_kbps=row["kbps"]))
    yuv = scratch / "source.yuv"
    subprocess.run(convert_command(tools["ffmpeg"]["path"], args.source, yuv, window.source_start, FRAMES),
                   check=True, capture_output=True, timeout=900)
    if digest(yuv) != window.source_yuv_sha:
        raise ValueError("Historical source conversion changed")
    raw = np.memmap(yuv, np.uint8, "r", shape=(FRAMES, 1620, 1920))
    truth = []
    for tick in ticks:
        plane = torch.from_numpy(np.array(raw[tick], copy=True)[None]).cuda()
        truth.append(yuv420_to_rgb(plane, 1920, 1080, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy())
    nets = {"lpips": load_lpips(args.lpips, "cuda"), "dists": load_dists(args.dists, "cuda")}
    if not all(next(net.parameters()).is_cuda for net in nets.values()):
        raise ValueError("Metric networks must run on CUDA")
    scores, elapsed = {}, {}
    for index, point in enumerate(points):
        result = score_point(point, scratch, ticks, truth, masks, nets, torch)
        scores[point["name"]] = result["values"]
        elapsed[point["name"]] = result["seconds"]
        if index == 0:
            errors = control_errors(result["values"], hybrid, 4, len(ticks))
            if max(errors.values()) > 1e-3:
                raise ValueError(f"Historical control does not reproduce: {errors}")
        np.savez_compressed(publish / "scores.npz", ticks=np.asarray(ticks), **scores)
        publish_progress(os.environ["PS_STAGE"], index + 1)
    torch.cuda.synchronize()
    summary = dict(ticks=ticks, points=[{k: v for k, v in row.items() if k != "path"} for row in points],
                   means={name: value.mean(axis=1).tolist() for name, value in scores.items()},
                   series=list(SERIES), control_errors=errors, score_seconds=elapsed,
                   seconds=time.monotonic() - begun, cuda=True, gpu=torch.cuda.get_device_name(0),
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   inputs={name: digest(path) for name, path in paths.items()},
                   tools=tools, source_yuv_sha256=window.source_yuv_sha)
    data = json.dumps(summary, indent=2) + "\n"
    (publish / "score.json").write_text(data)
    (stage / "score-summary.json").write_text(data)


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    summary = json.loads((stage / "score-summary.json").read_text())
    spec = json.loads((stage.parent / "spec.json").read_text())
    smoke = os.environ["PS_STAGE"] == "smoke"
    ticks = TICKS[:3] if smoke else TICKS
    names = ["control_s4_crf63", "plain_crf36", "plain_crf48", "plain_crf60"]
    checks = {
        "ticks": summary["ticks"] == ticks,
        "point_set": [row["name"] for row in summary["points"]] == names,
        "inputs": all(summary["inputs"][item["name"]] == item["sha256"] for item in spec["staged_inputs"]),
        "cuda": summary["cuda"] and qualified_gpu(summary["gpu"], True),
        "tools": all(summary["tools"][name]["sha256"] == value for name, value in w1_roi.TOOLS.items()),
        "control": set(summary["control_errors"]) == set(SERIES) and max(summary["control_errors"].values()) <= 1e-3,
        "memory": 0 < summary["peak_allocated_mib"] < 8000,
    }
    with tarfile.open(stage / "published.tar") as tar:
        member = tar.extractfile("publish/scores.npz")
        assert member is not None
        with np.load(io.BytesIO(member.read()), allow_pickle=False) as data:
            checks["scores"] = (sorted(data.files) == sorted(["ticks", *names])
                                and data["ticks"].tolist() == ticks
                                and all(data[name].shape == (4, len(ticks)) and np.isfinite(data[name]).all() for name in names))
    projection = None
    if smoke:
        seconds = sum(summary["score_seconds"].values())
        projection = 1.35 * (summary["seconds"] - seconds + seconds * len(TICKS) / len(ticks))
        checks["full_forecast"] = projection < spec["full"]["seconds"]
    result = dict(passed=all(checks.values()), checks=checks, projected_full_seconds=projection)
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("windows", "state", "source", "masks", "hybrid", "control", "lpips", "dists", "native"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--judged", type=int, choices=(0, 3))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate()
    elif args.judged is None or any(getattr(args, name) is None for name in
                                    ("windows", "state", "source", "masks", "hybrid", "control", "lpips", "dists", "native")):
        parser.error("All staged inputs and judged count are required")
    else:
        run(args)


if __name__ == "__main__":
    main()
