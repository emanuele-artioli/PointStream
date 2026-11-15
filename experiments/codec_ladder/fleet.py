"""Fleet CPU run/GPU score adapter; no dispatch or host/resource selection here.

CPU: --mode run --manifest JSON --source RAW --masks NPZ --frames N
     --target-frames FULL --full-cap-seconds CAP --publication-overhead-seconds S
GPU: --mode score --archive CPU_PUBLISHED.tar --archive-sha256 SHA --source RAW
     --lpips-backbone WEIGHT --dists-backbone WEIGHT [same cost arguments]
Validator: --validate (uses PS_STAGE_DIR/PS_SCRATCH_DIR, substantive pixel replay).

Both smoke and full retain their distinct stage published.tar. CPU archive
contains packets once, hashes, metadata, masks and receipts; no decoded raw
video, source raw video or duplicate native stream files are published.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import re
import math
import tarfile
import time
from pathlib import Path
from typing import Any

from experiments.codec_ladder import __main__ as worker
from experiments.codec_ladder import transport
from src.codecs import svtav1


def qualify_tools(encoder: str, decoder: str) -> dict[str, Any]:
    tools = {"encoder": svtav1.tool(encoder), "decoder": svtav1.tool(decoder)}
    if re.search(r"(?<![0-9])4\.2\.0\b", tools["encoder"]["version"]) is None:
        raise ValueError("production screen requires qualified SVT-AV1 4.2.0; other versions are infrastructure only")
    if re.search(r"\b1\.5(?:\.\d+)?\b", tools["decoder"]["version"]) is None:
        raise ValueError("production screen requires qualified dav1d 1.5.x")
    return tools


def cpu_seconds() -> float:
    values = [resource.getrusage(kind) for kind in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(value.ru_utime + value.ru_stime for value in values)


def archive_run(root: Path, output: Path) -> dict[str, Any]:
    """Only independently decodable packets and small support/provenance artifacts."""
    record = json.loads((root / "result.json").read_text())
    names = ["result.json", "input.json", "foreground.npz", "roi.txt"]
    names += [row["packet"] for row in record["rates"]]
    names += [row["decoded"].replace(".yuv", ".receipt.json") for row in record["rates"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate publication members")
    with tarfile.open(output, "x") as bundle:
        for name in sorted(names):
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise ValueError("invalid publication artifact")
            bundle.add(path, arcname="publish/" + name, recursive=False)
    return {"sha256": worker.file_sha(output), "bytes": output.stat().st_size,
            "members": {name: worker.file_sha(root / name) for name in sorted(names)}}


def extract_archive(archive: Path, expected: str, output: Path) -> Path:
    if worker.file_sha(archive) != expected:
        raise ValueError("CPU archive identity mismatch")
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if any(not member.isfile() or not member.name.startswith("publish/") or ".." in Path(member.name).parts for member in members):
            raise ValueError("unexpected archive member")
        if len({member.name for member in members}) != len(members):
            raise ValueError("duplicate archive members")
        bundle.extractall(output, filter="data")
    return output / "publish"


def prepared_inputs(archive: Path, expected: str, output: Path, config: dict[str, Any] | None = None) -> tuple[Path, Path, Path | None, dict[str, Any]]:
    """Materialize only authenticated prepared payloads on host-local scratch."""
    if worker.file_sha(archive) != expected:
        raise ValueError("prepared archive identity mismatch")
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if len(members) != 3 or {m.name for m in members} != {"source.yuv", "masks.npz", "preparation.json"} or any(not m.isfile() for m in members):
            raise ValueError("unexpected prepared archive members")
        bundle.extractall(output, filter="data")
    info = json.loads((output / "preparation.json").read_text())
    if set(info["outputs"]) != {"source.yuv", "masks.npz"}:
        raise ValueError("invalid preparation payload inventory")
    for name, item in info["outputs"].items():
        path = output / name
        if path.stat().st_size != item["bytes"] or worker.file_sha(path) != item["sha256"]:
            raise ValueError("prepared payload identity mismatch")
    source, masks, manifest = output / "source.yuv", output / "masks.npz", None
    if config is not None:
        expected_fps = worker.Fraction(info["fps"])
        for key, value in (("width", info["width"]), ("height", info["height"]), ("frames", info["frames"]),
                           ("fps", [expected_fps.numerator, expected_fps.denominator]),
                           ("full_range", info["source_record"]["full_range"]), ("source_group", info["source_group"]),
                           ("annotation_provenance", info["mask_record"]["provenance"])):
            if config[key] != value:
                raise ValueError("frozen configuration differs from actual preparation")
        manifest = output / "manifest.json"
        worker.prepare(config, source, masks, manifest)
    return source, masks, manifest, {"archive_sha256": expected, "preparation": info}


def restore_decoded(root: Path, decoder: str) -> dict[str, bool]:
    record = json.loads((root / "result.json").read_text())
    checks = {}
    for row in record["rates"]:
        packet, output = root / row["packet"], root / row["decoded"]
        if not output.exists():
            receipt = transport.decode(packet.read_bytes(), output, scratch=root, decoder=decoder)
        else:
            receipt = {"decoded_sha256": worker.file_sha(output)}
        checks[f'{row["arm"]}-{row["crf"]}'] = receipt["decoded_sha256"] == row["receipt"]["decoded_sha256"]
    if not all(checks.values()):
        raise ValueError("fresh CPU receiver pixels differ from archived receipt")
    return checks


def cost_record(*, wall: float, cpu: float, frames: int, target: int, cap: float, overhead: float) -> dict[str, Any]:
    if min(wall, cpu, cap) <= 0 or min(frames, target) < 1 or target < frames or overhead < 0:
        raise ValueError("invalid cost bound")
    forecast = 1.5 * wall * target / frames + overhead
    # Same-source measured smoke path; costs remain conservative and no fitted
    # intercept is subtracted. A full stage is already the measured target.
    accepted = wall <= cap if frames == target else forecast <= cap
    return {"command_wall_seconds": wall, "command_cpu_seconds": cpu, "frames": frames, "target_frames": target,
            "forecast_seconds": forecast, "forecast_basis": "1.5 * complete measured smoke wall * target/smoke + publication overhead",
            "full_cap_seconds": cap, "publication_overhead_seconds": overhead, "passed": accepted,
            "scope": "command cost feasibility only; no encoder/decoder speed claim"}


def validate_scores(scores: dict[str, Any]) -> dict[str, bool]:
    count = scores["frames"]
    control = scores["metric_control"]
    gpu = scores["gpu_execution"]
    checks = {"metric_self_control": all(type(control[k]) in (int, float) and math.isfinite(control[k]) and abs(control[k]) <= 1e-5 for k in ("source_self_lpips", "source_self_dists")),
              "cuda_networks": gpu["device_count"] == 1 and gpu["network_parameters_cuda"] is True,
              "reserved_memory_within_declared8000MiB": 0 < gpu["maximum_reserved_bytes"] <= 8000 * 1024**2,
              "all_rows": len(scores["rows"]) == 12,
              "matrix": {(row["arm"], row["crf"]) for row in scores["rows"]} == {(arm, crf) for arm in ("native", "roi", "semantic_patch", "error_patch") for crf in scores["config"]["rates"]},
              "coverage": all(len(scores["region_pixels"][name]) == count for name in ("foreground", "background"))}
    for row in scores["rows"]:
        key = f'{row["arm"]}-{row["crf"]}'
        values = row["frames"]
        checks[key + ":frame_count"] = all(len(array) == count for array in values.values())
        checks[key + ":required_finite"] = all(type(value) in (int, float) and math.isfinite(value)
            for name in ("full_lpips", "weighted_lpips", "full_dists", "weighted_dists", "full_rgb_psnr") for value in values[name])
        checks[key + ":temporal_pairs"] = row["temporal"]["pairs"] == count - 1 and all(len(array) == count - 1 for array in row["temporal"]["values"].values())
        checks[key + ":regional_coverage"] = all((value is None) == (pixels == 0)
            for metric in ("lpips", "dists") for name in ("foreground", "background")
            for value, pixels in zip(values[name + "_" + metric], scores["region_pixels"][name]))
        checks[key + ":regional_finite"] = all(value is None or type(value) in (int, float) and math.isfinite(value) and value >= 0
            for metric in ("lpips", "dists") for name in ("foreground", "background") for value in values[name + "_" + metric])
        for metric in ("lpips", "dists"):
            expected = [worker.evaluation.weighted_region(worker.evaluation.RegionalFrame(fg, bg, fp, bp))
                for fg, bg, fp, bp in zip(values["foreground_" + metric], values["background_" + metric], scores["region_pixels"]["foreground"], scores["region_pixels"]["background"])]
            checks[key + ":weighted_" + metric] = all(math.isclose(actual, reference, abs_tol=1e-12, rel_tol=1e-10)
                for actual, reference in zip(values["weighted_" + metric], expected))
            for name in ("full_", "weighted_"):
                array = values[name + metric]
                checks[key + ":mean_" + name + metric] = math.isclose(row["means"][name + metric], sum(array) / count, abs_tol=1e-12, rel_tol=1e-10)
    return checks


def validate(stage: Path, scratch: Path, *, decoder: str = "dav1d") -> dict[str, Any]:
    record = json.loads((stage / "ladder.json").read_text())
    archive = stage / "published.tar"
    checks: dict[str, bool] = {"archive_sha256": worker.file_sha(archive) == record["archive"]["sha256"], "cost_gate": record["cost"]["passed"]}
    if record["mode"] == "run":
        root = extract_archive(archive, record["archive"]["sha256"], scratch / "validation-replay")
        checks["members"] = all(worker.file_sha(root / name) == digest for name, digest in record["archive"]["members"].items())
        checks.update({"fresh_receiver:" + key: passed for key, passed in restore_decoded(root, decoder).items()})
        physical = worker.validate(root)
        checks.update(physical["checks"])
    else:
        with tarfile.open(archive) as bundle:
            if bundle.getnames() != ["publish/scores.json"]:
                raise ValueError("unexpected score publication")
            handle = bundle.extractfile("publish/scores.json")
            if handle is None:
                raise ValueError("missing score file")
            scores = json.load(handle)
        checks.update(validate_scores(scores))
        checks["fresh_receiver_replay"] = len(record["receiver_pixel_replay"]) == 12 and all(record["receiver_pixel_replay"].values())
    return {"passed": all(checks.values()), "checks": checks, "scope": "physical/metric-integrity and cost; not a scientific promotion verdict"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--mode", choices=["run", "score"], default="run")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--lpips-backbone", type=Path)
    parser.add_argument("--dists-backbone", type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--target-frames", type=int)
    parser.add_argument("--full-cap-seconds", type=float)
    parser.add_argument("--publication-overhead-seconds", type=float, default=30)
    parser.add_argument("--decoder", default="dav1d")
    parser.add_argument("--encoder", default="SvtAv1EncApp")
    args = parser.parse_args()
    stage, scratch = Path(os.environ["PS_STAGE_DIR"]), Path(os.environ["PS_SCRATCH_DIR"])
    if args.validate:
        result = validate(stage, scratch, decoder=args.decoder)
        worker.write_json(Path(os.environ["PS_VALIDATION_PATH"]), result)
        print(json.dumps(result))
        return 0 if result["passed"] else 1
    start, cpu = time.monotonic(), cpu_seconds()
    tools = qualify_tools(args.encoder, args.decoder)
    if args.target_frames is None or args.full_cap_seconds is None:
        parser.error("full target frame count and full-stage cap are required")
    replay: dict[str, bool] = {}
    preparation_binding = None
    if args.prepared is not None:
        if args.prepared_sha256 is None or (args.mode == "run" and args.config is None):
            parser.error("prepared input requires its SHA256 and run configuration")
        config = json.loads(args.config.read_text()) if args.config is not None else None
        args.source, args.masks, args.manifest, preparation_binding = prepared_inputs(
            args.prepared, args.prepared_sha256, scratch / "prepared-input", config)
    if args.mode == "run":
        if args.manifest is None or args.source is None or args.masks is None or args.frames is None:
            parser.error("run requires manifest/source/masks/frames")
        root = scratch / "native-run"
        result = worker.run(args.manifest, root, prefix=args.frames, source=args.source, masks=args.masks, encoder=args.encoder, decoder=args.decoder)
        physical = worker.validate(root)
        if not physical["passed"]:
            raise ValueError("physical or effective-ROI gate failed")
        publication = archive_run(root, stage / "published.tar")
        count = result["selected_prefix_frames"]
    else:
        if any(value is None for value in (args.archive, args.archive_sha256, args.source, args.lpips_backbone, args.dists_backbone)):
            parser.error("score requires archive/hash/source/backbones")
        root = extract_archive(args.archive, args.archive_sha256, scratch / "cpu-input")
        replay = restore_decoded(root, args.decoder)
        result = worker.score(root, args.lpips_backbone, args.dists_backbone, root / "scores.json", source=args.source, prefix=args.frames)
        result["receiver_pixel_replay"] = replay
        # score() wrote original immutable scores; retain replay separately in
        # thin stage record rather than mutate its saved scientific output.
        with tarfile.open(stage / "published.tar", "x") as bundle:
            bundle.add(root / "scores.json", arcname="publish/scores.json", recursive=False)
        publication = {"sha256": worker.file_sha(stage / "published.tar"), "bytes": (stage / "published.tar").stat().st_size}
        count = result["frames"]
    cost = cost_record(wall=time.monotonic() - start, cpu=cpu_seconds() - cpu, frames=count, target=args.target_frames,
                       cap=args.full_cap_seconds, overhead=args.publication_overhead_seconds)
    worker.write_json(stage / "ladder.json", {"mode": args.mode, "archive": publication, "cost": cost, "tools": tools, "receiver_pixel_replay": replay, "preparation_binding": preparation_binding})
    print(json.dumps({"mode": args.mode, "archive_bytes": publication["bytes"], "cost": cost}))
    return 0 if cost["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
