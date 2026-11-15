"""Bounded native ROI/patch screen; no scientific success inferred from correctness.

prepare --config CONFIG.json --source RAW.yuv --masks MASKS.npz --out NEW.json
run --manifest MANIFEST --out NEW_DIR [--frames PREFIX] [--source STAGED --masks STAGED]
validate --root RESULTS [--out REPORT]
decode --packet FILE --out NEW.yuv --scratch HOST_LOCAL_DIR
score --root RESULTS --lpips-backbone WEIGHT --dists-backbone WEIGHT --out NEW.json

Configuration: domain, source_group, annotation_provenance, width, height,
fps [numerator,denominator], frames, full_range, rates (three CRFs), preset,
patch_size [width,height], patch_crf, roi_outside_offset. Masks NPZ contains
boolean `foreground` with shape (frames,height,width). Native raw input is
planar 8-bit YUV420 at declared BT709/range; preparation never converts pixels.
All output directories/files are fresh. No model download or GPU dispatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.codec_ladder import transport
from experiments.tennis.w1_roi import roi_text
from src.codecs import quality, svtav1
from src.pointstream import codec, evaluation

CONFIG = {"domain", "source_group", "annotation_provenance", "width", "height", "fps", "frames", "full_range",
          "rates", "preset", "patch_size", "patch_crf", "roi_outside_offset"}


LEGACY_PROTOCOL = "all-frame70/30; empty-region weight reallocated; regional component means report available-frame subsets"
METRIC_PROTOCOL = "codec-ladder-metrics-v2"
STANDARD_LPIPS = "full_lpips_standard"
METRIC_DEFINITIONS = {
    "full_lpips": "mean of full-resolution bilinearly upsampled LPIPS-Alex spatial map; not standard scalar LPIPS",
    STANDARD_LPIPS: "standard LPIPS-Alex scalar: sum of feature-layer spatial means (spatial=False), same weights and preprocessing",
    "weighted_lpips": "all-frame70/30 source-mask regional means of full-resolution LPIPS-Alex spatial map; empty-region weight reallocated",
    "full_dists": "standard full-frame DISTS with all-ones feature-statistic weights",
    "weighted_dists": "all-frame70/30 source-mask regional DISTS feature statistics; empty-region weight reallocated",
}


def metric_profile(score: dict[str, Any]) -> bool:
    """Identify an explicit historical profile or require the complete v2 contract."""
    if score.get("metric_protocol") == METRIC_PROTOCOL:
        if score.get("metric_definitions") != METRIC_DEFINITIONS:
            raise ValueError("metric definitions differ from registered v2 profile")
        return True
    if "metric_protocol" not in score and score.get("protocol") == LEGACY_PROTOCOL:
        if any(STANDARD_LPIPS in row.get("frames", {}) or STANDARD_LPIPS in row.get("means", {}) for row in score["rows"]):
            raise ValueError("standard LPIPS requires explicit v2 metric profile")
        return False
    raise ValueError("unknown or unidentified metric profile")


def standard_lpips_scalar(network: Any, x: Any, y: Any) -> float:
    """Reuse exact loaded weights; caller holds inference_mode; restore on failure.

    One additional sequential AlexNet forward per image pair, no extra model.
    """
    spatial = network.spatial
    try:
        network.spatial = False
        value = float(network(x, y).item())
    finally:
        network.spatial = spatial
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid standard full-frame LPIPS")
    return value


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    codec.write_new(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def native_meta(config: dict[str, Any], decoder_sha: str, frames: int | None = None) -> dict[str, Any]:
    return {"version": 1, "profile": "native-av1", **{k: config[k] for k in ("width", "height", "fps", "full_range")},
            "frames": config["frames"] if frames is None else frames, "decoder_sha256": decoder_sha}


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or set(config) != CONFIG:
        raise ValueError("configuration fields differ from frozen screen contract")
    codec.validate(native_meta(config, "0" * 64))
    if config["domain"] not in ("tennis", "egocentric") or not all(isinstance(config[k], str) and config[k] for k in ("source_group", "annotation_provenance")):
        raise ValueError("invalid domain/source/provenance")
    for key in ("patch_crf", "roi_outside_offset", "preset"):
        if type(config[key]) is not int:
            raise ValueError("integer encoder settings required")
    if not 0 <= config["patch_crf"] <= 70 or not 1 <= config["roi_outside_offset"] <= 63 or not 0 <= config["preset"] <= 13:
        raise ValueError("invalid encoder setting")
    if not isinstance(config["rates"], list) or len(config["rates"]) != 3 or len(set(config["rates"])) != 3 or any(type(v) is not int or not 0 <= v <= 70 for v in config["rates"]):
        raise ValueError("exactly three distinct CRFs required")
    size = config["patch_size"]
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 64 or v % 2 for v in size) or size[0] > config["width"] or size[1] > config["height"]:
        raise ValueError("invalid even patch size")


def load_masks(path: Path, config: dict[str, Any]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"foreground"}:
            raise ValueError("mask archive must contain only foreground")
        masks = data["foreground"]
    if masks.dtype != bool or masks.shape != (config["frames"], config["height"], config["width"]):
        raise ValueError("source-resolution all-frame boolean masks required")
    return masks


def prepare(config: dict[str, Any], source: Path, masks: Path, output: Path) -> dict[str, Any]:
    validate_config(config)
    if source.stat().st_size != config["width"] * config["height"] * 3 // 2 * config["frames"]:
        raise ValueError("raw source length differs from declared video")
    regions = load_masks(masks, config)
    record = {"schema": 1, "config": config,
              "source": {"path": str(source.resolve()), "sha256": file_sha(source)},
              "masks": {"path": str(masks.resolve()), "sha256": file_sha(masks)},
              "selection": "development mechanism screen; no untouched-footage claim",
              "mask_coverage": {"foreground_pixels": int(regions.sum()), "empty_frames": int(np.count_nonzero(~regions.any(axis=(1, 2))))}}
    write_json(output, record)
    return record


def select_box(importance: np.ndarray, size: list[int]) -> list[int]:
    """Maximize summed source importance over one fixed even rectangle; first tie wins."""
    if importance.ndim != 2 or not np.isfinite(importance).all() or (importance < 0).any():
        raise ValueError("finite nonnegative selection map required")
    width, height = size
    if width > importance.shape[1] or height > importance.shape[0] or min(size) < 64 or any(v % 2 for v in size):
        raise ValueError("unsupported patch dimensions")
    integral = np.pad(importance.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = integral[height:, width:] - integral[:-height, width:] - integral[height:, :-width] + integral[:-height, :-width]
    yy, xx = np.unravel_index(np.argmax(sums[::2, ::2]), sums[::2, ::2].shape)
    return [int(xx * 2), int(yy * 2), width, height]


def raw_frames(path: Path, config: dict[str, Any], frames: int) -> np.memmap:
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(frames, config["height"] * 3 // 2, config["width"]))


def encode_native(source: Path, root: Path, config: dict[str, Any], meta: dict[str, Any], crf: int,
                  encoder: str, roi: Path | None = None) -> dict[str, Any]:
    root.mkdir()
    stream = root / "stream.ivf"
    extra = ["--color-primaries", "1", "--transfer-characteristics", "1", "--matrix-coefficients", "1"]
    if roi is not None:
        extra += ["--roi-map-file", str(roi)]
    command = svtav1.encode_command(encoder, source, stream, width=meta["width"], height=meta["height"],
                                  fps=Fraction(*meta["fps"]), frames=meta["frames"], crf=crf,
                                  preset=config["preset"], full_range=meta["full_range"], extra=extra)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    timing = svtav1.run(command, 600, cpus)
    payload = stream.read_bytes()
    codec.check_native(meta, payload)
    return {"stream": str(stream), "sha256": transport.digest(payload), "bytes": len(payload), "command": command, "timing": timing}


def decode_packet(packet: Path, output: Path, scratch: Path, decoder: str) -> dict[str, Any]:
    receipt = output.with_suffix(".receipt.json")
    command = [sys.executable, "-m", "experiments.codec_ladder", "decode", "--packet", str(packet), "--out", str(output),
               "--scratch", str(scratch), "--decoder", decoder, "--receipt", str(receipt)]
    timing = svtav1.run(command, 600)
    return {**json.loads(receipt.read_text()), "receiver_command": command, "receiver_process_timing": timing}


def run(manifest: Path, output: Path, *, prefix: int | None = None, source: Path | None = None,
        masks: Path | None = None, encoder: str = "SvtAv1EncApp", decoder: str = "dav1d") -> dict[str, Any]:
    started = time.monotonic()
    record = json.loads(manifest.read_text())
    config = record["config"]
    validate_config(config)
    source = source or Path(record["source"]["path"])
    masks = masks or Path(record["masks"]["path"])
    if file_sha(source) != record["source"]["sha256"] or file_sha(masks) != record["masks"]["sha256"]:
        raise ValueError("source/mask identity mismatch")
    count = config["frames"] if prefix is None else prefix
    if type(count) is not int or not 1 <= count <= config["frames"]:
        raise ValueError("invalid frozen prefix")
    regions = load_masks(masks, config)[:count]
    encoder_id, decoder_id = svtav1.tool(encoder), svtav1.tool(decoder)
    meta = native_meta(config, decoder_id["sha256"], count)
    frames = raw_frames(source, config, count)
    output.mkdir(parents=True, exist_ok=False)
    # Preserve immutable exact input/config manifest and all-frame source mask.
    write_json(output / "input.json", record)
    np.savez_compressed(output / "foreground.npz", foreground=regions)
    width, height = config["width"], config["height"]
    roi = output / "roi.txt"
    roi.write_text(roi_text(regions, width, height, -config["roi_outside_offset"], outside=True))
    rows, native_rows = [], {}
    for crf in config["rates"]:
        native_rows[crf] = encode_native(source, output / f"native-{crf}", config, meta, crf, encoder)
        for arm, native in (("native", native_rows[crf]), ("roi", encode_native(source, output / f"roi-{crf}", config, meta, crf, encoder, roi))):
            packet = output / f"{arm}-{crf}.plp"
            packet.write_bytes(transport.pack({"profile": "static-pixel-patch-v1", "native": meta, "box": None}, Path(native["stream"]).read_bytes()))
            decoded = output / f"{arm}-{crf}.yuv"
            receipt = decode_packet(packet, decoded, output, decoder)
            rows.append({"arm": arm, "crf": crf, "packet": packet.name, "decoded": decoded.name, "native": native,
                         "receipt": receipt, "box": None, "bound": None})
    reference_crf = max(config["rates"])
    base_pixels = raw_frames(output / f"native-{reference_crf}.yuv", config, count)
    error_map = np.zeros((height, width), np.float64)
    for original, baseline_frame in zip(frames, base_pixels):
        error_map += np.abs(original[:height].astype(np.float64) - baseline_frame[:height].astype(np.float64))
    selectors = {"semantic_patch": regions.sum(axis=0), "error_patch": error_map}
    for arm, importance in selectors.items():
        box = select_box(importance, config["patch_size"])
        crop_path = output / f"{arm}.yuv"
        with crop_path.open("xb") as target:
            for frame in frames:
                target.write(transport.crop(frame, width, height, box))
        patch = encode_native(crop_path, output / arm, config, {**meta, "width": box[2], "height": box[3]}, config["patch_crf"], encoder)
        patch_data = Path(patch["stream"]).read_bytes()
        for crf in config["rates"]:
            base = Path(native_rows[crf]["stream"]).read_bytes()
            data = transport.pack({"profile": "static-pixel-patch-v1", "native": meta, "box": box}, base, patch_data)
            packet, decoded = output / f"{arm}-{crf}.plp", output / f"{arm}-{crf}.decoded.yuv"
            packet.write_bytes(data)
            receipt = decode_packet(packet, decoded, output, decoder)
            rows.append({"arm": arm, "crf": crf, "packet": packet.name, "decoded": decoded.name, "native": native_rows[crf],
                         "patch": patch, "receipt": receipt, "box": box,
                         "bound": {"name": "free-placement-and-framing finite-family price", "bytes": len(base) + len(patch_data),
                                   "scope": "same real reconstructed pixels; grants all placement/framing at zero cost; not a delivered stream"}})
    by_key = {(row["arm"], row["crf"]): row for row in rows}
    changed = [by_key[("native", crf)]["receipt"]["decoded_sha256"] != by_key[("roi", crf)]["receipt"]["decoded_sha256"] for crf in config["rates"]]
    result = {"schema": 1, "scope": "finite static-rectangle development screen, not general codec superiority",
              "manifest_sha256": file_sha(manifest), "source_sha256": record["source"]["sha256"],
              "mask_sha256": record["masks"]["sha256"], "selected_prefix_frames": count, "config": config,
              "source_group": config["source_group"], "domain": config["domain"],
              "annotation_provenance": config["annotation_provenance"], "encoder": encoder_id, "decoder": decoder_id,
              "roi": {"map_sha256": file_sha(roi), "convention": "zero inside touched64 blocks; positive outside", "decoded_change_by_rate": changed,
                      "qualification": "effective pixels observed" if any(changed) else "UNQUALIFIED: no decoded change from no-ROI control"},
              "rates": rows, "anchor_rate_basis": "complete native IVF bytes; packet bytes also retained for shared-envelope comparison",
              "scientific_metrics": "not scored; score command required before quality or promotion", "command_wall_seconds": time.monotonic() - started}
    write_json(output / "result.json", result)
    return result


def validate(root: Path) -> dict[str, Any]:
    result = json.loads((root / "result.json").read_text())
    checks = {"rows": len(result["rates"]) == 12, "source_config": result["source_group"] == result["config"]["source_group"],
              "roi_effective": any(result["roi"]["decoded_change_by_rate"])}
    expected = result["config"]["width"] * result["config"]["height"] * 3 // 2 * result["selected_prefix_frames"]
    for row in result["rates"]:
        data = (root / row["packet"]).read_bytes()
        _, base, patch = transport.unpack(data)
        receipt = row["receipt"]
        key = f'{row["arm"]}-{row["crf"]}'
        checks[key + ":packet"] = len(data) == receipt["packet_bytes"] and transport.digest(data) == receipt["packet_sha256"]
        checks[key + ":pixels"] = (root / row["decoded"]).stat().st_size == expected and file_sha(root / row["decoded"]) == receipt["decoded_sha256"]
        checks[key + ":billing"] = len(base) + len(patch) + receipt["placement_and_framing_bytes"] == len(data)
    return {"passed": all(checks.values()), "checks": checks, "scientific_quality": "not established by physical validation"}


def score(root: Path, lpips_backbone: Path, dists_backbone: Path, output: Path, device: str = "cuda", source: Path | None = None, prefix: int | None = None) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from experiments.visor.h5 import dists_regions

    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("scoring requires the qualified CUDA device; no silent CPU fallback")
    if not validate(root)["passed"]:
        raise ValueError("physical result validation failed")
    record = json.loads((root / "result.json").read_text())
    config, available = record["config"], record["selected_prefix_frames"]
    count = available if prefix is None else prefix
    if type(count) is not int or not 1 <= count <= available:
        raise ValueError("invalid scoring prefix")
    inputs = json.loads((root / "input.json").read_text())
    source = source or Path(inputs["source"]["path"])
    if file_sha(source) != inputs["source"]["sha256"]:
        raise ValueError("scoring source identity mismatch")
    masks = load_masks(root / "foreground.npz", {**config, "frames": available})[:count]
    originals = raw_frames(source, config, count)
    if torch.cuda.device_count() != 1:
        raise ValueError("exactly one fleet-assigned CUDA device required")
    torch.cuda.reset_peak_memory_stats()
    lpips_net, dists_net = quality.load_lpips(lpips_backbone, device), quality.load_dists(dists_backbone, device)
    if any(parameter.device.type != "cuda" for model in (lpips_net, dists_net) for parameter in model.parameters()):
        raise ValueError("metric network silently left the assigned CUDA device")
    width, height = config["width"], config["height"]
    with torch.inference_mode():
        control = quality.yuv420_to_rgb(torch.from_numpy(originals[0].copy()[None]).to(device), width, height,
                                       full_range=config["full_range"]) / 255
        self_lpips = float(lpips_net(control * 2 - 1, control * 2 - 1).mean().item())
        self_standard_lpips = standard_lpips_scalar(lpips_net, control * 2 - 1, control * 2 - 1)
        self_dists = float(quality.masked_dists(dists_net, control, control,
                                             torch.ones((1, 1, height, width), device=device)).item())
    if any(not math.isfinite(value) or abs(value) > 1e-5 for value in (self_lpips, self_standard_lpips, self_dists)):
        raise ValueError("metric self-reconstruction control failed")
    metric_control = {"source_self_lpips": self_lpips, "source_self_dists": self_dists, "source_self_lpips_standard": self_standard_lpips, "tolerance": 1e-5}
    rows = []
    for row in record["rates"]:
        reconstructions = raw_frames(root / row["decoded"], config, count)
        values: dict[str, list[Any]] = {k: [] for k in ("full_lpips_standard", "full_lpips", "foreground_lpips", "background_lpips", "weighted_lpips", "full_dists", "foreground_dists", "background_dists", "weighted_dists", "full_rgb_psnr")}
        for frame, (original, decoded) in enumerate(zip(originals, reconstructions)):
            with torch.inference_mode():
                x = quality.yuv420_to_rgb(torch.from_numpy(original.copy()[None]).to(device), width, height, full_range=config["full_range"]) / 255
                y = quality.yuv420_to_rgb(torch.from_numpy(decoded.copy()[None]).to(device), width, height, full_range=config["full_range"]) / 255
                values[STANDARD_LPIPS].append(standard_lpips_scalar(lpips_net, x * 2 - 1, y * 2 - 1))
                spatial = lpips_net(x * 2 - 1, y * 2 - 1)
                spatial = F.interpolate(spatial, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
                fg = torch.from_numpy(masks[frame]).to(device)
                bg = ~fg
                full = torch.ones_like(fg)
                regions_for_dists = [region for region in (full, fg, bg) if region.any()]
                dists_values = iter(dists_regions(dists_net, x, y, [region[None, None] for region in regions_for_dists]))
                for metric, network in (("lpips", None), ("dists", dists_net)):
                    distances: list[float | None] = []
                    for name, region in (("full", full), ("foreground", fg), ("background", bg)):
                        value = None
                        if region.any():
                            value = float(spatial[region].mean().item()) if network is None else float(next(dists_values).item())
                            if not math.isfinite(value) or value < 0:
                                raise ValueError("missing/nonfinite/negative required metric")
                        distances.append(value)
                        values[f"{name}_{metric}"].append(value)
                    weighted = evaluation.weighted_region(evaluation.RegionalFrame(distances[1], distances[2], int(fg.sum()), int(bg.sum())))
                    values[f"weighted_{metric}"].append(weighted)
                mse = float(((x - y) * 255).square().mean().item())
                values["full_rgb_psnr"].append(quality.psnr(mse))
        # Regional component means retain explicit absent counts; all-frame
        # weighted/full means have exactly count finite entries.
        means = {key: sum(v for v in array if v is not None) / sum(v is not None for v in array)
                 if any(v is not None for v in array) else None for key, array in values.items()}
        temporal: dict[str, list[float | None]] = {key: [] for key in ("full_residual_mae", "foreground_residual_mae", "background_residual_mae", "full_source_change", "foreground_source_change", "background_source_change")}
        for tick in range(1, count):
            source_delta = originals[tick, :height].astype(np.float64) - originals[tick - 1, :height].astype(np.float64)
            shown_delta = reconstructions[tick, :height].astype(np.float64) - reconstructions[tick - 1, :height].astype(np.float64)
            residual_error, source_change = np.abs(shown_delta - source_delta), np.abs(source_delta)
            pair_mask = masks[tick] | masks[tick - 1]
            for name, support in (("full", np.ones_like(pair_mask)), ("foreground", pair_mask), ("background", ~pair_mask)):
                temporal[name + "_residual_mae"].append(float(residual_error[support].mean()) if support.any() else None)
                temporal[name + "_source_change"].append(float(source_change[support].mean()) if support.any() else None)
        duration = available / float(Fraction(*config["fps"]))
        rows.append({"arm": row["arm"], "crf": row["crf"], "packet_bytes": row["receipt"]["packet_bytes"],
                     "wire_kbps": row["receipt"]["packet_bytes"] * 8 / duration / 1000,
                     "native_base_bytes": row["receipt"]["native_base_bytes"], "native_kbps": row["receipt"]["native_base_bytes"] * 8 / duration / 1000,
                     "bound": row["bound"], "frames": values, "means": means,
                     "temporal": {"pairs": count - 1, "support": "union of adjacent source foreground masks; complement background", "units": "8-bit Y luma levels", "pass_rule": None, "values": temporal}})
    result = {"schema": 1, "physical_result_sha256": file_sha(root / "result.json"), "source_sha256": file_sha(source),
              "frames": count, "config": config, "fps": config["fps"], "duration_seconds": available / float(Fraction(*config["fps"])), "scored_duration_seconds": count / float(Fraction(*config["fps"])), "available_receiver_frames": available, "scoring_selection": "leading registered prefix; all selected frames retained", "source_group": record["source_group"], "domain": record["domain"],
              "region_pixels": {"foreground": [int(m.sum()) for m in masks], "background": [int((~m).sum()) for m in masks]},
              "annotation_provenance": record["annotation_provenance"], "device": torch.cuda.get_device_name(),
              "lpips": quality.lpips_record(lpips_backbone), "dists": quality.dists_record(dists_backbone), "rows": rows,
              "metric_control": metric_control,
              "gpu_execution": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                "cudnn": torch.backends.cudnn.version(), "cudnn_enabled": torch.backends.cudnn.enabled,
                                "compute_capability": list(torch.cuda.get_device_capability()),
                                "maximum_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "maximum_reserved_bytes": torch.cuda.max_memory_reserved(),
                                "device_count": torch.cuda.device_count(), "network_parameters_cuda": True,
                                "kernels": "torch CUDA LPIPS/VGG16 convolution and tensor/statistic operations; no attention model"},
              "protocol": LEGACY_PROTOCOL, "metric_protocol": METRIC_PROTOCOL, "metric_definitions": METRIC_DEFINITIONS}
    write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("prepare")
    for name in ("config", "source", "masks", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int)
    p.add_argument("--source", type=Path)
    p.add_argument("--masks", type=Path)
    p.add_argument("--encoder", default="SvtAv1EncApp")
    p.add_argument("--decoder", default="dav1d")
    p = sub.add_parser("validate")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p = sub.add_parser("decode")
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--decoder", default="dav1d")
    p.add_argument("--receipt", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lpips-backbone", type=Path, required=True)
    p.add_argument("--dists-backbone", type=Path, required=True)
    p.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        result = prepare(json.loads(args.config.read_text()), args.source, args.masks, args.out)
    elif args.operation == "run":
        result = run(args.manifest, args.out, prefix=args.frames, source=args.source, masks=args.masks, encoder=args.encoder, decoder=args.decoder)
    elif args.operation == "validate":
        result = validate(args.root)
        target = args.out or (Path(os.environ["PS_VALIDATION_PATH"]) if "PS_VALIDATION_PATH" in os.environ else None)
        if target:
            write_json(target, result)
        print(json.dumps(result))
        return 0 if result["passed"] else 1
    elif args.operation == "decode":
        result = transport.decode(codec.read_packet(args.packet), args.out, scratch=args.scratch, decoder=args.decoder)
        write_json(args.receipt, result)
    else:
        result = score(args.root, args.lpips_backbone, args.dists_backbone, args.out, source=args.source)
    print(json.dumps({"operation": args.operation, "rows": len(result.get("rates", result.get("rows", [])))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
