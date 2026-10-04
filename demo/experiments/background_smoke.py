"""Bounded background diagnostics (B0-B4) for the two-stage fleet schema.

Every subcommand processes only the named ``--parts`` (a whole-argument
scale placeholder), under ``--stage-seconds``, inside ``PS_STAGE_DIR``.
Outputs are diagnostic and non-citable. No training, latent fitting,
downloads, installs, or unrestricted scoring loops run here.

    inventory  B0 preview reuse and B1 provenance; writes selected-inputs.json
    codec      B2 DCVC-UF pretrained/fine-tuned arms and AV1 references
    drift      B3 one 32-frame stream versus four reset 8-frame streams
    latent     B4 frozen HNeRV latent packets
    validate   smoke-stage gate for one kind (writes PS_VALIDATION_PATH)
    summarize  B5 tables from saved status/event JSON (local, no GPU)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable
import uuid

import numpy as np

from demo.experiments import background_smoke_core as core

KINDS = ("inventory", "codec", "drift", "latent")
STATUSES = ("passed", "failed", "inconclusive", "blocked")
CODEC_ARMS = ("pre", "ft")
SHEET_ROWS = (0, 2, 4, 6)
P_COLLAPSE_DB = 6.0
DEGRADATION_DB = 2.0
DEGRADATION_BYTES = 2.0
INTRA_CHECK_SECONDS = 60.0
OPTIONAL_MIN_SECONDS = 120.0
DECISION_SETTLE_SECONDS = 15.0
SCRATCH_ROOT = Path("/var/tmp/pointstream-bg-smoke")
ADAPTER = Path(__file__).with_name("dcvc_uf_adapter.py")
INVENTORY_PARTS = (
    "env", "preview", "frames-f001c3", "frames-f002", "ckpt-image", "ckpt-htl", "ckpt-ld",
    "ckpt-hts", "ckpt-f002-htl", "ckpt-hnerv", "training",
)


class PartBlocked(Exception):
    """A prerequisite is missing; the part is reported as blocked."""


class PartFailed(Exception):
    """The part ran and a correctness check failed."""


# ------------------------------------------------------------------ parsing

def parse_parts(kind: str, value: str) -> list[str]:
    parts = [item for item in value.split(",") if item]
    if not parts or len(set(parts)) != len(parts):
        raise ValueError("parts must be a nonempty list without duplicates")
    for part in parts:
        if kind == "inventory" and part not in INVENTORY_PARTS:
            raise ValueError(f"unknown inventory part {part}")
        if kind == "codec":
            cut, *rest = part.split("-")
            if cut not in core.CUTS or rest not in (["av1"], ["intra"]) and not (
                len(rest) == 2 and rest[0] in core.STRUCTURES and rest[1] in CODEC_ARMS
            ):
                raise ValueError(f"unknown codec part {part}")
        if kind == "drift":
            cut, *rest = part.split("-")
            if cut != "f001c3" or len(rest) != 2 or rest[0] not in ("ld", "hts") or rest[1] not in CODEC_ARMS:
                raise ValueError(f"drift is planned only for primary LD/HT-S arms, got {part}")
        if kind == "latent":
            cut, _, bits = part.partition("-")
            if cut not in core.CUTS or bits not in ("b6", "b4"):
                raise ValueError(f"unknown latent part {part}")
    if kind == "latent":
        for part in parts:
            if part.endswith("-b4") and part.replace("-b4", "-b6") not in parts[: parts.index(part)]:
                raise ValueError("the 4-bit comparison requires the same cut's 6-bit part first")
    return parts


def video_checkpoint_key(factory: str, structure: str, arm: str) -> str:
    return f"uf_video:{structure}:pretrained" if arm == "pre" else f"uf_video:{structure}:{factory}:s1"


def video_checkpoint_path(factory: str, structure: str, arm: str) -> Path:
    if arm == "pre":
        return core.UF_VIDEO[structure]
    return core.WORK_ROOT / "checkpoints" / factory / structure / "s1" / "ckpt.pth.tar"


# ------------------------------------------------------------ stage context

class Stage:
    """Shared clock, ledger, scratch space and outputs for one stage."""

    def __init__(self, kind: str, seconds: float, manifests: list[Path]) -> None:
        self.kind = kind
        self.root = core.stage_dir()
        self.clock = core.StageClock(seconds, reserve=DECISION_SETTLE_SECONDS + 5)
        self.ledger = core.Ledger(self.root / "ledger.json")
        self.scratch = SCRATCH_ROOT / f"{self.root.parent.name}-{self.root.name}-{uuid.uuid4().hex[:8]}"
        self.manifest = merge_manifests([core.read_json_bounded(path) for path in manifests]) if manifests else {}
        self.manifest_paths = [str(path) for path in manifests]
        self.parts: dict[str, dict[str, Any]] = {}
        self.copies: dict[str, Path] = {}
        self.failures: list[str] = []
        self.outputs: dict[str, Any] = {}
        self.selected: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.environment: dict[str, Any] = {}

    def path(self, *names: str) -> Path:
        return core.require_stage_path(self.root.joinpath(*names))

    def run_part(self, name: str, function: Callable[[], dict[str, Any]], *, minimum_seconds: float = 0.0) -> dict[str, Any]:
        started = time.time()
        if self.clock.remaining() < max(minimum_seconds, 1.0):
            record = {"status": "blocked", "reason": f"insufficient stage time ({self.clock.remaining():.0f}s) for {name}"}
        elif len(self.failures) >= 2 and self.failures[-1] == self.failures[-2]:
            record = {"status": "blocked", "reason": "stopped after a second identical failure"}
        else:
            try:
                record = {"status": "passed", **function()}
            except PartBlocked as exc:
                record = {"status": "blocked", "reason": str(exc)}
            except (PartFailed, ValueError, OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
                record = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:2000]}
                self.failures.append(record["reason"])
        record["seconds"] = round(time.time() - started, 3)
        self.parts[name] = record
        self.ledger.record(f"{self.kind}:{name}", started, record["status"], reason=record.get("reason"))
        core.write_json_new(self.path("parts", f"{name}.json"), record)
        return record

    def checkpoint(self, key: str) -> dict[str, Any]:
        record = (self.manifest.get("checkpoints") or {}).get(key)
        if not record or not core.is_sha256(record.get("sha256")):
            raise PartBlocked(f"checkpoint {key} has no verified identity in the selected-input manifest")
        return record

    def local_copy(self, key: str) -> tuple[Path, dict[str, Any]]:
        """Hash-verified scratch copy, one bounded read of the original."""
        record = self.checkpoint(key)
        if key not in self.copies:
            started = time.time()
            destination = self.scratch / f"{len(self.copies):02d}-{Path(record['path']).name}"
            self.scratch.mkdir(parents=True, exist_ok=True)
            receipt = core.copy_verified(Path(record["path"]), destination, record["sha256"], timeout=self.clock.bounded(core.MAX_READ_SECONDS))
            self.ledger.record(f"copy:{key}", started, "passed", bytes=receipt["bytes"])
            self.copies[key] = destination
        return self.copies[key], record

    def release(self, key: str) -> None:
        path = self.copies.pop(key, None)
        if path is not None:
            path.unlink(missing_ok=True)

    def close(self) -> None:
        for key in list(self.copies):
            self.release(key)
        shutil.rmtree(self.scratch, ignore_errors=True)


def merge_manifests(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"sources": []}
    for manifest in manifests:
        if manifest.get("schema") != "pointstream.background-smoke.inputs.v1":
            raise ValueError("selected-input manifest has the wrong schema")
        merged["sources"].append(manifest.get("provenance"))
        for section in ("frames", "masks", "checkpoints", "environment"):
            for key, value in (manifest.get(section) or {}).items():
                existing = merged.setdefault(section, {}).get(key)
                if existing is not None and existing != value:
                    raise ValueError(f"selected-input manifests disagree on {section}/{key}")
                merged[section][key] = value
    return merged


def run_command(command: list[str], *, timeout: float, cwd: Path | None = None, env: dict[str, str] | None = None, log: Path | None = None) -> subprocess.CompletedProcess:
    """argv only, inside this job's process group, bounded by ``timeout``."""
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{Path(command[0]).name} exceeded {timeout:.0f}s") from exc
    if log is not None:
        with log.open("a") as stream:
            stream.write(f"$ {' '.join(command)}\n{result.stdout}\n{result.stderr}\n")
    return result


def adapter_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(core.DCVC_ROOT)
    if not env.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
        raise PartBlocked("the dispatcher did not pin CUDA_VISIBLE_DEVICES to a claimed GPU UUID")
    return env


def run_adapter(stage: Stage, action: str, arguments: list[str], *, timeout: float, report: Path) -> dict[str, Any]:
    expected = stage.path("dcvc-reference.json")
    if not expected.exists():
        core.write_json_new(expected, core.DCVC_REFERENCE_SHA256)
    command = [str(core.DCVC_PYTHON), str(ADAPTER), action, "--expected", str(expected), "--report", str(report), *arguments]
    result = run_command(command, timeout=timeout, cwd=core.DCVC_ROOT, env=adapter_env(), log=stage.path("adapter.log"))
    if result.returncode != 0:
        raise PartFailed(f"adapter {action} exited {result.returncode}: {result.stderr[-1500:]}")
    return core.read_json_bounded(report)


# ------------------------------------------------------------ frames/masks

def stage_frames(stage: Stage, cut: str, start: int, length: int, directory: Path) -> list[np.ndarray]:
    """Verify each selected source against the manifest and write PNG im00001.."""
    from PIL import Image

    records = [row for row in (stage.manifest.get("frames") or {}).get(cut, []) if start <= row["index"] < start + length]
    records.sort(key=lambda row: row["index"])
    core.validate_cut(records, start=start, length=length, clock=stage.clock)
    directory.mkdir(parents=True, exist_ok=False)
    pixels = []
    for offset, record in enumerate(records):
        with Image.open(record["path"]) as image:
            rgb = core.require_rgb_frame(np.asarray(image.convert("RGB")), expected_shape=core.FRAME_SHAPE)
        if record.get("rgb_sha256") and core.rgb_identity(rgb) != record["rgb_sha256"]:
            raise PartFailed(f"decoded RGB identity changed for {record['path']}")
        Image.fromarray(rgb, mode="RGB").save(directory / f"im{offset + 1:05d}.png")
        pixels.append(rgb)
    return pixels


def frame_ids(stage: Stage, cut: str, start: int, length: int) -> list[str]:
    rows = sorted((row for row in stage.manifest["frames"][cut] if start <= row["index"] < start + length), key=lambda row: row["index"])
    return [row["sha256"] for row in rows]


def load_masks(stage: Stage, cut: str, start: int, length: int) -> list[np.ndarray] | None:
    """Hand/arm union per frame from the chunk's sam/union.npy, when shaped sensibly."""
    rows = sorted((row for row in stage.manifest["frames"][cut] if start <= row["index"] < start + length), key=lambda row: row["index"])
    masks_meta = (stage.manifest.get("masks") or {}).get(cut) or {}
    cache: dict[str, np.ndarray] = {}
    result = []
    for row in rows:
        meta = masks_meta.get(row["chunk"])
        if not meta:
            return None
        if row["chunk"] not in cache:
            if core.sha256_file(Path(meta["path"]), timeout=stage.clock.bounded(core.MAX_READ_SECONDS))["sha256"] != meta["sha256"]:
                raise PartFailed(f"mask identity changed: {meta['path']}")
            cache[row["chunk"]] = np.load(meta["path"], allow_pickle=False)
        mask = cache[row["chunk"]]
        if mask.shape == core.FRAME_SHAPE[:2]:
            result.append(mask.astype(bool))
        elif mask.ndim == 3 and mask.shape[1:] == core.FRAME_SHAPE[:2] and row["chunk_position"] < mask.shape[0]:
            result.append(mask[row["chunk_position"]].astype(bool))
        else:
            return None
    return result


def read_pngs(directory: Path, count: int) -> list[np.ndarray]:
    from PIL import Image

    frames = []
    for index in range(count):
        with Image.open(directory / f"im{index + 1:05d}.png") as image:
            if image.mode != "RGB":
                raise PartFailed(f"decoded frame is {image.mode}, not RGB")
            frames.append(core.require_rgb_frame(np.asarray(image), expected_shape=core.FRAME_SHAPE))
    return frames


class Lpips:
    """LPIPS-Alex from locally cached weights only; never downloads."""

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.status = "not loaded"

    def _load(self) -> None:
        try:
            import lpips
            import torch
        except ImportError as exc:
            self.status = f"blocked: {exc}"
            return
        hub = Path(torch.hub.get_dir()) / "checkpoints"
        cached = sorted(hub.glob("alexnet-owt-*.pth")) if hub.is_dir() else []
        linear = Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"
        if not cached or not linear.is_file():
            self.status = "blocked: LPIPS-Alex weights are not cached locally; no download attempted"
            return
        torch.set_num_threads(4)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = lpips.LPIPS(net="alex", verbose=False).eval().to(self.device)
        self.status = f"lpips {getattr(lpips, '__version__', '?')} alex on {self.device}, cached {cached[0].name}"

    def score(self, reference: list[np.ndarray], reconstruction: list[np.ndarray]) -> dict[str, Any]:
        if self.model is None and self.status == "not loaded":
            self._load()
        if self.model is None:
            return {"status": self.status}
        import torch

        values = []
        with torch.inference_mode():
            for ref, rec in zip(reference, reconstruction):
                a = torch.from_numpy(core.lpips_input(ref)).unsqueeze(0).to(self.device)
                b = torch.from_numpy(core.lpips_input(rec)).unsqueeze(0).to(self.device)
                values.append(float(self.model(a, b).item()))
        return {"status": self.status, "input_range": "[-1, 1] RGB", "per_frame": values, "mean": float(np.mean(values))}


def contact_sheet(path: Path, sources: list[np.ndarray], decoded: list[np.ndarray], rows: tuple[int, ...] = SHEET_ROWS) -> str:
    """At most four source/decoded rows, each tile 480x270, native frames downscaled only for the sheet."""
    from PIL import Image

    tile = (480, 270)
    selected = [index for index in rows if index < len(sources)][:4]
    sheet = Image.new("RGB", (tile[0] * 2, tile[1] * len(selected)))
    for row, index in enumerate(selected):
        for column, frame in enumerate((sources[index], decoded[index])):
            sheet.paste(Image.fromarray(frame).resize(tile, Image.BICUBIC), (column * tile[0], row * tile[1]))
    sheet.save(path, quality=90)
    return str(path)


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return True


# --------------------------------------------------------------- inventory

def _git(root: Path, *args: str) -> str:
    result = run_command(["git", "-C", str(root), *args], timeout=30)
    if result.returncode != 0:
        raise PartFailed(f"git {' '.join(args)} failed in {root}: {result.stderr[-300:]}")
    return result.stdout


def inventory_env(stage: Stage) -> dict[str, Any]:
    environment: dict[str, Any] = {}
    for name, root, references in (("dcvc", core.DCVC_ROOT, core.DCVC_REFERENCE_SHA256), ("hnerv", core.HNERV_ROOT, core.HNERV_REFERENCE_SHA256)):
        diff = _git(root, "diff", "--no-color", "HEAD")
        files = {}
        for rel in references:
            receipt = core.sha256_file(root / rel, timeout=stage.clock.bounded(30))
            files[rel] = {"sha256": receipt["sha256"], "matches_reference": receipt["sha256"] == references[rel]}
        environment[name] = {
            "root": str(root), "head": _git(root, "rev-parse", "HEAD").strip(),
            "status_porcelain": _git(root, "status", "--porcelain").splitlines(),
            "tracked_diff_sha256": core.sha256_bytes(diff.encode()), "tracked_diff_bytes": len(diff.encode()),
            "files": files,
        }
    report = stage.path("dcvc-check.json")
    expected = stage.path("dcvc-reference.json")
    core.write_json_new(expected, core.DCVC_REFERENCE_SHA256)
    result = run_command([str(core.DCVC_PYTHON), str(ADAPTER), "check", "--expected", str(expected), "--report", str(report)],
                         timeout=stage.clock.bounded(90), cwd=core.DCVC_ROOT, env={**os.environ, "PYTHONPATH": str(core.DCVC_ROOT)})
    if result.returncode != 0:
        raise PartFailed(f"DCVC adapter check failed: {result.stderr[-800:]}")
    environment["dcvc"]["adapter_check"] = core.read_json_bounded(report)
    environment["dcvc"]["revision_matches_mirror"] = environment["dcvc"]["head"] == core.DCVC_REVISION
    probe = run_command([sys.executable, "-m", "demo.experiments.hnerv_frozen", "probe", "--stub", str(stage.path("hnerv-stub"))],
                        timeout=stage.clock.bounded(90))
    environment["hnerv"]["import_probe"] = json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode == 0 else {"error": probe.stderr[-800:]}
    ffmpeg_path = core.resolve_ffmpeg()
    if ffmpeg_path is None:
        environment["ffmpeg"] = {"available": False, "path": None}
    else:
        ffmpeg = run_command([ffmpeg_path, "-hide_banner", "-version"], timeout=20)
        environment["ffmpeg"] = {"available": ffmpeg.returncode == 0, "path": ffmpeg_path, "version": ffmpeg.stdout.splitlines()[:1]}
    environment["lpips"] = _lpips_status()
    blockers = []
    if not environment["dcvc"]["adapter_check"]["reference"]["matches_reference"]:
        blockers.append("installed DCVC differs from the mirrored inference files")
    if "error" in environment["dcvc"]["adapter_check"]["environment"].get("extension", {}):
        blockers.append("DCVC CUDA extension does not import")
    if "error" in environment["hnerv"]["import_probe"]:
        blockers.append("HNeRV modules do not import in the worker interpreter")
    if not all(f["matches_reference"] for f in environment["hnerv"]["files"].values()):
        blockers.append("installed HNeRV model/quantizer files differ from the reviewed revision")
    stage.environment = environment
    if blockers:
        raise PartBlocked("; ".join(blockers))
    return {"dcvc_head": environment["dcvc"]["head"], "dcvc_dirty": environment["dcvc"]["status_porcelain"],
            "hnerv_head": environment["hnerv"]["head"], "lpips": environment["lpips"],
            "extension": environment["dcvc"]["adapter_check"]["environment"].get("extension")}


def _lpips_status() -> str:
    metric = Lpips()
    metric._load()
    return metric.status


def inventory_preview(stage: Stage) -> dict[str, Any]:
    """B0: verify the saved preview without writing to it."""
    root = core.PREVIEW_ROOT / "canvas"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise PartBlocked(f"preview manifest is absent: {manifest_path}")
    receipt = core.sha256_file(manifest_path, timeout=stage.clock.bounded(30))
    manifest = core.read_json_bounded(manifest_path)
    rows = []
    for clip in manifest.get("clips", []):
        panels = {}
        for index, rel in (clip.get("panels") or {}).items():
            path = core.PREVIEW_ROOT / rel
            panels[index] = core.sha256_file(path, timeout=stage.clock.bounded(30))["sha256"] if path.is_file() else None
        rates = clip.get("rates_kbps") or {}
        corrected = {key: value for key, value in rates.items() if key != "hnerv"}
        rows.append({
            "factory": clip.get("factory"), "segment_start": clip.get("segment_start_in_holdout"),
            "frames": clip.get("frames"), "panels_sha256": panels, "all_panels_present": all(panels.values()),
            "display_rates_kbps": corrected,
            "hnerv_display_rate_label": "setup-inclusive estimate (total_bits includes decoder weights); not a streaming rate",
            "hnerv_setup_inclusive_estimate_kbps": rates.get("hnerv"),
            "psnr_db": clip.get("psnr_db"),
            "verified": "manifest and panel file identities today",
            "unknown": "checkpoint hashes at preview time; native decoded frames were not retained by the preview",
        })
    return {"manifest": {"path": str(manifest_path), "sha256": receipt["sha256"]}, "scope": manifest.get("scope"),
            "qp": manifest.get("qp"), "reuse_table": rows, "citable": False}


def inventory_frames(stage: Stage, cut: str) -> dict[str, Any]:
    """Hold-out 120..151 identities (and their masks) for one cut."""
    from PIL import Image

    factory, stem, expected = core.CUTS[cut]
    root = core.WORK_ROOT / "holdouts" / stem
    chunks = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("chunk_"))
    frames: list[tuple[Path, str, int]] = []
    for chunk in chunks:
        kept = sorted((chunk / "fill" / "kept").glob("*.jpg"))
        frames.extend((path, chunk.name, position) for position, path in enumerate(kept))
    if len(frames) != expected:
        raise PartFailed(f"{stem} has {len(frames)} kept frames, expected {expected}")
    records, masks = [], {}
    for index in range(core.PRIMARY_START, core.PRIMARY_START + core.DRIFT_LENGTH):
        path, chunk, position = frames[index]
        receipt = core.sha256_file(path, timeout=stage.clock.bounded(30))
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
            mode, size = image.mode, image.size
        core.require_rgb_frame(rgb, expected_shape=core.FRAME_SHAPE)
        records.append({
            "index": index, "path": str(path), "sha256": receipt["sha256"], "bytes": receipt["bytes"],
            "chunk": chunk, "chunk_position": position, "stored_mode": mode, "size": list(size),
            "rgb_sha256": core.rgb_identity(rgb), "fill_passthrough_symlink": path.is_symlink(),
        })
        union = root / chunk / "sam" / "union.npy"
        if chunk not in masks and union.is_file():
            mask_receipt = core.sha256_file(union, timeout=stage.clock.bounded(30))
            shape = list(np.load(union, mmap_mode="r", allow_pickle=False).shape)
            masks[chunk] = {"path": str(union), "sha256": mask_receipt["sha256"], "shape": shape}
    stage.selected.setdefault("frames", {})[cut] = records
    stage.selected.setdefault("masks", {})[cut] = masks
    return {"cut": cut, "factory": factory, "stem": stem, "holdout_frames": len(frames), "selected": [120, 151],
            "masks": {k: v["shape"] for k, v in masks.items()}, "decoder": "Pillow JPEG to RGB"}


def _metadata_summary(path: Path) -> dict[str, Any]:
    obj, storages, _prefix = core.read_torch_zip_metadata(path)
    state = core.canonical_state_dict(core.checkpoint_state(obj))
    dtypes = sorted({value.dtype for value in state.values() if isinstance(value, core.TensorRef)})
    epoch = obj.get("epoch") if isinstance(obj, dict) else None
    return {"keys": len(state), "signature": core.state_signature(state), "dtypes": dtypes,
            "embedded_epoch": epoch if isinstance(epoch, int) else None, "_state": state, "_storages": storages}


def register_checkpoint(stage: Stage, key: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PartBlocked(f"missing checkpoint {key}: {path}")
    receipt = core.sha256_file(path, timeout=stage.clock.bounded(core.MAX_READ_SECONDS))
    record = {"path": str(path), "sha256": receipt["sha256"], "bytes": receipt["bytes"], "hash_seconds": receipt["seconds"]}
    if path.suffix == ".tar" or path.name.endswith(".pth"):
        try:
            summary = _metadata_summary(path)
            record.update({k: v for k, v in summary.items() if not k.startswith("_")})
            stage.metadata[key] = summary
        except Exception as exc:  # legacy/non-zip or unsupported pickle: recorded, strict load decides later
            record["metadata_error"] = f"{type(exc).__name__}: {exc}"[:500]
    stage.selected.setdefault("checkpoints", {})[key] = record
    return record


def inventory_video(stage: Stage, structure: str, factories: tuple[str, ...], *, pretrained: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if pretrained:
        result["pretrained"] = register_checkpoint(stage, video_checkpoint_key("", structure, "pre"), video_checkpoint_path("", structure, "pre"))
    for factory in factories:
        key = video_checkpoint_key(factory, structure, "ft")
        record = register_checkpoint(stage, key, video_checkpoint_path(factory, structure, "ft"))
        pre_key = video_checkpoint_key("", structure, "pre")
        if pre_key in stage.metadata and key in stage.metadata:
            try:
                record["strict_shapes_vs_pretrained"] = core.compare_state_dict_shapes(
                    {k: _ShapeOnly(v.shape) for k, v in stage.metadata[pre_key]["_state"].items()}, stage.metadata[key]["_state"])
            except ValueError as exc:
                record["strict_shapes_vs_pretrained"] = {"strict_compatible": False, "error": str(exc)[:1500]}
        record["stages"] = stage_status_report(factory, structure, stage.metadata.get(key))
        result[factory] = record
    return result


class _ShapeOnly:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def stage_status_report(factory: str, structure: str, final: dict[str, Any] | None) -> dict[str, Any]:
    """Embedded epochs per stage, from status-file metadata only."""
    report: dict[str, Any] = {}
    base = core.WORK_ROOT / "checkpoints" / factory / structure
    for name in ("s0", "s1"):
        directory = base / name
        statuses = sorted(directory.glob("status_epo*.pth.tar")) if directory.is_dir() else []
        if not statuses:
            report[name] = {"status_files": []}
            continue
        last = statuses[-1]
        try:
            obj, storages, _prefix = core.read_torch_zip_metadata(last)
            epoch = obj.get("epoch") if isinstance(obj, dict) else None
            entry: dict[str, Any] = {"status_files": [p.name for p in statuses], "last": last.name,
                                     "embedded_epoch_zero_based": epoch if isinstance(epoch, int) else None}
            if name == "s1" and final is not None and isinstance(obj, dict) and isinstance(obj.get("net"), dict):
                entry["s1_ckpt_vs_status_net"] = core.serialized_state_equality((obj["net"], storages), (final["_state"], final["_storages"]))
        except Exception as exc:
            entry = {"status_files": [p.name for p in statuses], "metadata_error": f"{type(exc).__name__}: {exc}"[:500]}
        report[name] = entry
    epochs = {n: report[n].get("embedded_epoch_zero_based") for n in ("s0", "s1")}
    report["epochs_reported_separately"] = epochs
    report["note"] = "zero-based per-stage epochs; never summed"
    return report


def inventory_hnerv(stage: Stage) -> dict[str, Any]:
    result = {}
    for factory in ("factory001", "factory002"):
        root = core.WORK_ROOT / "checkpoints" / factory / "hnerv"
        candidates = sorted(path for path in root.glob("*/model_latest.pth") if "Dim64_0.5" in str(path)) if root.is_dir() else []
        if not candidates:
            candidates = sorted(path for path in root.glob("*/*/model_latest.pth") if "Dim64_0.5" in str(path)) if root.is_dir() else []
        if len(candidates) != 1:
            result[factory] = {"status": "blocked", "candidates": [str(p) for p in candidates]}
            continue
        record = register_checkpoint(stage, f"hnerv:{factory}", candidates[0])
        state = stage.metadata.get(f"hnerv:{factory}", {}).get("_state", {})
        embed = state.get("decoder.0.conv.downconv.weight")
        record["embed_channels"] = embed.shape[1] if embed is not None else None
        record["decoder_width"] = embed.shape[0] if embed is not None else None
        result[factory] = record
    if not any("sha256" in value for value in result.values()):
        raise PartBlocked("no unambiguous Dim64_0.5 HNeRV checkpoint")
    return result


def inventory_training(stage: Stage) -> dict[str, Any]:
    holdouts_path = core.DEMO_ROOT / "holdouts" / "holdouts.json"
    holdouts = core.read_json_bounded(holdouts_path) if holdouts_path.is_file() else []
    ranges: dict[str, range] = {}
    for item in holdouts if isinstance(holdouts, list) else []:
        start = next((item.get(k) for k in ("source_start_frame", "start_frame", "first_frame") if isinstance(item.get(k), int)), None)
        count = next((item.get(k) for k in ("frames", "frame_count") if isinstance(item.get(k), int)), None)
        source = item.get("source_stem") or item.get("source")
        if start is not None and count and source:
            ranges[Path(str(source)).stem] = range(start, start + count)
    reports = {}
    for factory in ("factory001", "factory002"):
        path = core.WORK_ROOT / "datasets" / factory / "description.json"
        if not path.is_file():
            reports[factory] = {"status": "blocked", "reason": f"missing {path}"}
            continue
        receipt = core.sha256_file(path, timeout=stage.clock.bounded(30))
        report = core.training_identity_report(core.read_json_bounded(path), factory, {k: v for k, v in ranges.items() if any(k == s[0] for s in core.TRAINING_SOURCES[factory])})
        report.update(description={"path": str(path), "sha256": receipt["sha256"]})
        if not ranges:
            report["holdout_overlap"] = "unknown: holdouts.json carries no source frame offsets; hold-outs are separate last-10s videos"
        reports[factory] = report
    if any(r.get("status") == "blocked" for r in reports.values()):
        raise PartBlocked(json.dumps(reports)[:1500])
    return {"holdouts_json": str(holdouts_path) if holdouts_path.is_file() else None, "factories": reports}


def run_inventory(stage: Stage, parts: list[str]) -> None:
    actions = {
        "env": lambda: inventory_env(stage),
        "preview": lambda: inventory_preview(stage),
        "frames-f001c3": lambda: inventory_frames(stage, "f001c3"),
        "frames-f002": lambda: inventory_frames(stage, "f002"),
        "ckpt-image": lambda: register_checkpoint(stage, "uf_image", core.UF_IMAGE),
        "ckpt-htl": lambda: inventory_video(stage, "htl", ("factory001",)),
        "ckpt-ld": lambda: inventory_video(stage, "ld", ("factory001",)),
        "ckpt-hts": lambda: inventory_video(stage, "hts", ("factory001",)),
        "ckpt-f002-htl": lambda: inventory_video(stage, "htl", ("factory002",), pretrained=False),
        "ckpt-hnerv": lambda: inventory_hnerv(stage),
        "training": lambda: inventory_training(stage),
    }
    for part in parts:
        stage.run_part(part, actions[part])
    manifest = {
        "schema": "pointstream.background-smoke.inputs.v1", "provenance": core.job_provenance(),
        "environment": {"stage_" + os.environ.get("PS_STAGE", "local"): stage.environment} if stage.environment else {},
        **stage.selected,
    }
    path = core.write_json_new(stage.path("selected-inputs.json"), manifest)
    path.chmod(0o444)
    stage.outputs["selected_inputs"] = {"path": str(path), "sha256": core.sha256_file(path, timeout=30)["sha256"]}


# ------------------------------------------------------------------- codec

def _arm_summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = ("container_bytes", "kbps", "mean_frame_psnr_db", "pooled_mse_psnr_db", "i_frame_psnr_db", "p_frames_mean_psnr_db",
            "temporal_reconstruction_error", "lpips_mean", "mask_region", "decode_independent", "peak_memory_mib", "flags")
    return {key: report.get(key) for key in keys if key in report}


def encode_decode(stage: Stage, name: str, structure: str, streams: list[dict[str, Any]], *, image: tuple[Path, dict], video: tuple[Path, dict] | None,
                  force_intra: bool, cap: float, repeat: list[str]) -> tuple[dict, dict]:
    work = stage.path(name)
    plan = {"qp": core.QP, "reset_interval": 0, "force_intra": force_intra, "streams": streams}
    plan_path = work / "encode-plan.json"
    core.write_json_new(plan_path, plan)
    checkpoint_args = ["--structure", structure, "--image-ckpt", str(image[0]), "--image-sha256", image[1]["sha256"]]
    if video is not None:
        checkpoint_args += ["--video-ckpt", str(video[0]), "--video-sha256", video[1]["sha256"]]
    case = core.StageClock(min(cap, stage.clock.remaining()))
    encoded = run_adapter(stage, "encode", ["--plan", str(plan_path), *checkpoint_args], timeout=case.bounded(cap), report=work / "encode-report.json")
    sources = sorted({Path(s["frames_dir"]) for s in streams})
    sealed = [path.with_name(path.name + ".sealed") for path in sources]
    decode_plan = {"streams": [{"name": s["name"], "container": s["container"], "out_dir": str(work / f"decoded-{s['name']}")} for s in streams], "repeat": repeat}
    core.write_json_new(work / "decode-plan.json", decode_plan)
    for path, hidden in zip(sources, sealed):
        path.rename(hidden)
    try:
        decoded = run_adapter(stage, "decode", ["--plan", str(work / "decode-plan.json"), *checkpoint_args], timeout=case.bounded(cap), report=work / "decode-report.json")
    finally:
        for path, hidden in zip(sources, sealed):
            hidden.rename(path)
    return encoded, decoded


def evaluate_stream(stage: Stage, encoded: dict, decoded: dict, name: str, sources: list[np.ndarray], masks: list[np.ndarray] | None,
                    lpips: Lpips, work: Path, *, sheet: bool = True) -> dict[str, Any]:
    enc = next(s for s in encoded["streams"] if s["name"] == name)
    decs = [s for s in decoded["streams"] if s["name"] == name]
    first = next(s for s in decs if not s["repeat"])
    frames = read_pngs(Path(first["out_dir"]), len(sources))
    container = Path(enc["container"])
    if container.stat().st_size != enc["container_bytes"]:
        raise PartFailed("container size differs from the encoder report")
    repeats = [s for s in decs if s["repeat"]]
    if any(s["decoded_png_sha256"] != first["decoded_png_sha256"] for s in repeats):
        raise PartFailed("decoding the same container after other streams changed the output")
    metrics = core.sequence_metrics(sources, frames, masks=masks, expected_shape=core.FRAME_SHAPE)
    psnr = metrics["frame_psnr_db"]
    i_recon = Path(work / f"{name}-i-recon.png")
    i_match = None
    if i_recon.is_file():
        i_match = bool(np.array_equal(read_png(i_recon), frames[0]))
    report = {
        "frames": len(frames), "container": str(container), "container_bytes": enc["container_bytes"],
        "container_header_bytes": enc["container_header_bytes"], "native_stream_bytes": enc["native_stream_bytes"],
        "kbps": core.rate_kbps(enc["container_bytes"], frames=len(frames)), "nals": enc["nals"],
        "decoded_nal_types": first["nal_types"], "frame_psnr_db": psnr,
        "mean_frame_psnr_db": metrics["mean_frame_psnr_db"], "pooled_mse_psnr_db": metrics["pooled_mse_psnr_db"],
        "i_frame_psnr_db": psnr[0], "p_frames_mean_psnr_db": float(np.mean(psnr[1:])) if len(psnr) > 1 else None,
        "temporal_reconstruction_error": metrics["temporal_reconstruction_error"],
        "mask_region": metrics.get("mask_region", "unavailable: no aligned hand/arm union for these frames"),
        "encoder_i_recon_equals_decoder": i_match,
        "decode_independent": True, "decode_inputs": decoded["inputs"],
        "decoded_rgb_sha256": [core.rgb_identity(frame) for frame in frames],
    }
    scores = lpips.score(sources, frames)
    report["lpips"] = scores
    report["lpips_mean"] = scores.get("mean")
    if sheet:
        report["sheet"] = contact_sheet(work / f"{name}-sheet.jpg", sources, frames)
    if i_match is False:
        raise PartFailed("encoder I-frame reconstruction differs from the independent decode")
    if not finite_tree({k: v for k, v in report.items() if k != "lpips"}):
        raise PartFailed("nonfinite metric")
    return report


def read_png(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return core.require_rgb_frame(np.asarray(image.convert("RGB")))


def codec_case(stage: Stage, part: str, lpips: Lpips) -> dict[str, Any]:
    cut, structure, arm = part.split("-")
    factory = core.CUTS[cut][0]
    image = stage.local_copy("uf_image")
    video_key = video_checkpoint_key(factory, structure, arm)
    video = stage.local_copy(video_key)
    work = stage.path(part)
    sources = stage_frames(stage, cut, core.PRIMARY_START, core.SHORT_LENGTH, work / "src")
    stream = {"name": part, "frames_dir": str(work / "src"), "frame_count": core.SHORT_LENGTH, "height": core.HEIGHT, "width": core.WIDTH,
              "container": str(work / f"{part}.psdc"), "i_recon_png": str(work / f"{part}-i-recon.png")}
    encoded, decoded = encode_decode(stage, part, structure, [stream], image=image, video=video, force_intra=False, cap=core.CASE_SECONDS, repeat=[part])
    stage.release(video_key)
    report = evaluate_stream(stage, encoded, decoded, part, sources, load_masks(stage, cut, core.PRIMARY_START, core.SHORT_LENGTH), lpips, work)
    expected_types = ["I"] + ["P"] * (len(report["nals"]) - 1)
    if [n["type"] for n in report["nals"]] != expected_types or report["decoded_nal_types"] != expected_types:
        raise PartFailed("frame types differ from one I frame followed by prediction")
    report.update(checkpoints={"image": image[1], "video": video[1]}, structure=structure, arm=arm, qp=core.QP,
                  reset_interval=0, frame_ids=frame_ids(stage, cut, core.PRIMARY_START, core.SHORT_LENGTH),
                  setup_weights_bytes={"image": image[1]["bytes"], "video": video[1]["bytes"]},
                  peak_memory={"encode": encoded["peak_memory"], "decode": decoded["peak_memory"]},
                  peak_memory_mib=max(encoded["peak_memory"]["torch_max_reserved_mib"], decoded["peak_memory"]["torch_max_reserved_mib"]),
                  load_seconds={"encode": encoded["load_seconds"], "decode": decoded["load_seconds"]})
    flags = []
    if report["p_frames_mean_psnr_db"] is not None and report["p_frames_mean_psnr_db"] < report["i_frame_psnr_db"] - P_COLLAPSE_DB:
        flags.append("prediction_frames_collapse")
    report["flags"] = flags
    return report


def intra_check(stage: Stage, cut: str, lpips: Lpips) -> dict[str, Any]:
    """Single-frame force-intra round trip with the shared image checkpoint."""
    image = stage.local_copy("uf_image")
    part = f"{cut}-intra"
    work = stage.path(part)
    sources = _stage_one(stage, cut, work / "src")
    stream = {"name": part, "frames_dir": str(work / "src"), "frame_count": 1, "height": core.HEIGHT, "width": core.WIDTH,
              "container": str(work / f"{part}.psdc"), "i_recon_png": str(work / f"{part}-i-recon.png")}
    encoded, decoded = encode_decode(stage, part, "ld", [stream], image=image, video=None, force_intra=True, cap=INTRA_CHECK_SECONDS, repeat=[])
    report = evaluate_stream(stage, encoded, decoded, part, sources, None, lpips, work)
    report.update(checkpoints={"image": image[1]}, force_intra=True, qp=core.QP)
    return report


def _stage_one(stage: Stage, cut: str, directory: Path) -> list[np.ndarray]:
    """First primary frame alone, verified against the manifest."""
    from PIL import Image

    record = next(row for row in stage.manifest["frames"][cut] if row["index"] == core.PRIMARY_START)
    if core.sha256_file(Path(record["path"]), timeout=stage.clock.bounded(30))["sha256"] != record["sha256"]:
        raise PartFailed(f"frame content changed: {record['path']}")
    directory.mkdir(parents=True, exist_ok=False)
    with Image.open(record["path"]) as image:
        rgb = core.require_rgb_frame(np.asarray(image.convert("RGB")), expected_shape=core.FRAME_SHAPE)
    Image.fromarray(rgb, mode="RGB").save(directory / "im00001.png")
    return [rgb]


def av1_reference(stage: Stage, cut: str, lpips: Lpips) -> dict[str, Any]:
    from demo.pipeline.maps.av1_crf import AV1_CRF, AV1_PRESET, av1_output_args

    ffmpeg_bin = core.resolve_ffmpeg()
    if ffmpeg_bin is None:
        raise PartBlocked("ffmpeg is not available at /opt/local/bin/ffmpeg or on PATH")
    part = f"{cut}-av1"
    work = stage.path(part)
    sources = stage_frames(stage, cut, core.PRIMARY_START, core.SHORT_LENGTH, work / "src")
    masks = load_masks(stage, cut, core.PRIMARY_START, core.SHORT_LENGTH)
    rungs = {}
    for rung, scale in (("240p", "426:240"), ("1080p", None)):
        stream = work / f"{rung}.mp4"
        encode = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-framerate", str(core.FPS), "-start_number", "1",
                  "-i", str(work / "src" / "im%05d.png"), *av1_output_args(scale), "-g", "8",
                  "-svtav1-params", "keyint=8:keyint-min=8", str(stream)]
        result = run_command(encode, timeout=stage.clock.bounded(90), log=work / "ffmpeg.log")
        if result.returncode != 0 or not stream.is_file():
            raise PartFailed(f"AV1 {rung} encode failed: {result.stderr[-500:]}")
        out = work / f"decoded-{rung}"
        out.mkdir()
        decode = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", str(stream), "-vf", "scale=1920:1080:flags=bicubic",
                  "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-start_number", "1", str(out / "im%05d.png")]
        result = run_command(decode, timeout=stage.clock.bounded(60), log=work / "ffmpeg.log")
        if result.returncode != 0 or len(list(out.glob("im*.png"))) != core.SHORT_LENGTH:
            raise PartFailed(f"AV1 {rung} decode did not return {core.SHORT_LENGTH} frames")
        frames = read_pngs(out, core.SHORT_LENGTH)
        metrics = core.sequence_metrics(sources, frames, masks=masks, expected_shape=core.FRAME_SHAPE)
        size = stream.stat().st_size
        scores = lpips.score(sources, frames)
        rungs[rung] = {
            "file_bytes": size, "kbps": core.rate_kbps(size, frames=core.SHORT_LENGTH), "crf": AV1_CRF, "preset": AV1_PRESET,
            "gop": 8, "upscale": "bicubic to 1920x1080" if scale else "native", "mean_frame_psnr_db": metrics["mean_frame_psnr_db"],
            "pooled_mse_psnr_db": metrics["pooled_mse_psnr_db"], "frame_psnr_db": metrics["frame_psnr_db"],
            "temporal_reconstruction_error": metrics["temporal_reconstruction_error"], "mask_region": metrics.get("mask_region"),
            "lpips_mean": scores.get("mean"), "lpips": scores, "sheet": contact_sheet(work / f"{rung}-sheet.jpg", sources, frames),
        }
    return {"rungs": rungs, "frame_ids": frame_ids(stage, cut, core.PRIMARY_START, core.SHORT_LENGTH), "container": "mp4 file bytes, all headers included"}


def run_codec(stage: Stage, parts: list[str]) -> None:
    lpips = Lpips()
    intra_done: set[str] = set()
    for part in parts:
        cut, *rest = part.split("-")
        if rest == ["av1"]:
            stage.run_part(part, lambda: av1_reference(stage, cut, lpips))
            continue
        if rest == ["intra"]:
            stage.run_part(part, lambda: intra_check(stage, cut, lpips), minimum_seconds=INTRA_CHECK_SECONDS)
            intra_done.add(cut)
            continue
        record = stage.run_part(part, lambda: codec_case(stage, part, lpips), minimum_seconds=45)
        if "prediction_frames_collapse" in record.get("flags", []) and cut not in intra_done:
            stage.run_part(f"{cut}-intra", lambda: intra_check(stage, cut, lpips), minimum_seconds=INTRA_CHECK_SECONDS)
            intra_done.add(cut)
    stage.outputs["degradation"] = degradation_flags(stage.parts)


def degradation_flags(parts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Fine-tune versus pretrained of the same cut and structure."""
    result = {}
    for name, record in parts.items():
        bits = name.split("-")
        if len(bits) != 3 or bits[2] != "ft" or record.get("status") != "passed":
            continue
        pre = parts.get(f"{bits[0]}-{bits[1]}-pre")
        if not pre or pre.get("status") != "passed":
            result[name] = "untested: pretrained arm unavailable"
            continue
        drop = pre["mean_frame_psnr_db"] - record["mean_frame_psnr_db"]
        ratio = record["container_bytes"] / pre["container_bytes"]
        result[name] = {"psnr_drop_db": drop, "bytes_ratio": ratio,
                        "degraded": drop > DEGRADATION_DB or ratio > DEGRADATION_BYTES}
    return result


# ------------------------------------------------------------------- drift

def drift_case(stage: Stage, part: str, lpips: Lpips) -> dict[str, Any]:
    cut, structure, arm = part.split("-")
    factory = core.CUTS[cut][0]
    image = stage.local_copy("uf_image")
    video_key = video_checkpoint_key(factory, structure, arm)
    video = stage.local_copy(video_key)
    work = stage.path(part)
    sources = stage_frames(stage, cut, core.PRIMARY_START, core.DRIFT_LENGTH, work / "src")
    segment_dirs = []
    for segment in range(4):
        directory = work / f"src-seg{segment}"
        directory.mkdir()
        for offset in range(core.SHORT_LENGTH):
            os.link(work / "src" / f"im{segment * 8 + offset + 1:05d}.png", directory / f"im{offset + 1:05d}.png")
        segment_dirs.append(directory)
    def stream(name: str, directory: Path, count: int) -> dict[str, Any]:
        return {"name": name, "frames_dir": str(directory), "frame_count": count, "height": core.HEIGHT, "width": core.WIDTH,
                "container": str(work / f"{name}.psdc"), "i_recon_png": None}
    streams = [stream("seg3", segment_dirs[3], 8), stream("seg0", segment_dirs[0], 8), stream("seg1", segment_dirs[1], 8),
               stream("seg2", segment_dirs[2], 8), stream("long", work / "src", core.DRIFT_LENGTH)]
    encoded, decoded = encode_decode(stage, part, structure, streams, image=image, video=video, force_intra=False,
                                     cap=min(stage.clock.remaining(), 150.0), repeat=["seg3"])
    stage.release(video_key)
    masks = load_masks(stage, cut, core.PRIMARY_START, core.DRIFT_LENGTH)
    long = evaluate_stream(stage, encoded, decoded, "long", sources, masks, lpips, work)
    segments = []
    for segment in range(4):
        span = slice(segment * 8, segment * 8 + 8)
        segments.append(evaluate_stream(stage, encoded, decoded, f"seg{segment}", sources[span], masks[span] if masks else None, lpips, work, sheet=False))
    reset_frames = [frame for segment in segments for frame in segment["decoded_rgb_sha256"]]
    if len(reset_frames) != core.DRIFT_LENGTH:
        raise PartFailed("reset segments do not cover the 32 frames")
    reset_psnr = [value for segment in segments for value in segment["frame_psnr_db"]]
    rows = lambda values: [{"index": core.PRIMARY_START + i, "psnr_db": v} for i, v in enumerate(values)]
    return {
        "structure": structure, "arm": arm, "qp": core.QP, "reset_interval": 0, "checkpoints": {"image": image[1], "video": video[1]},
        "frame_ids": frame_ids(stage, cut, core.PRIMARY_START, core.DRIFT_LENGTH),
        "one_stream": {**_arm_summary(long), "drift": core.summarize_drift(rows(long["frame_psnr_db"])), "first_frame_mse_psnr_db": long["frame_psnr_db"][0], "last_frame_psnr_db": long["frame_psnr_db"][-1]},
        "four_reset_streams": {"container_bytes_sum": sum(s["container_bytes"] for s in segments),
                               "kbps": core.rate_kbps(sum(s["container_bytes"] for s in segments), frames=core.DRIFT_LENGTH),
                               "drift": core.summarize_drift(rows(reset_psnr)),
                               "per_segment": [_arm_summary(s) for s in segments]},
        "segment_decode_order": [s["name"] for s in decoded["streams"]],
        "state_independence": "seg3 decoded first and again after seg0..seg2 and the 32-frame stream; identical output required",
        "setup_weights_bytes": {"image": image[1]["bytes"], "video": video[1]["bytes"], "charged": "once, outside stream totals"},
        "peak_memory_mib": max(encoded["peak_memory"]["torch_max_reserved_mib"], decoded["peak_memory"]["torch_max_reserved_mib"]),
    }


def run_drift(stage: Stage, parts: list[str]) -> None:
    lpips = Lpips()
    for part in parts:
        optional = part.split("-")[1] == "hts"
        stage.run_part(part, lambda: drift_case(stage, part, lpips), minimum_seconds=OPTIONAL_MIN_SECONDS if optional else 60)


# ------------------------------------------------------------------ latent

def latent_case(stage: Stage, part: str, session: dict[str, Any], lpips: Lpips) -> dict[str, Any]:
    import torch

    from demo.experiments import hnerv_frozen as frozen
    from demo.experiments import hnerv_latent_packet as packets

    cut, _, bits_name = part.partition("-")
    bits = int(bits_name[1:])
    factory = core.CUTS[cut][0]
    key = f"hnerv:{factory}"
    if session.get("factory") != factory:
        session.clear()
        session["imports"] = frozen.enable_imports(stage.path("hnerv-stub"))
        path, record = stage.local_copy(key)
        state, meta = frozen.load_verified_state(path, record["sha256"])
        model, args, dims = frozen.build_model(state)
        quantized, codes = frozen.quantize_decoder(model)
        setup = frozen.write_setup_package(codes, stage.path(f"{factory}-decoder-setup.npz"))
        decoder, restored = frozen.decoder_from_setup(state, Path(setup["path"]))
        quantized_state = quantized.state_dict()
        setup_exact = all(torch.equal(restored[k], quantized_state[k]) for k in restored)
        work = stage.path(f"{cut}-latent")
        sources = stage_frames(stage, cut, core.PRIMARY_START, core.DRIFT_LENGTH, work / "src")
        embed = frozen.encode_frames(model, sources)
        np.save(work / "embeddings-float32.npy", embed.detach().cpu().numpy())
        session.update(factory=factory, state=state, meta=meta, model=model, quantized=quantized, decoder=decoder, setup=setup,
                       direct_decoder=frozen.decoder_of(quantized),
                       setup_exact=setup_exact, sources=sources, embed=embed, dims=dims, work=work)
        stage.release(key)
    embed, sources, work = session["embed"], session["sources"], session["work"]
    expected_channels = int(session["dims"]["embed_dim"])
    if tuple(embed.shape) != (core.DRIFT_LENGTH, expected_channels, 9, 16):
        raise PartFailed(f"embedding shape {tuple(embed.shape)} differs from the checkpoint's {expected_channels}x9x16")
    ids = frame_ids(stage, cut, core.PRIMARY_START, core.DRIFT_LENGTH)
    results = {}
    for length in packets.SEGMENT_LENGTHS:
        segments, direct = [], []
        for start in range(0, core.DRIFT_LENGTH, length):
            codes, quantizer, new_value = frozen.quantize_embeddings(embed[start:start + length], bits)
            numpy_dequant = packets.dequantize(codes, quantizer)
            if not np.array_equal(numpy_dequant, new_value.detach().cpu().numpy()):
                raise PartFailed("packet dequantization differs from hnerv_utils.quant_tensor's reconstruction")
            metadata = {"checkpoint_sha256": session["meta"]["sha256"], "frame_start": core.PRIMARY_START + start,
                        "fps": {"numerator": 30, "denominator": 1}, "frame_ids": ids[start:start + length],
                        "quantizer": quantizer, "decoder_setup_bytes": session["setup"]["bytes"]}
            segments.append({"codes": codes, "metadata": metadata})
            direct.extend(frozen.decode_pixels(session["direct_decoder"], new_value))
        written = packets.write_segment_packets(segments, work / f"b{bits}-L{length:02d}", bit_depth=bits)
        decoded_by_method: dict[str, list[np.ndarray]] = {}
        for method in packets.METHODS:
            frames = []
            for record in sorted((r for r in written["packets"] if r["method"] == method), key=lambda r: r["frame_start"]):
                codes, header = packets.decode_packet(Path(record["path"]).read_bytes(), expected_checkpoint_sha256=session["meta"]["sha256"])
                values = torch.from_numpy(packets.dequantize(codes, header["quantizer"])).to("cuda")
                frames.extend(frozen.decode_pixels(session["decoder"], values))
            decoded_by_method[method] = frames
        identical = all(all(np.array_equal(a, b) for a, b in zip(frames, direct)) for frames in decoded_by_method.values())
        if not identical:
            raise PartFailed(f"packet-decoded pixels differ from direct quantized decoding at L={length}")
        metrics = core.sequence_metrics(sources, decoded_by_method["packed"], expected_shape=core.FRAME_SHAPE)
        scores = lpips.score(sources, decoded_by_method["packed"])
        table = packets.summarize_packets(written["packets"], frames=core.DRIFT_LENGTH, setup_bytes=session["setup"]["bytes"])
        results[str(length)] = {
            "packets": table, "pixels_identical_across_methods_and_direct": identical,
            "mean_frame_psnr_db": metrics["mean_frame_psnr_db"], "pooled_mse_psnr_db": metrics["pooled_mse_psnr_db"],
            "temporal_reconstruction_error": metrics["temporal_reconstruction_error"], "lpips_mean": scores.get("mean"),
            "frame_psnr_db": metrics["frame_psnr_db"],
        }
        if length == 8:
            results[str(length)]["sheet"] = contact_sheet(work / f"b{bits}-L08-sheet.jpg", sources, decoded_by_method["packed"])
    independence = frozen.source_independence(session["quantized"], session["decoder"], embed[:4], sources[:4])
    if not independence["independent"]:
        raise PartFailed("reconstruction depends on the source tensor")
    if not session["setup_exact"]:
        raise PartFailed("decoder rebuilt from the setup package differs from the quantized decoder")
    best_saving = max(row["saving_vs_packed"] for r in results.values() for row in r["packets"] if row["method"] == "delta-zlib")
    return {
        "bit_depth": bits, "checkpoint": session["meta"], "architecture": session["dims"], "imports": session["imports"],
        "decoder_setup": session["setup"], "decoder_setup_exact": session["setup_exact"], "segments": results,
        "source_independence": independence, "temporal_delta_best_saving": best_saving,
        "temporal_gate": "merits a later temporal-code study" if best_saving >= 0.10 else "below the 10% gate",
        "embedding_quantizer": "hnerv_utils.quant_tensor per independent segment", "peak_memory_mib": torch.cuda.max_memory_reserved() / 2**20,
        "rate_note": "latent-only bytes exclude decoder weights; setup-inclusive totals add them once per cut",
    }


def _blocked(reason: str) -> dict[str, Any]:
    raise PartBlocked(reason)


def run_latent(stage: Stage, parts: list[str]) -> None:
    lpips = Lpips()
    session: dict[str, Any] = {}
    for part in parts:
        if part.endswith("-b4") and stage.parts.get(part.replace("-b4", "-b6"), {}).get("status") != "passed":
            stage.run_part(part, lambda: _blocked("6-bit correctness gate did not pass"))
            continue
        minimum = OPTIONAL_MIN_SECONDS if part.endswith("-b4") or not part.startswith("f001c3") else 60
        stage.run_part(part, lambda: latent_case(stage, part, session, lpips), minimum_seconds=minimum)


# ----------------------------------------------------------------- results

def compact(kind: str, parts: dict[str, dict[str, Any]], outputs: dict[str, Any]) -> dict[str, Any]:
    """Short per-part record for the decision event and validator checks."""
    rows = {}
    for name, record in parts.items():
        row: dict[str, Any] = {"status": record["status"], "seconds": record.get("seconds")}
        if record.get("reason"):
            row["reason"] = record["reason"][:400]
        if record["status"] == "passed":
            if kind == "codec" and "rungs" in record:
                row["rungs"] = {k: {m: v.get(m) for m in ("file_bytes", "kbps", "mean_frame_psnr_db", "pooled_mse_psnr_db", "lpips_mean", "temporal_reconstruction_error")} for k, v in record["rungs"].items()}
            elif kind == "codec":
                row.update(_arm_summary(record))
                row["frame_psnr_db"] = [round(v, 2) for v in record.get("frame_psnr_db", [])]
            elif kind == "drift":
                row.update({k: record[k] for k in ("one_stream", "peak_memory_mib")})
                row["four_reset_streams"] = {k: v for k, v in record["four_reset_streams"].items() if k != "per_segment"}
            elif kind == "latent":
                row.update({"bit_depth": record["bit_depth"], "setup_bytes": record["decoder_setup"]["bytes"],
                            "embed_channels": record["architecture"]["embed_dim"], "temporal_gate": record["temporal_gate"],
                            "best_delta_saving": record["temporal_delta_best_saving"]})
                row["segments"] = {L: {"psnr": v["mean_frame_psnr_db"], "lpips": v["lpips_mean"],
                                       "bytes": {p["method"]: p["latent_only_bytes"] for p in v["packets"]}} for L, v in record["segments"].items()}
            else:
                row["detail"] = {k: v for k, v in record.items() if k not in ("status", "seconds", "reuse_table")}
                if "reuse_table" in record:
                    row["reuse_table"] = record["reuse_table"]
        rows[name] = row
    return {"kind": kind, "parts": rows, "outputs": outputs, "citable": False}


def publish(stage: Stage, summary: dict[str, Any]) -> None:
    if not os.environ.get("PS_JOB_DIR"):
        return
    from experiments.jobs import monitor

    text = json.dumps(summary, sort_keys=True, default=str, allow_nan=False)
    monitor.publish_progress(f"{os.environ.get('PS_STAGE', 'stage')}-result", len(stage.parts), decision=text)
    time.sleep(DECISION_SETTLE_SECONDS)


def run(kind: str, args: argparse.Namespace) -> int:
    parts = parse_parts(kind, args.parts)
    if kind != "inventory" and not args.manifest:
        raise SystemExit(f"{kind} requires --manifest selected-inputs.json")
    stage = Stage(kind, float(args.stage_seconds), args.manifest)
    try:
        {"inventory": run_inventory, "codec": run_codec, "drift": run_drift, "latent": run_latent}[kind](stage, parts)
    finally:
        stage.close()
    summary = compact(kind, stage.parts, stage.outputs)
    summary["ledger_seconds"] = round(time.time() - stage.ledger.started, 1)
    summary["provenance"] = {k: v for k, v in core.job_provenance().items() if k in ("job_id", "stage", "code_revision", "host", "gpu_uuid", "gpu_name")}
    result = {"kind": kind, "parts_requested": parts, "manifests": stage.manifest_paths, "summary": summary,
              "parts": stage.parts, "outputs": stage.outputs, "provenance": core.job_provenance(),
              "label": "diagnostic smoke; not paper evidence", "citable": False}
    core.write_json_new(stage.path("result.json"), result)
    publish(stage, summary)
    return 0


# --------------------------------------------------------------- validator

def validate(kind: str) -> tuple[bool, dict[str, Any]]:
    root = core.stage_dir()
    checks: dict[str, Any] = {}
    path = root / "result.json"
    if not path.is_file():
        return False, {"result_present": False}
    result = core.read_json_bounded(path)
    checks["kind_matches"] = result.get("kind") == kind
    provenance = result.get("provenance") or {}
    checks["provenance"] = bool(provenance.get("code_revision")) and len(provenance.get("code_revision") or "") == 40 and bool(provenance.get("gpu_uuid"))
    checks["non_citable"] = result.get("citable") is False
    statuses = {name: record.get("status") for name, record in result.get("parts", {}).items()}
    checks["parts_requested_ran"] = sorted(statuses) == sorted(result.get("parts_requested", [])) or set(result.get("parts_requested", [])) <= set(statuses)
    checks["all_parts_passed"] = bool(statuses) and all(status == "passed" for status in statuses.values())
    checks["metrics_finite"] = finite_tree(result.get("parts"))
    if kind == "inventory":
        selected = result.get("outputs", {}).get("selected_inputs") or {}
        manifest = Path(selected.get("path", ""))
        checks["manifest_identity"] = manifest.is_file() and core.sha256_file(manifest, timeout=30)["sha256"] == selected.get("sha256")
    if kind in ("codec", "drift"):
        cases = [r for n, r in result.get("parts", {}).items() if r.get("status") == "passed" and "container_bytes" in r]
        checks["decode_independent"] = all(r.get("decode_independent") for r in cases)
        checks["container_sizes"] = all(Path(r["container"]).stat().st_size == r["container_bytes"] for r in cases)
    if kind == "latent":
        cases = [r for r in result.get("parts", {}).values() if r.get("status") == "passed"]
        checks["source_independent"] = all(r["source_independence"]["independent"] for r in cases)
        checks["pixels_identical"] = all(s["pixels_identical_across_methods_and_direct"] for r in cases for s in r["segments"].values())
    passed = all(value is True for value in checks.values())
    return passed, {"assertions": checks, "summary": result.get("summary")}


def run_validator(kind: str) -> int:
    passed, checks = validate(kind)
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        Path(target).write_text(json.dumps({"passed": passed, "checks": checks}, sort_keys=True, default=str))
    print(json.dumps({"passed": passed, "assertions": checks.get("assertions")}))
    return 0 if passed else 1


# --------------------------------------------------------------- summarize

def summarize(records: list[Path], output: Path) -> int:
    """B5 tables from saved `ps-fleet status`/`events` JSON. No GPU, local only."""
    rows = []
    for path in records:
        value = json.loads(path.read_text())
        for event in value.get("events", []) if isinstance(value, dict) else []:
            decision = ((event.get("status") or {}).get("progress") or {}).get("decision")
            if decision and event.get("kind", "").startswith("decision:"):
                rows.append({"job": value.get("job_id"), "event": event.get("id"), **json.loads(decision)})
        gate = value.get("gate") if isinstance(value, dict) else None
        if gate and (gate.get("validation") or {}).get("checks", {}).get("summary"):
            rows.append({"job": value.get("job_id"), "event": "gate", **gate["validation"]["checks"]["summary"]})
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"{len(rows)} stage summaries -> {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)
    for kind in KINDS:
        command = commands.add_parser(kind)
        command.add_argument("--parts", required=True)
        command.add_argument("--stage-seconds", type=float, required=True)
        command.add_argument("--manifest", type=Path, action="append", default=[])
    check = commands.add_parser("validate")
    check.add_argument("--kind", choices=KINDS, required=True)
    report = commands.add_parser("summarize")
    report.add_argument("records", type=Path, nargs="+")
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "validate":
        return run_validator(args.kind)
    if args.action == "summarize":
        return summarize(args.records, args.output)
    return run(args.action, args)


if __name__ == "__main__":
    raise SystemExit(main())
