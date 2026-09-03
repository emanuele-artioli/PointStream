"""Durable per-chunk checkpoints for a long ``run()``.

A PointStream point can sit in one ``run()`` for more than an hour. The
per-point JSON cannot resume a killed encoder, so each finished chunk is
written here before the next one starts.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts.lattice import ART_DELIVERED, ART_QUALITY
from src.pipeline.encoder.encoder import SOURCE
from src.pipeline.reconstruction.device import DeviceDecision
from src.pipeline.reconstruction.quality import Closeness, QualityReport, RegionScore
from src.pipeline.reconstruction.reconstruct import ReconstructionResult
from src.runner.accounting import SizesBytes
from src.runner.stages import _delivered_frames


def sizes_from_dict(data: dict[str, Any]) -> SizesBytes:
    return SizesBytes(
        source=int(data["source"]),
        residual=int(data.get("residual", 0)),
        panorama=int(data.get("panorama", 0)),
        actor_reference=int(data.get("actor_reference", 0)),
        metadata=int(data.get("metadata", 0)),
        transport_total=int(data["transport_total"]),
        raw_parts=tuple(str(item) for item in (data.get("raw_parts") or ())),
    )


def quality_from_dict(data: dict[str, Any]) -> QualityReport:
    return QualityReport(
        closeness=Closeness(**data["closeness"]),
        scoped=tuple(RegionScore(**item) for item in data["scoped"]),
        enforced=tuple(data["enforced"]),
    )


def publish(directory: Path, destination: Path) -> None:
    """Flush and hash all files before atomically publishing the snapshot."""
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(directory))] = file_digest(path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    with (directory / "done").open("w") as handle:
        json.dump(hashes, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    # Directory metadata must reach disk before the commit rename.
    for path in [p for p in directory.rglob("*") if p.is_dir()] + [directory]:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    os.replace(directory, destination)
    fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_snapshot(directory: Path) -> None:
    try:
        hashes = json.loads((directory / "done").read_text())
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("legacy or empty commit record")
        for name, digest in hashes.items():
            path = directory / name
            if path.resolve().is_relative_to(directory.resolve()) is False:
                raise ValueError("invalid checkpoint path")
            if file_digest(path) != digest:
                raise ValueError(f"checksum mismatch: {name}")
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"incomplete or corrupt checkpoint {directory}: {exc}") from exc


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_background(root: Path, state: dict[str, Any] | None) -> None:
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    (pending / "meta.json").write_text(json.dumps(_jsonable_background(state)))
    if state is not None:
        _save_arrays(pending / "background", state)
    publish(pending, root / "prepared")


def load_background(root: Path) -> dict[str, Any] | None:
    directory = root / "prepared"
    verify_snapshot(directory)
    state = json.loads((directory / "meta.json").read_text())
    return _load_arrays(directory / "background", state) if state is not None else None


def chunk_dir(root: Path, index: int) -> Path:
    return root / f"chunk_{index:02d}"


def completed_indices(root: Path) -> tuple[int, ...]:
    if not root.is_dir():
        return ()
    found = sorted(
        int(path.name.split("_")[1])
        for path in root.glob("chunk_*")
        if path.is_dir() and (path / "done").is_file()
    )
    if found != list(range(len(found))):
        raise SystemExit(
            f"{root} has completed chunks {found}; a gap cannot be resumed. "
            "Use a new output directory."
        )
    return tuple(found)


def save_chunk(
    root: Path,
    index: int,
    chunk: Any,
    *,
    stage_seconds: dict[str, float],
    background_state: dict[str, Any] | None,
    background_chunk_index: int,
) -> Path:
    dest = chunk_dir(root, index)
    root.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise ValueError(f"refusing to overwrite checkpoint {dest}")
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    target = dest
    dest = pending
    source = np.asarray(chunk.bag[SOURCE])
    delivered = _delivered_frames(chunk.bag[ART_DELIVERED])
    np.save(dest / "source.npy", source)
    np.save(dest / "delivered.npy", delivered)
    np.save(dest / "frames.npy", np.asarray(chunk.frames))
    np.save(dest / "encoder_frames.npy", np.asarray(chunk.encoder_frames))
    np.save(dest / "reconstruction.npy", np.asarray(chunk.reconstruction.frames))
    mask = chunk.reconstruction.object_mask
    if mask is not None:
        np.save(dest / "object_mask.npy", np.asarray(mask))
    payload = {
        "index": index,
        "stage_seconds": stage_seconds,
        "sizes": chunk.sizes.as_dict(),
        "background_chunk_index": background_chunk_index,
        "background_state": _jsonable_background(background_state),
        "quality": asdict(chunk.quality),
        "delivered_quality": asdict(chunk.delivered_quality),
        "reconstruction_quality": asdict(chunk.reconstruction.quality),
        "reconstruction_path": chunk.reconstruction.path,
        "reconstruction_device": asdict(chunk.reconstruction.device),
        "delivered_byte_count": int(chunk.bag[ART_DELIVERED].get("byte_count", chunk.sizes.transport_total)),
    }
    (dest / "meta.json").write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    if background_state is not None:
        _save_arrays(dest / "background", background_state)
    publish(pending, target)
    return target


def load_chunk(
    root: Path, index: int
) -> tuple[Any, dict[str, float], dict[str, Any] | None, int]:
    from src.runner.run import ChunkResult
    dest = chunk_dir(root, index)
    verify_snapshot(dest)
    source = np.load(dest / "source.npy")
    delivered = np.load(dest / "delivered.npy")
    frames = np.load(dest / "frames.npy")
    encoder_frames = np.load(dest / "encoder_frames.npy")
    mask_path = dest / "object_mask.npy"
    object_mask = np.load(mask_path) if mask_path.is_file() else None
    meta = json.loads((dest / "meta.json").read_text())
    quality = quality_from_dict(meta["quality"])
    delivered_quality = quality_from_dict(meta["delivered_quality"])
    reconstruction = ReconstructionResult(
        frames=np.load(dest / "reconstruction.npy", allow_pickle=False),
        quality=quality_from_dict(meta["reconstruction_quality"]),
        path=meta["reconstruction_path"],
        device=DeviceDecision(**meta["reconstruction_device"]),
        object_mask=object_mask,
    )
    bag = {
        SOURCE: source,
        ART_DELIVERED: {"frames": delivered, "byte_count": meta["delivered_byte_count"]},
        ART_QUALITY: delivered_quality,
    }
    chunk = ChunkResult(
        frames=frames,
        encoder_frames=encoder_frames,
        reconstruction=reconstruction,
        quality=quality,
        delivered_quality=delivered_quality,
        sizes=sizes_from_dict(meta["sizes"]),
        symmetry=chunk_symmetry_from_arrays(encoder_frames, frames),
        bag=bag,
    )
    background = meta.get("background_state")
    if background is not None:
        background = _load_arrays(dest / "background", background)
    return chunk, dict(meta.get("stage_seconds") or {}), background, int(
        meta.get("background_chunk_index", index + 1)
    )


def chunk_symmetry_from_arrays(encoder: np.ndarray, client: np.ndarray):
    from src.pipeline.reconstruction.quality import measure_symmetry

    return measure_symmetry(encoder, client)


def _json_default(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": True, "hex": value.hex()}
    if isinstance(value, np.ndarray):
        return {"__array__": True}
    raise TypeError(f"cannot JSON-encode {type(value)}")


def _jsonable_background(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    out = dict(state)
    transmitter = dict(state.get("transmitter") or {})
    transmitter.pop("originals", None)
    transmitter.pop("reconstructions", None)
    transmitter["payloads"] = [
        {"__bytes__": True, "hex": bytes(item).hex()} for item in transmitter.get("payloads") or []
    ]
    out["transmitter"] = transmitter
    return out


def _save_arrays(directory: Path, state: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    transmitter = state.get("transmitter") or {}
    originals = transmitter.get("originals") or []
    reconstructions = transmitter.get("reconstructions") or []
    if originals:
        np.save(directory / "originals.npy", np.stack([np.asarray(item) for item in originals]))
    if reconstructions:
        np.save(
            directory / "reconstructions.npy",
            np.stack([np.asarray(item) for item in reconstructions]),
        )


def _load_arrays(directory: Path, state: dict[str, Any]) -> dict[str, Any]:
    out = dict(state)
    transmitter = dict(state.get("transmitter") or {})
    payloads = []
    for item in transmitter.get("payloads") or []:
        if isinstance(item, dict) and item.get("__bytes__"):
            payloads.append(bytes.fromhex(item["hex"]))
        else:
            payloads.append(bytes(item))
    transmitter["payloads"] = payloads
    originals_path = directory / "originals.npy"
    reconstructions_path = directory / "reconstructions.npy"
    if originals_path.is_file():
        transmitter["originals"] = [np.asarray(frame) for frame in np.load(originals_path)]
    if reconstructions_path.is_file():
        transmitter["reconstructions"] = [
            np.asarray(frame) for frame in np.load(reconstructions_path)
        ]
    out["transmitter"] = transmitter
    return out
