"""Lossless per-clip instance masks: the one format every consumer reads.

A clip is stored as a COCO-RLE JSON stream (schema ``pointstream.maps.coco_rle.v1``,
the format the demo maps gallery already wrote), compressed with zstd when the
``zstandard`` package is present and zlib otherwise. The loader detects which.

This module must import inside the SAM 3.1 environment too, so it depends only
on numpy and the standard library; pycocotools and zstandard are optional.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - the SAM env may lack zstandard
    zstd = None  # type: ignore[assignment]

SCHEMA = "pointstream.maps.coco_rle.v1"
MASKS_NAME = "masks.rle"
_LEGACY_NAMES = ("masks.rle.zst", "payload.bin")
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    """COCO RLE of an HxW mask; compressed counts when pycocotools is installed."""
    binary = np.asfortranarray(np.asarray(mask) > 0, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"mask must be HxW, got {binary.shape}")
    height, width = binary.shape
    try:
        from pycocotools import mask as mask_util

        counts = mask_util.encode(binary)["counts"]
        return {"size": [height, width], "counts": counts.decode("ascii")}
    except ImportError:
        flat = binary.ravel(order="F")
        change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        bounds = np.concatenate(([0], change, [flat.size]))
        runs = np.diff(bounds).tolist() if flat.size else []
        if flat.size and flat[0]:
            runs.insert(0, 0)
        return {"size": [height, width], "counts": runs}


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    """Boolean HxW mask from a COCO RLE (list or compressed-string counts)."""
    height, width = (int(v) for v in rle["size"])
    counts = rle["counts"]
    if isinstance(counts, (bytes, str)):
        text = counts.decode("ascii") if isinstance(counts, bytes) else counts
        try:
            from pycocotools import mask as mask_util

            decoded = mask_util.decode({"size": [height, width], "counts": text.encode("ascii")})
            return np.asarray(decoded) > 0
        except ImportError:
            counts = _string_to_runs(text)
    runs = np.asarray(counts, dtype=np.int64)
    if (runs < 0).any() or int(runs.sum()) != height * width:
        raise ValueError("RLE runs do not cover the mask exactly")
    values = np.arange(runs.size) % 2 == 1
    flat = np.repeat(values, runs)
    return flat.reshape((height, width), order="F")


def _string_to_runs(text: str) -> list[int]:
    """COCO compressed counts (maskApi.c rleFrString)."""
    runs: list[int] = []
    i = 0
    while i < len(text):
        x = k = 0
        more = 1
        while more and i < len(text):
            c = ord(text[i]) - 48
            i += 1
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            k += 1
            if not more and c & 0x10:
                x |= -1 << (5 * k)
        if len(runs) > 2:
            x += runs[-2]
        runs.append(int(x))
    return runs


def compress(raw: bytes) -> bytes:
    if zstd is not None:
        return zstd.ZstdCompressor(level=3).compress(raw)
    return zlib.compress(raw, level=6)


def decompress(blob: bytes) -> bytes:
    if blob.startswith(_ZSTD_MAGIC):
        if zstd is None:
            raise RuntimeError("this mask file is zstd-compressed; install zstandard")
        return zstd.ZstdDecompressor().decompress(blob)
    return zlib.decompress(blob)


@dataclass(frozen=True)
class Instance:
    """One object in one frame. ``track_id`` is stable within a clip and class.

    Dataset labels also carry ``provenance``, the tier the mask comes from
    (``human``, ``interpolated``, …; docs/resources.md), and ``label``, the
    dataset's own name for the object when it is finer than ``class_name``.
    """

    class_name: str
    track_id: int
    score: float
    rle: dict[str, Any]
    bbox: tuple[float, float, float, float]
    provenance: str | None = None
    label: str | None = None

    @classmethod
    def from_mask(
        cls,
        class_name: str,
        track_id: int,
        mask: np.ndarray,
        score: float = 1.0,
        *,
        provenance: str | None = None,
        label: str | None = None,
    ) -> Instance | None:
        binary = np.asarray(mask) > 0
        if not binary.any():
            return None
        ys, xs = np.nonzero(binary)
        bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        return cls(
            str(class_name), int(track_id), float(score), encode_rle(binary), bbox, provenance, label
        )

    def mask(self) -> np.ndarray:
        return decode_rle(self.rle)


@dataclass
class ClipMasks:
    """Every instance mask of one clip, in the domain's class order.

    Empty frames hold no instances; nothing ever becomes a whole-frame fill.
    Later classes in ``classes`` paint over earlier ones in ``labels``.

    ``labelled`` is None for a segmenter's output, where every frame was
    segmented. For dataset labels it flags the frames that carry labels: an
    unlabelled frame is unknown, not empty, and is never scored.
    """

    classes: tuple[str, ...]
    height: int
    width: int
    fps: float
    frames: list[list[Instance]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    labelled: list[bool] | None = None

    def __len__(self) -> int:
        return len(self.frames)

    def ensure_frames(self, count: int) -> None:
        while len(self.frames) < count:
            self.frames.append([])
        if self.labelled is not None and len(self.labelled) < count:
            self.labelled.extend([False] * (count - len(self.labelled)))

    def add(
        self,
        index: int,
        class_name: str,
        track_id: int,
        mask: np.ndarray,
        score: float = 1.0,
        *,
        provenance: str | None = None,
        label: str | None = None,
    ) -> None:
        if class_name not in self.classes:
            raise ValueError(f"class {class_name!r} is not one of {self.classes}")
        if np.asarray(mask).shape != (self.height, self.width):
            raise ValueError(f"mask shape {np.asarray(mask).shape} != {(self.height, self.width)}")
        self.ensure_frames(index + 1)
        instance = Instance.from_mask(
            class_name, track_id, mask, score, provenance=provenance, label=label
        )
        if instance is not None:
            self.frames[index].append(instance)

    def labelled_frames(self) -> list[int]:
        """Indices of the frames that carry labels (all of them for a segmenter)."""
        if self.labelled is None:
            return list(range(len(self.frames)))
        return [index for index, flag in enumerate(self.labelled) if flag]

    def class_mask(self, index: int, class_name: str) -> np.ndarray:
        out = np.zeros((self.height, self.width), dtype=bool)
        for inst in self.frames[index]:
            if inst.class_name == class_name:
                out |= inst.mask()
        return out

    def foreground(self, index: int) -> np.ndarray:
        out = np.zeros((self.height, self.width), dtype=bool)
        for inst in self.frames[index]:
            out |= inst.mask()
        return out

    def match(
        self, index: int, bbox: tuple[float, float, float, float], class_name: str | None = None
    ) -> np.ndarray | None:
        """The instance mask (of ``class_name``, if given) whose box best overlaps ``bbox``."""
        x0, y0, x1, y1 = (float(v) for v in bbox)
        best, best_iou = None, 0.0
        for inst in self.frames[index] if index < len(self.frames) else ():
            if class_name is not None and inst.class_name != class_name:
                continue
            ix0, iy0, ix1, iy1 = inst.bbox
            inter = max(0.0, min(x1, ix1) - max(x0, ix0)) * max(0.0, min(y1, iy1) - max(y0, iy0))
            union = (x1 - x0) * (y1 - y0) + (ix1 - ix0) * (iy1 - iy0) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best, best_iou = inst, iou
        return best.mask() if best is not None else None

    def labels(self, index: int) -> np.ndarray:
        """uint8 map: 0 background, ``i + 1`` for ``classes[i]``."""
        out = np.zeros((self.height, self.width), dtype=np.uint8)
        for code, name in enumerate(self.classes, start=1):
            out[self.class_mask(index, name)] = code
        return out

    def to_doc(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "height": self.height,
            "width": self.width,
            "fps": self.fps,
            "classes": list(self.classes),
            "mask_empty_policy": "skip_frame",
            "meta": self.meta,
            "frames": [
                {
                    "index": index,
                    **(
                        {"labelled": bool(self.labelled[index])}
                        if self.labelled is not None
                        else {}
                    ),
                    "instances": [_instance_doc(self.classes, inst) for inst in instances],
                }
                for index, instances in enumerate(self.frames)
            ],
        }

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> ClipMasks:
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"unknown mask schema {doc.get('schema')!r}")
        classes = tuple(str(name) for name in doc["classes"])
        clip = cls(
            classes,
            int(doc["height"]),
            int(doc["width"]),
            float(doc["fps"]),
            meta=dict(doc.get("meta") or {}),
        )
        records = doc.get("frames") or []
        if any("labelled" in record for record in records):
            clip.labelled = [False] * (max(int(r["index"]) for r in records) + 1)
        for record in records:
            index = int(record["index"])
            clip.ensure_frames(index + 1)
            if clip.labelled is not None:
                clip.labelled[index] = bool(record.get("labelled", False))
            for order, raw in enumerate(record.get("instances") or []):
                name = str(raw.get("class_name") or classes[int(raw["class_id"])])
                clip.frames[index].append(
                    Instance(
                        name,
                        int(raw.get("track_id", order)),
                        float(raw.get("score", 1.0)),
                        raw["rle"],
                        tuple(float(v) for v in raw["bbox"][:4]),  # type: ignore[arg-type]
                        raw.get("provenance"),
                        raw.get("label"),
                    )
                )
        return clip

    def save(self, path: Path | str) -> Path:
        """Write ``masks.rle`` into a directory, or to an explicit file path."""
        target = Path(path)
        if target.suffix == "" or target.is_dir():
            target = target / MASKS_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            compress(json.dumps(self.to_doc(), separators=(",", ":")).encode("utf-8"))
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> ClipMasks:
        return cls.from_doc(json.loads(decompress(mask_file(path).read_bytes()).decode("utf-8")))


def _instance_doc(classes: tuple[str, ...], inst: Instance) -> dict[str, Any]:
    record: dict[str, Any] = {
        "class_id": classes.index(inst.class_name),
        "class_name": inst.class_name,
        "track_id": inst.track_id,
        "score": inst.score,
        "bbox": list(inst.bbox),
        "rle": inst.rle,
    }
    if inst.provenance is not None:
        record["provenance"] = inst.provenance
    if inst.label is not None:
        record["label"] = inst.label
    return record


def mask_file(path: Path | str) -> Path:
    """The mask payload in ``path``, accepting the legacy maps-gallery names."""
    target = Path(path)
    if target.is_file():
        return target
    for name in (MASKS_NAME, *_LEGACY_NAMES):
        if (target / name).is_file():
            return target / name
    raise FileNotFoundError(f"no {MASKS_NAME} under {target}")


__all__ = ["SCHEMA", "ClipMasks", "Instance", "decode_rle", "encode_rle", "mask_file"]
