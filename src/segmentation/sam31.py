"""SAM 3.1 Object Multiplex: the reference segmenter.

Two halves share this file:

* `Sam31SequenceSegmenter` loads the model. It verifies the pinned ``sam3``
  code (the installed distribution, or a git checkout named by
  ``SAM31_SOURCE_ROOT``) and the checkpoint, never downloads either, and
  exposes Meta's session API (start / prompt / propagate / close) with explicit
  MISSING records.
* `Sam31Segmenter` extracts frames, launches this module as a worker process
  (the same interpreter by default) and reads back `ClipMasks`.

Each text prompt resets the multiplex tracker, so every class gets its own
session and propagate pass. Offline callers may propagate both ways; runtime
callers are forward-only. This module imports only numpy and PIL at the top so
the worker starts quickly.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from src.segmentation.masks import ClipMasks

Role = str
Propagation = Literal["forward", "backward", "both"]
Policy = Literal["offline_bidirectional", "offline_causal", "runtime_causal"]


class ObservationStatus(str, Enum):
    OBSERVED = "observed"
    MISSING = "missing"
    INVALID = "invalid"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class EstimatorProvenance:
    """Which model, revision, checkpoint, config and policy produced the masks."""

    name: str
    model_revision: str
    checkpoint_sha256: str
    config_sha256: str
    policy: str

    def __post_init__(self) -> None:
        if not all((self.name, self.model_revision, self.checkpoint_sha256, self.config_sha256)):
            raise ValueError(
                "estimator provenance requires model, revision, checkpoint, and config hashes"
            )
        if self.policy not in {"offline_bidirectional", "offline_causal", "runtime_causal"}:
            raise ValueError(f"unsupported perception policy {self.policy!r}")


DEFAULT_SOURCE_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
DEFAULT_CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
HF_CHECKPOINT = (
    "~/.cache/huggingface/hub/models--facebook--sam3.1/snapshots/"
    "daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt"
)
DEFAULT_CONFIG = {
    "model": "sam3.1-multiplex",
    "max_num_objects": 16,
    "multiplex_count": 16,
    "use_fa3": False,
    "use_rope_real": False,
    "compile": False,
    "warm_up": False,
    "async_loading_frames": False,
    "default_output_prob_thresh": 0.35,
}


def default_checkpoint() -> Path | None:
    """SAM31_CHECKPOINT, then Models/SAM, then the Hugging Face cache snapshot."""
    env = os.environ.get("SAM31_CHECKPOINT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    candidates = [Path(HF_CHECKPOINT).expanduser()]
    from src.segmentation.storage import models_root

    candidates.insert(0, models_root() / "SAM" / "sam3.1_multiplex.pt")
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _configure_sdpa_backend(torch_module: Any) -> str:
    """Allow PyTorch's efficient/math SDPA kernels when Flash Attention is unsupported."""
    if not torch_module.cuda.is_available():
        return "native_sdpa"
    if torch_module.cuda.get_device_capability()[0] >= 8:
        return "native_flash"
    attention = torch_module.nn.attention
    original = attention.sdpa_kernel
    if getattr(original, "_pointstream_sdpa_fallback", False):
        return "efficient_then_math_fallback"
    backends = attention.SDPBackend

    @wraps(original)
    def compatible_sdpa_kernel(selected: Any, set_priority: bool = False) -> Any:
        items = list(selected) if isinstance(selected, (list, tuple)) else [selected]
        if items == [backends.FLASH_ATTENTION]:
            return original([backends.EFFICIENT_ATTENTION, backends.MATH], set_priority=True)
        return original(selected, set_priority=set_priority)

    compatible_sdpa_kernel._pointstream_sdpa_fallback = True  # type: ignore[attr-defined]
    attention.sdpa_kernel = compatible_sdpa_kernel
    return "efficient_then_math_fallback"


@dataclass(frozen=True)
class MaskObservation:
    role: Role
    object_id: str
    tracker_id: int | None
    frame_index: int
    mask: np.ndarray | None
    score: float | None = None
    status: ObservationStatus = ObservationStatus.OBSERVED
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0 or not self.object_id:
            raise ValueError("mask observations require a non-negative frame and object identity")
        if self.status is ObservationStatus.OBSERVED:
            if self.mask is None:
                raise ValueError("observed SAM3.1 output must contain a mask")
            if np.asarray(self.mask).ndim != 2:
                raise ValueError("SAM3.1 masks must have two spatial dimensions")
        elif self.mask is not None:
            raise ValueError("missing or invalid SAM3.1 observations cannot contain a mask")
        if self.status is not ObservationStatus.OBSERVED and not self.reason:
            raise ValueError("missing or invalid SAM3.1 observations need a reason")


@dataclass
class _Session:
    role: Role
    session_id: str
    resource_path: str
    frame_width: int
    frame_height: int
    key: str = "default"
    tracker_to_object: dict[int, str] = field(default_factory=dict)
    object_to_tracker: dict[str, int] = field(default_factory=dict)
    expected_objects: set[str] = field(default_factory=set)


class Sam31SequenceSegmenter:
    """Pinned SAM 3.1 multiplex predictor with per-role sessions (SAM env only)."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        source_root: str | Path | None = None,
        source_revision: str | None = None,
        checkpoint_sha256: str | None = None,
        predictor: Any | None = None,
        builder: Any | None = None,
        prob_threshold: float = 0.35,
        max_num_objects: int = 16,
        multiplex_count: int = 16,
        model_options: dict[str, Any] | None = None,
    ) -> None:
        if not 0.0 <= prob_threshold <= 1.0:
            raise ValueError("prob_threshold must be in [0, 1]")
        if max_num_objects <= 0 or multiplex_count <= 0:
            raise ValueError("SAM3.1 object capacities must be positive")
        self.checkpoint_path = _path_from(checkpoint_path, "SAM31_CHECKPOINT")
        self.source_root = _path_from(source_root, "SAM31_SOURCE_ROOT")
        self.source_revision = (
            source_revision
            or os.environ.get("SAM31_SOURCE_REVISION", "").strip()
            or DEFAULT_SOURCE_REVISION
        )
        self.expected_checkpoint_sha256 = (
            checkpoint_sha256 or os.environ.get("SAM31_CHECKPOINT_SHA256", "").strip()
        )
        self.prob_threshold = float(prob_threshold)
        self.max_num_objects = int(max_num_objects)
        self.multiplex_count = int(multiplex_count)
        self.model_options = dict(model_options or {})
        self.sessions: dict[tuple[Role, str], _Session] = {}
        self.predictor = predictor
        self.model_revision: str | None = None
        self.checkpoint_hash: str | None = None
        self.sdpa_backend_policy = "native_sdpa"
        if self.predictor is None:
            self._verify_artifacts()
            self.predictor = self._load_predictor(builder)

    def _require_predictor(self) -> Any:
        if self.predictor is None:
            raise RuntimeError("SAM3.1 predictor is not loaded")
        return self.predictor

    def _verify_artifacts(self) -> None:
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "SAM 3.1 checkpoint is missing. Set SAM31_CHECKPOINT to the local "
                "sam3.1_multiplex.pt file; automatic downloads are disabled."
            )
        if not self.source_revision:
            raise ValueError("SAM31_SOURCE_REVISION must pin the SAM 3.1 source")
        if self.source_root is None:
            installed = installed_sam3_revision()
            if installed != self.source_revision:
                raise RuntimeError(
                    f"installed sam3 is at {installed!r}, not the pinned {self.source_revision!r}"
                )
            self._verify_checkpoint(installed)
            return
        if not self.source_root.is_dir():
            raise FileNotFoundError(
                f"SAM 3.1 source checkout {self.source_root} is missing; automatic cloning is disabled."
            )
        git = ["git", "-C", str(self.source_root)]
        try:
            actual_revision = subprocess.run(
                [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=300
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                f"Cannot read the SAM 3.1 source revision at {self.source_root}"
            ) from exc
        if actual_revision != self.source_revision:
            raise RuntimeError(
                f"SAM31_SOURCE_REVISION={self.source_revision!r} does not match checkout HEAD {actual_revision!r}"
            )
        status = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        ).stdout.strip()
        if status:
            raise RuntimeError(
                f"SAM 3.1 source checkout at {self.source_root} is dirty; "
                "the pinned revision alone does not identify its code"
            )
        self._verify_checkpoint(actual_revision)

    def _verify_checkpoint(self, revision: str) -> None:
        assert self.checkpoint_path is not None
        actual_hash = _sha256(self.checkpoint_path)
        if self.expected_checkpoint_sha256 and actual_hash != self.expected_checkpoint_sha256:
            raise RuntimeError(
                f"SAM 3.1 checkpoint hash mismatch: expected {self.expected_checkpoint_sha256}, found {actual_hash}"
            )
        self.model_revision = revision
        self.checkpoint_hash = actual_hash

    def _load_predictor(self, builder: Any | None) -> Any:
        if self.source_root is not None and str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        if builder is None:
            import torch

            self.sdpa_backend_policy = _configure_sdpa_backend(torch)
            try:
                from sam3.model_builder import build_sam3_multiplex_video_predictor
            except ImportError as exc:
                raise RuntimeError(
                    "Pinned SAM 3.1 checkout lacks build_sam3_multiplex_video_predictor"
                ) from exc
            builder = build_sam3_multiplex_video_predictor
        assert self.checkpoint_path is not None
        options: dict[str, Any] = {
            **DEFAULT_CONFIG,
            "max_num_objects": self.max_num_objects,
            "multiplex_count": self.multiplex_count,
            "default_output_prob_thresh": self.prob_threshold,
            "checkpoint_path": str(self.checkpoint_path),
            **self.model_options,
        }
        signature = inspect.signature(builder)
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            options = {key: value for key, value in options.items() if key in signature.parameters}
        try:
            predictor = builder(**options)
            _install_compatible_session_start(predictor)
            return predictor
        except Exception as exc:
            raise RuntimeError("Failed to initialize pinned SAM 3.1 multiplex predictor") from exc

    def provenance(self, policy: Policy) -> EstimatorProvenance:
        revision = self.model_revision or self.source_revision or "injected-test-predictor"
        checkpoint_hash = (
            self.checkpoint_hash
            or self.expected_checkpoint_sha256
            or hashlib.sha256(b"injected-test-predictor").hexdigest()
        )
        config = {
            **DEFAULT_CONFIG,
            "prob_threshold": self.prob_threshold,
            "max_num_objects": self.max_num_objects,
            "multiplex_count": self.multiplex_count,
            **self.model_options,
        }
        # The fallback is provenance-relevant only where it changes execution.
        if self.sdpa_backend_policy == "efficient_then_math_fallback":
            config["sdpa_backend_policy"] = self.sdpa_backend_policy
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return EstimatorProvenance(
            name="sam3.1_multiplex",
            model_revision=revision,
            checkpoint_sha256=checkpoint_hash,
            config_sha256=config_hash,
            policy=policy,
        )

    def start_session(
        self,
        role: Role,
        resource_path: str | Path,
        *,
        frame_width: int,
        frame_height: int,
        policy: Policy,
        session_key: str = "default",
    ) -> str:
        _validate_role(role)
        key = _session_key(session_key)
        if (role, key) in self.sessions:
            raise RuntimeError(f"SAM3.1 already has an active {role!r}/{key!r} session")
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("SAM3.1 session dimensions must be positive")
        self.provenance(policy)
        response = self._require_predictor().handle_request(
            {"type": "start_session", "resource_path": str(resource_path)}
        )
        session_id = str(response["session_id"])
        self.sessions[(role, key)] = _Session(
            role=role,
            session_id=session_id,
            resource_path=str(resource_path),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            key=key,
        )
        return session_id

    def add_prompt(
        self,
        role: Role,
        *,
        frame_index: int,
        object_id: str,
        text: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        points: list[tuple[float, float]] | None = None,
        point_labels: list[int] | None = None,
        tracker_id: int | None = None,
        session_key: str = "default",
    ) -> tuple[MaskObservation, ...]:
        session = self._session(role, session_key)
        if frame_index < 0 or not object_id:
            raise ValueError("frame_index and object_id are required for SAM3.1 prompts")
        if text is None and bbox is None and not points:
            raise ValueError("SAM3.1 prompt needs text, a box, or points")
        if points and (text is not None or bbox is not None):
            raise ValueError("SAM3.1 point prompts cannot be combined with text or box prompts")
        if points and tracker_id is None:
            raise ValueError("SAM3.1 point prompts require an explicit tracker_id")
        if tracker_id is not None and type(tracker_id) is not int:
            raise ValueError("SAM3.1 tracker_id must be an integer")
        if point_labels is not None and len(point_labels) != len(points or ()):
            raise ValueError("point_labels must have one label for every point")
        request: dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session.session_id,
            "frame_index": int(frame_index),
            "output_prob_thresh": self.prob_threshold,
            "rel_coordinates": True,
        }
        if text is not None:
            request["text"] = text
        if bbox is not None:
            x0, y0, x1, y1 = map(float, bbox)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("SAM3.1 box prompt must have positive width and height")
            request["bounding_boxes"] = [
                [
                    (x0 + x1) / (2 * session.frame_width),
                    (y0 + y1) / (2 * session.frame_height),
                    (x1 - x0) / session.frame_width,
                    (y1 - y0) / session.frame_height,
                ]
            ]
            request["bounding_box_labels"] = [1]
        if points:
            request["points"] = [
                [float(x) / session.frame_width, float(y) / session.frame_height] for x, y in points
            ]
            request["point_labels"] = list(point_labels or [1] * len(points))
            request["obj_id"] = tracker_id
        response = self._require_predictor().handle_request(request)
        masks = _unpack_outputs(
            response.get("outputs", {}), session.frame_height, session.frame_width
        )
        if not masks:
            session.expected_objects.add(object_id)
        target_index = _prompt_target_index(
            masks, bbox=bbox, points=points, point_labels=point_labels
        )
        observations: list[MaskObservation] = []
        for index, (tracker, mask, score) in enumerate(masks):
            assigned_id = object_id if index == target_index else f"{role}:{tracker}"
            session.expected_objects.add(assigned_id)
            session.tracker_to_object[tracker] = assigned_id
            session.object_to_tracker[assigned_id] = tracker
            observations.append(
                MaskObservation(role, assigned_id, tracker, frame_index, mask, score)
            )
        if not observations:
            observations.append(
                MaskObservation(
                    role,
                    object_id,
                    None,
                    frame_index,
                    None,
                    status=ObservationStatus.MISSING,
                    reason="prompt_returned_no_mask",
                )
            )
        return tuple(observations)

    def iter_propagate(
        self,
        role: Role,
        *,
        policy: Policy,
        direction: Propagation,
        start_frame_index: int | None = None,
        session_key: str = "default",
    ) -> Iterator[tuple[int, int, np.ndarray, float | None]]:
        """Stream ``(frame, tracker_id, mask, score)`` without holding the clip in memory."""
        session = self._session(role, session_key)
        self.provenance(policy)
        if policy == "runtime_causal" and direction != "forward":
            raise ValueError("runtime_causal SAM3.1 propagation must be forward-only")
        if direction not in {"forward", "backward", "both"}:
            raise ValueError(f"unsupported SAM3.1 propagation direction {direction!r}")
        request: dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": session.session_id,
            "propagation_direction": direction,
            "output_prob_thresh": self.prob_threshold,
        }
        if start_frame_index is not None:
            request["start_frame_index"] = int(start_frame_index)
        for response in self._require_predictor().handle_stream_request(request):
            frame_index = int(response["frame_index"])
            for tracker, mask, score in _unpack_outputs(
                response.get("outputs", {}), session.frame_height, session.frame_width
            ):
                if tracker not in session.tracker_to_object:
                    object_id = f"{role}:{tracker}"
                    session.tracker_to_object[tracker] = object_id
                    session.object_to_tracker[object_id] = tracker
                    session.expected_objects.add(object_id)
                yield frame_index, tracker, mask, score

    def propagate(
        self,
        role: Role,
        *,
        policy: Policy,
        direction: Propagation,
        frame_count: int,
        start_frame_index: int | None = None,
        session_key: str = "default",
    ) -> tuple[MaskObservation, ...]:
        """Dense (frame x expected object) records; absent pairs are MISSING."""
        if frame_count <= 0:
            raise ValueError("frame_count must be positive to account for missing frames")
        responses = {
            (frame, tracker): (mask, score)
            for frame, tracker, mask, score in self.iter_propagate(
                role,
                policy=policy,
                direction=direction,
                start_frame_index=start_frame_index,
                session_key=session_key,
            )
        }
        session = self._session(role, session_key)
        records: list[MaskObservation] = []
        for frame_index in range(frame_count):
            for object_id in sorted(session.expected_objects):
                tracker = session.object_to_tracker.get(object_id)
                result = responses.get((frame_index, tracker)) if tracker is not None else None
                records.append(
                    MaskObservation(
                        role,
                        object_id,
                        tracker,
                        frame_index,
                        result[0] if result is not None else None,
                        result[1] if result is not None else None,
                        status=ObservationStatus.OBSERVED
                        if result is not None
                        else ObservationStatus.MISSING,
                        reason=None if result is not None else "propagation_missing_frame_object",
                    )
                )
        return tuple(records)

    def close_session(self, role: Role, *, session_key: str = "default") -> None:
        session = self.sessions.pop((role, _session_key(session_key)), None)
        if session is not None:
            self._require_predictor().handle_request(
                {"type": "close_session", "session_id": session.session_id}
            )

    def segment_frames(
        self,
        frames_dir: Path,
        concepts: dict[str, str],
        *,
        policy: Policy = "offline_bidirectional",
        fps: float = 30.0,
        on_frame: Any = None,
    ) -> ClipMasks:
        """Text-prompt every class on frame 0 and propagate through ``frames_dir``.

        ``concepts`` maps class name to prompt text, in paint order. Classes with
        identical text share one pass.
        """
        frames = sorted(Path(frames_dir).glob("*.jpg"))
        if not frames:
            raise RuntimeError(f"no frames in {frames_dir}")
        width, height = Image.open(frames[0]).size
        clip = ClipMasks(tuple(concepts), height, width, fps)
        clip.ensure_frames(len(frames))
        direction: Propagation = "both" if policy == "offline_bidirectional" else "forward"
        done: dict[str, str] = {}
        for class_name, text in concepts.items():
            if text in done:
                for index in range(len(frames)):
                    for inst in [i for i in clip.frames[index] if i.class_name == done[text]]:
                        clip.add(index, class_name, inst.track_id, inst.mask(), inst.score)
                continue
            done[text] = class_name
            self.start_session(
                class_name, frames_dir, frame_width=width, frame_height=height, policy=policy
            )
            try:
                self.add_prompt(
                    class_name, frame_index=0, object_id=f"{class_name}:prompt", text=text
                )
                for frame, tracker, mask, score in self.iter_propagate(
                    class_name, policy=policy, direction=direction
                ):
                    if 0 <= frame < len(frames):
                        clip.add(frame, class_name, tracker, mask, 1.0 if score is None else score)
                    if on_frame is not None:
                        on_frame(class_name, frame)
            except Exception as exc:
                # The pinned predictor refuses to propagate a prompt that found
                # nothing; that class is simply absent from this clip.
                if "No points are provided" not in str(exc):
                    raise
                clip.meta.setdefault("empty_classes", []).append(class_name)
            finally:
                self.close_session(class_name)
        return clip

    def _session(self, role: Role, session_key: str = "default") -> _Session:
        _validate_role(role)
        try:
            return self.sessions[(role, _session_key(session_key))]
        except KeyError as exc:
            raise RuntimeError(
                f"SAM3.1 {role!r}/{session_key!r} session has not been started"
            ) from exc


def _unpack_outputs(
    outputs: Any, height: int, width: int
) -> list[tuple[int, np.ndarray, float | None]]:
    if not isinstance(outputs, dict):
        return []
    object_ids = outputs.get("out_obj_ids", [])
    masks = outputs.get("out_binary_masks", [])
    if hasattr(object_ids, "detach"):
        object_ids = object_ids.detach().cpu().numpy()
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    ids = [int(value) for value in np.asarray(object_ids).reshape(-1)]
    masks_array = np.asarray(masks)
    if masks_array.ndim == 4 and masks_array.shape[1] == 1:
        masks_array = masks_array[:, 0]
    if masks_array.ndim == 2:
        masks_array = masks_array[None, ...]
    if masks_array.ndim != 3 or masks_array.size == 0:
        return []
    if not ids:
        ids = list(range(len(masks_array)))
    if len(ids) != len(masks_array):
        raise ValueError(f"SAM3.1 returned {len(ids)} tracker IDs for {len(masks_array)} masks")
    scores = outputs.get("out_probs")
    if scores is not None and hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    scores_array = np.asarray(scores).reshape(-1) if scores is not None else np.array([])
    result: list[tuple[int, np.ndarray, float | None]] = []
    for index, (tracker, mask) in enumerate(zip(ids, masks_array)):
        binary = np.asarray(mask) > 0
        if binary.shape != (height, width):
            binary = (
                np.asarray(
                    Image.fromarray(binary.astype(np.uint8) * 255).resize(
                        (width, height), Image.Resampling.NEAREST
                    )
                )
                > 0
            )
        score = float(scores_array[index]) if index < len(scores_array) else None
        result.append((tracker, binary.astype(np.uint8), score))
    return result


def _prompt_target_index(
    masks: list[tuple[int, np.ndarray, float | None]],
    *,
    bbox: tuple[float, float, float, float] | None,
    points: list[tuple[float, float]] | None,
    point_labels: list[int] | None,
) -> int | None:
    """Choose the returned mask best supported by explicit spatial guidance."""
    if len(masks) == 1:
        return 0
    if not masks or (bbox is None and not points):
        return None
    positive = [
        point
        for point, label in zip(points or (), point_labels or [1] * len(points or ()), strict=True)
        if int(label) > 0
    ]
    x0, y0, x1, y1 = bbox if bbox is not None else (0.0, 0.0, 0.0, 0.0)
    scored: list[tuple[float, int, int]] = []
    for index, (tracker, raw_mask, _score) in enumerate(masks):
        mask = np.asarray(raw_mask) != 0
        height, width = mask.shape
        score = 0.0
        if bbox is not None:
            bx0 = max(0, min(width, int(np.floor(x0))))
            by0 = max(0, min(height, int(np.floor(y0))))
            bx1 = max(bx0, min(width, int(np.ceil(x1))))
            by1 = max(by0, min(height, int(np.ceil(y1))))
            box = np.zeros(mask.shape, dtype=bool)
            box[by0:by1, bx0:bx1] = True
            union = int(np.count_nonzero(mask | box))
            score += 100.0 * (np.count_nonzero(mask & box) / union if union else 0.0)
        for x, y in positive:
            ix = min(width - 1, max(0, int(round(x))))
            iy = min(height - 1, max(0, int(round(y))))
            score += 1000.0 if mask[iy, ix] else 0.0
        scored.append((score, -int(tracker), index))
    best_score, _negative_id, best_index = max(scored)
    return best_index if best_score > 0 else None


def _path_from(value: str | Path | None, env_name: str) -> Path | None:
    raw = str(value).strip() if value is not None else os.environ.get(env_name, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_sam3_revision() -> str | None:
    """Commit of the installed ``sam3`` distribution, if it is intact.

    pip records the commit of a ``git+https://...@<sha>`` install in
    ``direct_url.json``; every installed file must still match its ``RECORD``
    hash. Editable or path installs have no commit and return ``None``.
    """
    import base64
    from importlib import metadata

    try:
        dist = metadata.distribution("sam3")
    except metadata.PackageNotFoundError:
        return None
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = direct.get("vcs_info", {}).get("commit_id")
    if not commit or direct.get("dir_info", {}).get("editable"):
        return None
    for entry in dist.files or ():
        if entry.hash is None:
            continue
        data = Path(str(dist.locate_file(entry))).read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.new(entry.hash.mode, data).digest()).rstrip(b"=")
        if digest.decode() != entry.hash.value:
            raise RuntimeError(f"installed sam3 file {entry} differs from its RECORD hash")
    return str(commit)


def _validate_role(role: str) -> None:
    if not isinstance(role, str) or not role.strip() or "/" in role:
        raise ValueError(f"SAM3.1 role must be a non-empty class name, got {role!r}")


def _install_compatible_session_start(predictor: Any) -> None:
    """Adapt the pinned predictor's obsolete ``start_session`` → ``init_state`` call.

    Only predictors that expose the concrete state store and initializer are
    patched; test doubles and future compatible APIs are left alone.
    """
    model = getattr(predictor, "model", None)
    state_store = getattr(predictor, "_all_inference_states", None)
    if not callable(getattr(model, "init_state", None)) or not isinstance(state_store, dict):
        return

    def compatible_start_session(
        self: Any,
        resource_path: str,
        session_id: str | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, str]:
        init_kwargs: dict[str, Any] = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        valid = inspect.signature(self.model.init_state).parameters
        state = self.model.init_state(
            **{key: value for key, value in init_kwargs.items() if key in valid}
        )
        identifier = session_id or uuid.uuid4().hex
        now = time.time()
        self._all_inference_states[identifier] = {
            "state": state,
            "session_id": identifier,
            "start_time": now,
            "last_use_time": now,
        }
        return {"session_id": identifier}

    predictor.start_session = types.MethodType(compatible_start_session, predictor)


def _session_key(value: str) -> str:
    key = str(value).strip()
    if not key or "/" in key or "\\" in key:
        raise ValueError("SAM3.1 session keys must be non-empty plain identifiers")
    return key


class Sam31Segmenter:
    """Main-env front end: frames out, SAM 3.1 worker, `ClipMasks` back."""

    name = "sam31"

    def __init__(
        self,
        *,
        python: str | Path | None = None,
        checkpoint: str | Path | None = None,
        source_root: str | Path | None = None,
        source_revision: str | None = None,
        checkpoint_sha256: str | None = None,
        prob_threshold: float = 0.35,
        policy: Policy = "offline_bidirectional",
        chunk_frames: int = 300,
    ) -> None:
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        self.chunk_frames = int(chunk_frames)
        self.python = Path(python or os.environ.get("SAM31_PYTHON") or sys.executable).expanduser()
        self.checkpoint = Path(checkpoint).expanduser() if checkpoint else default_checkpoint()
        root = source_root or os.environ.get("SAM31_SOURCE_ROOT")
        self.source_root = Path(root).expanduser() if root else None
        self.source_revision = (
            source_revision or os.environ.get("SAM31_SOURCE_REVISION") or DEFAULT_SOURCE_REVISION
        )
        self.checkpoint_sha256 = (
            checkpoint_sha256
            or os.environ.get("SAM31_CHECKPOINT_SHA256")
            or DEFAULT_CHECKPOINT_SHA256
        )
        self.prob_threshold = prob_threshold
        self.policy = policy

    def worker_command(self, request: Path, out: Path) -> list[str]:
        return [
            str(self.python),
            "-m",
            "src.segmentation.sam31",
            "--request",
            str(request),
            "--out",
            str(out),
        ]

    def segment(
        self, source: Path | str, domain: Any, *, max_frames: int | None = None
    ) -> ClipMasks:
        import shutil

        from src.segmentation.sources import REPO_ROOT, video_fps

        if self.checkpoint is None or not self.checkpoint.is_file():
            raise FileNotFoundError(
                "sam3.1_multiplex.pt is not on disk; set SAM31_CHECKPOINT (no auto-download)"
            )
        if not self.python.is_file():
            raise FileNotFoundError(
                f"SAM 3.1 interpreter not found: {self.python}; set SAM31_PYTHON"
            )
        with tempfile.TemporaryDirectory(prefix="ps-sam31-") as tmp:
            work = Path(tmp)
            request = work / "request.json"
            request.write_text(
                json.dumps(
                    {
                        "source": str(Path(source).resolve()),
                        "max_frames": max_frames,
                        "chunk_frames": self.chunk_frames,
                        "ffmpeg": shutil.which("ffmpeg"),
                        "concepts": domain.prompts_for("sam"),
                        "policy": self.policy,
                        "prob_threshold": self.prob_threshold,
                        "fps": video_fps(source),
                        "checkpoint": str(self.checkpoint),
                        "source_root": str(self.source_root) if self.source_root else None,
                        "source_revision": self.source_revision,
                        "checkpoint_sha256": self.checkpoint_sha256,
                    }
                )
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")])
            )
            result = subprocess.run(
                self.worker_command(request, work / "out"),
                capture_output=True,
                text=True,
                env=env,
                cwd=REPO_ROOT,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"SAM 3.1 worker failed ({result.returncode}):\n{result.stderr[-4000:]}"
                )
            return ClipMasks.load(work / "out")


TRACK_ID_STRIDE = 100_000


def segment_chunks(segmenter: Any, request: dict[str, Any]) -> tuple[ClipMasks, float, float]:
    """Segment a clip in windows of ``chunk_frames`` with one loaded model.

    A long clip does not fit one SAM 3.1 session, so each window is its own
    prompt-and-propagate pass. Track ids restart per window; they are offset by
    ``TRACK_ID_STRIDE`` per window so they stay unique within the clip. Returns
    the masks, segmentation seconds and frame-extraction seconds.
    """
    from dataclasses import replace

    from src.segmentation.sources import extract_jpegs

    limit = request.get("max_frames")
    chunk = int(request["chunk_frames"])
    fps = float(request["fps"])
    clip: ClipMasks | None = None
    start, window, segment_s, extract_s = 0, 0, 0.0, 0.0
    while limit is None or start < limit:
        want = chunk if limit is None else min(chunk, limit - start)
        with tempfile.TemporaryDirectory(prefix="ps-sam31-chunk-") as tmp:
            t0 = time.perf_counter()
            count = extract_jpegs(
                request["source"],
                Path(tmp),
                want,
                start=start,
                fps=fps,
                ffmpeg=request.get("ffmpeg"),
            )
            extract_s += time.perf_counter() - t0
            if count == 0:
                break
            t0 = time.perf_counter()
            part = segmenter.segment_frames(
                Path(tmp), dict(request["concepts"]), policy=request["policy"], fps=fps
            )
            segment_s += time.perf_counter() - t0
        if clip is None:
            clip = ClipMasks(part.classes, part.height, part.width, fps)
        for instances in part.frames:
            clip.frames.append(
                [replace(i, track_id=window * TRACK_ID_STRIDE + i.track_id) for i in instances]
            )
        for name in part.meta.get("empty_classes", []):
            clip.meta.setdefault("empty_windows", []).append({"start": start, "class": name})
        start += count
        window += 1
        if count < want:
            break
    if clip is None:
        raise RuntimeError(f"no frames in {request['source']}")
    clip.meta["windows"] = window
    return clip, segment_s, extract_s


def _worker(argv: list[str] | None = None) -> int:
    from src.segmentation.sources import runtime_identity, timing_summary

    parser = argparse.ArgumentParser(
        description="SAM 3.1 worker"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text())

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the SAM 3.1 env")
    load_t0 = time.perf_counter()
    segmenter = Sam31SequenceSegmenter(
        checkpoint_path=request["checkpoint"],
        source_root=request["source_root"],
        source_revision=request["source_revision"],
        checkpoint_sha256=request["checkpoint_sha256"],
        prob_threshold=float(request["prob_threshold"]),
    )
    model_load_s = time.perf_counter() - load_t0
    torch.cuda.reset_peak_memory_stats()
    masks, total_s, extract_s = segment_chunks(segmenter, request)
    estimator = segmenter.provenance(request["policy"])
    masks.meta.update(
        {
            "backend": "sam31",
            "prompts": request["concepts"],
            "policy": request["policy"],
            "prob_threshold": request["prob_threshold"],
            "model": {
                "name": estimator.name,
                "checkpoint": str(segmenter.checkpoint_path),
                "checkpoint_sha256": estimator.checkpoint_sha256,
                "source_root": str(segmenter.source_root) if segmenter.source_root else "installed",
                "source_revision": estimator.model_revision,
                "config_sha256": estimator.config_sha256,
                "sdpa_backend_policy": segmenter.sdpa_backend_policy,
            },
            "worker_runtime": runtime_identity(),
            # One propagate pass per class, so per-step latency is not meaningful;
            # throughput over the whole clip is.
            "timing": {
                **timing_summary([], model_load_s=model_load_s, total_s=total_s, frames=len(masks)),
                "frame_extract_s": round(extract_s, 3),
                "chunk_frames": int(request["chunk_frames"]),
                "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            },
        }
    )
    masks.save(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker())


__all__ = ["MaskObservation", "Policy", "Role", "Sam31SequenceSegmenter", "Sam31Segmenter"]
