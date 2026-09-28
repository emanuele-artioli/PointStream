"""SAM 3.1 multiplex video segmentation with auditable causal/offline policies.

The implementation imports the pinned Meta predictor only after validating a
local source checkout and checkpoint. It never downloads either artifact.
Offline callers may propagate in both directions; runtime callers are restricted
to forward-only propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import types
import uuid
from typing import Any, Literal

import numpy as np
from PIL import Image

from src.components.detection.types import Detection, is_person
from src.contracts.observation import EstimatorProvenance, ObservationStatus

Role = Literal["player", "racket"]
Propagation = Literal["forward", "backward", "both"]
Policy = Literal["offline_bidirectional", "offline_causal", "runtime_causal"]

DEFAULT_TEXT = {"player": "tennis player", "racket": "tennis racket"}
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


def _configure_sdpa_backend(torch_module: Any) -> str:
    """Allow PyTorch's efficient/math SDPA kernels when Flash Attention is unsupported."""
    if not torch_module.cuda.is_available():
        return "native_sdpa"
    capability = torch_module.cuda.get_device_capability()
    if capability[0] >= 8:
        return "native_flash"

    attention = torch_module.nn.attention
    original = attention.sdpa_kernel
    if getattr(original, "_pointstream_sdpa_fallback", False):
        return "efficient_then_math_fallback"
    backends = attention.SDPBackend

    @wraps(original)
    def compatible_sdpa_kernel(selected: Any, set_priority: bool = False) -> Any:
        selected_items = list(selected) if isinstance(selected, (list, tuple)) else [selected]
        if selected_items == [backends.FLASH_ATTENTION]:
            return original(
                [backends.EFFICIENT_ATTENTION, backends.MATH],
                set_priority=True,
            )
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
    """Shared SAM 3.1 predictor for offline preparation and runtime use."""

    def __init__(
        self,
        model_name: str = "sam3.1_multiplex.pt",
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
        self.model_name = model_name
        self.checkpoint_path = _path_from(checkpoint_path, "SAM31_CHECKPOINT")
        self.source_root = _path_from(source_root, "SAM31_SOURCE_ROOT")
        self.source_revision = source_revision or os.environ.get("SAM31_SOURCE_REVISION", "").strip()
        self.expected_checkpoint_sha256 = checkpoint_sha256 or os.environ.get(
            "SAM31_CHECKPOINT_SHA256", ""
        ).strip()
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
        predictor = self.predictor
        if predictor is None:
            raise RuntimeError("SAM3.1 predictor is not loaded")
        return predictor

    def _verify_artifacts(self) -> None:
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "SAM 3.1 checkpoint is missing. Set SAM31_CHECKPOINT to the local "
                "sam3.1_multiplex.pt file; automatic downloads are disabled."
            )
        if self.source_root is None or not self.source_root.is_dir():
            raise FileNotFoundError(
                "SAM 3.1 source checkout is missing. Set SAM31_SOURCE_ROOT to the "
                "verified source tree; automatic cloning is disabled."
            )
        if not self.source_revision:
            raise ValueError("SAM31_SOURCE_REVISION must pin the SAM 3.1 source checkout")
        try:
            actual_revision = subprocess.run(
                ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Cannot read the SAM 3.1 source revision at {self.source_root}") from exc
        if actual_revision != self.source_revision:
            raise RuntimeError(
                f"SAM31_SOURCE_REVISION={self.source_revision!r} does not match "
                f"checkout HEAD {actual_revision!r}"
            )
        status = subprocess.run(
            ["git", "-C", str(self.source_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if status:
            raise RuntimeError(
                f"SAM 3.1 source checkout at {self.source_root} is dirty; "
                "the pinned revision alone does not identify its code"
            )
        actual_hash = _sha256(self.checkpoint_path)
        if self.expected_checkpoint_sha256 and actual_hash != self.expected_checkpoint_sha256:
            raise RuntimeError(
                "SAM 3.1 checkpoint hash mismatch: expected "
                f"{self.expected_checkpoint_sha256}, found {actual_hash}"
            )
        self.model_revision = actual_revision
        self.checkpoint_hash = actual_hash

    def _load_predictor(self, builder: Any | None) -> Any:
        if self.source_root is not None:
            source = str(self.source_root)
            import sys

            if source not in sys.path:
                sys.path.insert(0, source)
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
        checkpoint_hash = self.checkpoint_hash or self.expected_checkpoint_sha256 or hashlib.sha256(
            b"injected-test-predictor"
        ).hexdigest()
        config = {
            **DEFAULT_CONFIG,
            "prob_threshold": self.prob_threshold,
            "max_num_objects": self.max_num_objects,
            "multiplex_count": self.multiplex_count,
            **self.model_options,
        }
        # Preserve the existing native-kernel identity on supported GPUs. The
        # fallback is provenance-relevant only where it changes execution.
        if self.sdpa_backend_policy == "efficient_then_math_fallback":
            config["sdpa_backend_policy"] = self.sdpa_backend_policy
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
        session_index = (role, key)
        if session_index in self.sessions:
            raise RuntimeError(f"SAM3.1 already has an active {role!r}/{key!r} session")
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("SAM3.1 session dimensions must be positive")
        self.provenance(policy)
        response = self._require_predictor().handle_request(
            {"type": "start_session", "resource_path": str(resource_path)}
        )
        session_id = str(response["session_id"])
        self.sessions[session_index] = _Session(
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
            request["bounding_boxes"] = [[
                (x0 + x1) / (2 * session.frame_width),
                (y0 + y1) / (2 * session.frame_height),
                (x1 - x0) / session.frame_width,
                (y1 - y0) / session.frame_height,
            ]]
            request["bounding_box_labels"] = [1]
        if points:
            request["points"] = [
                [float(x) / session.frame_width, float(y) / session.frame_height]
                for x, y in points
            ]
            request["point_labels"] = list(point_labels or [1] * len(points))
            request["obj_id"] = tracker_id
        response = self._require_predictor().handle_request(request)
        masks = _unpack_outputs(response.get("outputs", {}), session.frame_height, session.frame_width)
        observations: list[MaskObservation] = []
        if not masks:
            session.expected_objects.add(object_id)
        target_index = _prompt_target_index(
            masks,
            bbox=bbox,
            points=points,
            point_labels=point_labels,
        )
        for index, (tracker_id, mask, score) in enumerate(masks):
            assigned_id = object_id if index == target_index else f"{role}:{tracker_id}"
            session.expected_objects.add(assigned_id)
            session.tracker_to_object[tracker_id] = assigned_id
            session.object_to_tracker[assigned_id] = tracker_id
            observations.append(
                MaskObservation(role, assigned_id, tracker_id, frame_index, mask, score)
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
        session = self._session(role, session_key)
        self.provenance(policy)
        if frame_count <= 0:
            raise ValueError("frame_count must be positive to account for missing frames")
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
        responses: dict[tuple[int, int], tuple[np.ndarray, float | None]] = {}
        for response in self._require_predictor().handle_stream_request(request):
            frame_index = int(response["frame_index"])
            for tracker_id, mask, score in _unpack_outputs(
                response.get("outputs", {}), session.frame_height, session.frame_width
            ):
                if tracker_id not in session.tracker_to_object:
                    session.tracker_to_object[tracker_id] = f"{role}:{tracker_id}"
                    session.object_to_tracker[session.tracker_to_object[tracker_id]] = tracker_id
                    session.expected_objects.add(session.tracker_to_object[tracker_id])
                responses[(frame_index, tracker_id)] = (mask, score)
        frame_indices = range(frame_count)
        records: list[MaskObservation] = []
        for frame_index in frame_indices:
            for object_id in sorted(session.expected_objects):
                object_tracker = session.object_to_tracker.get(object_id)
                result = (
                    responses.get((frame_index, object_tracker))
                    if object_tracker is not None
                    else None
                )
                records.append(
                    MaskObservation(
                        role,
                        object_id,
                        object_tracker,
                        int(frame_index),
                        result[0] if result is not None else None,
                        result[1] if result is not None else None,
                        status=(
                            ObservationStatus.OBSERVED
                            if result is not None
                            else ObservationStatus.MISSING
                        ),
                        reason=None if result is not None else "propagation_missing_frame_object",
                    )
                )
        return tuple(records)

    def close_session(self, role: Role, *, session_key: str = "default") -> None:
        key = _session_key(session_key)
        session = self.sessions.pop((role, key), None)
        if session is None:
            return
        self._require_predictor().handle_request({"type": "close_session", "session_id": session.session_id})

    def segment(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        """Causal single-frame adapter for PointStream's registry segmenter API."""
        if is_person(detection.class_name):
            role: Role = "player"
        elif "racket" in detection.class_name.casefold():
            role = "racket"
        else:
            return None
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("SAM3.1 runtime input must be an HWC color image")
        height, width = image.shape[:2]
        with tempfile.TemporaryDirectory(prefix="pointstream-sam31-") as tmp:
            path = Path(tmp) / "00000.jpg"
            Image.fromarray(np.asarray(image[..., :3], dtype=np.uint8)).save(path, format="JPEG", quality=95)
            self.start_session(
                role,
                tmp,
                frame_width=width,
                frame_height=height,
                policy="runtime_causal",
            )
            try:
                observations = self.add_prompt(
                    role,
                    frame_index=0,
                    object_id=detection.track_id or f"{role}:frame0",
                    text=DEFAULT_TEXT[role],
                    bbox=(
                        detection.bbox.x1,
                        detection.bbox.y1,
                        detection.bbox.x2,
                        detection.bbox.y2,
                    ),
                )
                chosen = next((item.mask for item in observations if item.mask is not None), None)
                return chosen
            finally:
                self.close_session(role)

    def _session(self, role: Role, session_key: str = "default") -> _Session:
        _validate_role(role)
        try:
            return self.sessions[(role, _session_key(session_key))]
        except KeyError as exc:
            raise RuntimeError(
                f"SAM3.1 {role!r}/{session_key!r} session has not been started"
            ) from exc


def _unpack_outputs(
    outputs: Any,
    height: int,
    width: int,
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
        raise ValueError(
            f"SAM3.1 returned {len(ids)} tracker IDs for {len(masks_array)} masks"
        )
    scores = outputs.get("out_probs")
    if scores is not None and hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    scores_array = np.asarray(scores).reshape(-1) if scores is not None else np.array([])
    result: list[tuple[int, np.ndarray, float | None]] = []
    for index, (tracker_id, mask) in enumerate(zip(ids, masks_array)):
        binary = np.asarray(mask) > 0
        if binary.shape != (height, width):
            binary = np.asarray(
                Image.fromarray(binary.astype(np.uint8) * 255).resize(
                    (width, height), Image.Resampling.NEAREST
                )
            ) > 0
        score = float(scores_array[index]) if index < len(scores_array) else None
        result.append((tracker_id, binary.astype(np.uint8), score))
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
    for index, (tracker_id, raw_mask, _score) in enumerate(masks):
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
        scored.append((score, -int(tracker_id), index))
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


def _validate_role(role: str) -> None:
    if role not in {"player", "racket"}:
        raise ValueError(f"SAM3.1 role must be player or racket, got {role!r}")


def _install_compatible_session_start(predictor: Any) -> None:
    """Adapt the verified multiplex predictor's current session initializer.

    The pinned SAM3.1 checkout used by the existing smoke exposes a
    ``start_session`` implementation with an obsolete ``model.init_state``
    signature. Its request dispatcher still routes through that method. Keep
    the compatibility bridge narrowly scoped to predictors that expose the
    concrete state store and initializer; test doubles and future compatible
    APIs are left alone.
    """
    model = getattr(predictor, "model", None)
    state_store = getattr(predictor, "_all_inference_states", None)
    initializer = getattr(model, "init_state", None)
    if not callable(initializer) or not isinstance(state_store, dict):
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


__all__ = ["MaskObservation", "Sam31SequenceSegmenter"]
