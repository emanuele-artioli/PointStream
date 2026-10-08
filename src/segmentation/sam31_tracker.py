"""SAM 3.1's tracker, prompted with masks: semi-supervised video object segmentation.

`Sam31SequenceSegmenter` (`sam31.py`) drives Meta's multiplex *predictor*, which
takes text, boxes and points. Mask prompts are not part of that API at the
pinned commit: ``Sam3BasePredictor.add_mask`` raises for the multiplex model.
Inside it, though, every detection reaches the tracker as a mask
(``Sam3MultiplexBase._tracker_add_new_objects`` calls the tracker's
``add_new_masks``), and the tracker is a SAM 2-style video model with
``init_state``, ``add_new_masks`` and ``propagate_in_video``.

`Sam31MaskTracker` builds that tracker on its own with sam3's builder
(``build_sam3_multiplex_video_model``: the multiplex tracker plus a vision-only
copy of the shared tri-head backbone) and loads the weights of the same
checkpoint, renamed from the combined model's layout (`remap_checkpoint`): the
tracker from ``tracker.model.*`` and the backbone from
``detector.backbone.vision_backbone.*``, the module the combined model feeds the
tracker from. Every key must load.

A mask prompt on a frame makes it a conditioning frame for every object; an
empty mask says the object is absent there. Propagation conditions each frame
on the nearest conditioning frames before and after it (up to
``max_cond_frames_in_attn``) and on the frames tracked just before it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.segmentation.sam31 import (
    DEFAULT_CHECKPOINT_SHA256,
    DEFAULT_SOURCE_REVISION,
    _configure_sdpa_backend,
    _sha256,
    installed_sam3_revision,
)

TRACKER_PREFIX = "tracker.model."
BACKBONE_PREFIX = "detector.backbone.vision_backbone."
#: Builder arguments; the rest are sam3's own defaults for the multiplex tracker.
BUILD_OPTIONS = {"multiplex_count": 16, "use_fa3": False, "use_rope_real": False, "compile": False}


def remap_checkpoint(state: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The combined checkpoint's tracker and backbone weights, named as the standalone tracker's.

    Returns the renamed weights and the keys left out (the detector's own heads,
    its text encoder, and the tracker's interactive heads stay under their
    prefixes only if the standalone model has them; see `Sam31MaskTracker`).
    """
    out: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in state.items():
        if key.startswith(TRACKER_PREFIX):
            out[key[len(TRACKER_PREFIX):]] = value
        elif key.startswith(BACKBONE_PREFIX):
            out["backbone.vision_backbone." + key[len(BACKBONE_PREFIX):]] = value
        else:
            skipped.append(key)
    return out, skipped


def _drop_detector_head(model: Any) -> None:
    """Compute only the tracker's backbone heads.

    The demo tracker's ``_get_image_feature`` asks the backbone for all three
    heads. The vision-only backbone returns the detector's (SAM 3) head
    flattened into the top level of its output, which the tracker's
    ``forward_image`` then iterates as if it were a head, and fails (job
    ``20261007T203825Z-0e10810c``, smoke). The tracker reads only the
    interactive and propagation heads (``_prepare_backbone_features``), as
    upstream's own ``_prepare_backbone_features_per_frame`` asks for.
    """
    original = model.forward_image

    def forward_image(img_batch: Any, *, need_sam3_out: bool = False, need_interactive_out: bool = False,
                      need_propagation_out: bool = False) -> Any:
        return original(img_batch, need_sam3_out=False, need_interactive_out=need_interactive_out,
                        need_propagation_out=need_propagation_out)

    model.forward_image = forward_image


class Sam31MaskTracker:
    """The pinned SAM 3.1 tracker on CUDA, prompted with binary masks."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        checkpoint_sha256: str = DEFAULT_CHECKPOINT_SHA256,
        source_revision: str = DEFAULT_SOURCE_REVISION,
        model: Any | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.load_report: dict[str, Any] = {}
        if model is not None:  # tests inject a stand-in
            self.model = model
            return
        installed = installed_sam3_revision()
        if installed != source_revision:
            raise RuntimeError(f"installed sam3 is at {installed!r}, not the pinned {source_revision!r}")
        actual = _sha256(self.checkpoint_path)
        if actual != checkpoint_sha256:
            raise RuntimeError(f"SAM 3.1 checkpoint hash {actual} is not the pinned {checkpoint_sha256}")
        import torch
        from sam3.model_builder import build_sam3_multiplex_video_model

        sdpa = _configure_sdpa_backend(torch)
        model = build_sam3_multiplex_video_model(
            checkpoint_path=None, load_from_HF=False, device="cuda", **BUILD_OPTIONS
        )
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        renamed, skipped = remap_checkpoint(state)
        result = model.load_state_dict(renamed, strict=False)
        expected = set(model.state_dict())
        self.load_report = {
            "source_revision": installed,
            "checkpoint_sha256": actual,
            "sdpa_backend_policy": sdpa,
            "build_options": BUILD_OPTIONS,
            "loaded_keys": len(renamed),
            "model_keys": len(expected),
            "missing_keys": sorted(result.missing_keys),
            "unexpected_keys": sorted(result.unexpected_keys),
            "skipped_checkpoint_prefixes": sorted({".".join(k.split(".")[:2]) for k in skipped}),
            "image_size": int(model.image_size),
            "max_cond_frames_in_attn": int(getattr(model, "max_cond_frames_in_attn", -1)),
        }
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"SAM 3.1 tracker weights do not match: {len(result.missing_keys)} missing "
                f"({result.missing_keys[:5]}), {len(result.unexpected_keys)} unexpected ({result.unexpected_keys[:5]})"
            )
        _drop_detector_head(model)
        self.model = model.cuda().eval()

    def load_frames(self, frames_dir: Path | str) -> tuple[Any, int, int]:
        """Frames ``0.png .. n-1.png`` resized and normalised as sam3's own loader does (on CPU)."""
        from sam3.model.io_utils import load_video_frames

        return load_video_frames(
            video_path=str(frames_dir), image_size=int(self.model.image_size), offload_video_to_cpu=True,
            img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5), async_loading_frames=False,
        )

    def track(
        self,
        images: Any,
        height: int,
        width: int,
        prompts: Mapping[int, Mapping[int, np.ndarray]],
        *,
        start: int = 0,
        frames: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, dict[int, tuple[np.ndarray, float]]]]:
        """Track from ``start`` through ``frames`` frames of ``images`` (to the end by default), backward with ``reverse``.

        ``prompts`` maps a frame to ``{object id: HxW bool mask}``; every object
        must be given on every prompted frame (an empty mask: absent). Yields
        ``(frame, {object id: (mask, presence probability)})`` for every frame
        tracked, prompted frames included.
        """
        import torch

        if not prompts:
            raise ValueError("SAM 3.1 tracking needs at least one prompted frame")
        objects = sorted(next(iter(prompts.values())))
        if any(sorted(masks) != objects for masks in prompts.values()):
            raise ValueError("every prompted frame must give a mask for every object")
        model = self.model
        count = len(images)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = model.init_state(video_height=height, video_width=width, num_frames=count,
                                     offload_video_to_cpu=True)
            state["images"] = images
            for frame in sorted(prompts):
                stack = np.stack([np.asarray(prompts[frame][obj], dtype=bool) for obj in objects])
                masks = torch.from_numpy(stack).to("cuda")
                model.add_new_masks(inference_state=state, frame_idx=int(frame), obj_ids=list(objects),
                                    masks=masks, add_mask_to_memory=True)
            model.propagate_in_video_preflight(state, run_mem_encoder=True)
            limit = count if frames is None else frames - 1
            for out in model.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=limit,
                                                reverse=reverse, tqdm_disable=True):
                frame, obj_ids, _low, video_res = out[:4]
                scores = out[4] if len(out) > 4 else None
                binary = (video_res[:, 0] > 0.0).cpu().numpy()
                probs = (torch.sigmoid(scores.float()).reshape(-1).cpu().numpy()
                         if scores is not None else np.ones(len(obj_ids)))
                yield int(frame), {int(o): (binary[i], float(probs[i])) for i, o in enumerate(obj_ids)}

    def peak_gpu_mib(self) -> float:
        import torch

        return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def runtime() -> dict[str, Any]:
    """Device facts for the result record."""
    import torch

    device = torch.cuda.current_device()
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


__all__ = ["BUILD_OPTIONS", "Sam31MaskTracker", "remap_checkpoint", "runtime"]
