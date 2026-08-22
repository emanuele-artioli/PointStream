"""Animate-Anyone: temporal pose-to-video.

The shipped ``finetuned_tennis`` checkpoint was trained from
``assets/dataset/pointstream_aa_meta.json``: **7 broadcast matches, 114 tracks**,
not a general human model. Every score it posts is scoped to that tennis set.
The registry summary still says "single tennis match"; that wording is stale —
the training meta is the source of truth. A full retrain is out of scope.

This wrapper loads the Moore-AnimateAnyone pipeline against a local profile
(default: ``~/Models/AnimateAnyone/profiles/finetuned_tennis``) and consumes
``ConditioningBundle`` pose *images*, not raw keypoints. ``scripts/eval_checkpoint.py``
still has no ``animate-anyone`` entry in ``ARCH_CHOICES`` — that script is owned
elsewhere; the required change is reported with this stream, not made here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.components.generation._numpy import as_chw, as_hwc
from src.components.generation.base import BaseFrameGenerator
from src.components.generation.pose import letterbox_from_bbox, letterbox_image
from src.contracts.capabilities import CONDITION_APPEARANCE, CONDITION_POSE
from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams

TENNIS_MATCH_FINETUNE_CAVEAT = (
    "This checkpoint was fine-tuned on PointStream tennis tracks listed in "
    "assets/dataset/pointstream_aa_meta.json (7 broadcast matches, 114 tracks), "
    "not a general human model. The registry summary still says 'single tennis "
    "match'; that is stale — the training meta is the source of truth."
)

FINETUNE_META = "assets/dataset/pointstream_aa_meta.json"
FINETUNE_MATCHES = (
    "alcaraz_highlights",
    "alcaraz_perricard",
    "alcaraz_ruud",
    "djokovic_federer",
    "djokovic_zverev",
    "federer_djokovic",
    "sinner_alcaraz",
)

REQUIRED_PROFILE_ENTRIES = (
    "stable-diffusion-v1-5",
    "sd-vae-ft-mse",
    "image_encoder",
    "denoising_unet.pth",
    "reference_unet.pth",
    "pose_guider.pth",
    "motion_module.pth",
)

_DEFAULT_PROFILE_CANDIDATES = (
    Path.home() / "Models" / "AnimateAnyone" / "profiles" / "finetuned_tennis",
    Path("assets") / "animate-anyone" / "profiles" / "finetuned_tennis",
)


def resolve_checkpoint(checkpoint: str | Path | None = None) -> Path:
    """Resolve a Moore-AnimateAnyone profile directory.

    ``checkpoint`` may be a profile root. If omitted, the canonical tennis
    fine-tune locations are searched. Missing required files are named.

    Raises:
        FileNotFoundError: No candidate exists, or a named directory is incomplete.
    """
    if checkpoint is not None:
        root = Path(checkpoint).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"animate-anyone checkpoint is not a directory: {root}. "
                + TENNIS_MATCH_FINETUNE_CAVEAT
            )
        _require_profile_entries(root)
        return root

    existing = [path.expanduser().resolve() for path in _DEFAULT_PROFILE_CANDIDATES if path.exists()]
    if not existing:
        searched = ", ".join(str(path) for path in _DEFAULT_PROFILE_CANDIDATES)
        raise FileNotFoundError(
            "animate-anyone profile was not found. Pass checkpoint=... or place "
            f"the tennis fine-tune under one of: {searched}. "
            + TENNIS_MATCH_FINETUNE_CAVEAT
        )
    root = existing[0]
    _require_profile_entries(root)
    return root


def _require_profile_entries(root: Path) -> None:
    missing = [name for name in REQUIRED_PROFILE_ENTRIES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            "animate-anyone profile is missing required entries: "
            f"{', '.join(missing)}. Directory checked: {root}. "
            + TENNIS_MATCH_FINETUNE_CAVEAT
        )


class AnimateAnyoneGenerator(BaseFrameGenerator):
    """Sequence generator. Temporal capability is declared, not inferred."""

    required = (CONDITION_POSE, CONDITION_APPEARANCE)
    caveat = TENNIS_MATCH_FINETUNE_CAVEAT

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        guidance: float = 7.5,
        checkpoint: str | None = None,
        runtime: Any = None,
    ) -> None:
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.checkpoint = checkpoint
        self._runtime = runtime
        self._pipeline: Any = None
        self._pipeline_device: str | None = None
        self.last_run: dict[str, Any] = {}

    def generate_sequence(
        self,
        conditioning: Sequence[ConditioningBundle],
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> Sequence[np.ndarray]:
        if not conditioning:
            raise ValueError("generate_sequence needs at least one ConditioningBundle.")
        for bundle in conditioning:
            bundle.require(*self.required)
            bundle.validate_shapes()
        if self._runtime is not None:
            output = self._runtime(
                list(conditioning), seed=seed, device=device, params=params
            )
            return tuple(as_chw(frame) for frame in output)
        return self._run_runtime(tuple(conditioning), seed=seed, device=device, params=params)

    def _generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        # Single-frame path still letterboxes through the shared geometry so a
        # per-frame call and a length-1 sequence cannot disagree about layout.
        prepared = self._prepare(conditioning, params)
        if self._runtime is not None:
            output = self._runtime(
                [conditioning], seed=seed, device=device, params=params, prepared=prepared
            )
            frame = output[0] if isinstance(output, (list, tuple)) else output
            return as_chw(frame)
        return self._run_runtime((conditioning,), seed=seed, device=device, params=params)[0]

    def _prepare(
        self, conditioning: ConditioningBundle, params: GenerationParams
    ) -> dict[str, Any]:
        """Letterbox appearance and pose independently onto the same canvas.

        The coding task feeds a keyframe as appearance and a *later* pose.
        Those crops have different sizes on every probe clip. Resizing the
        pose onto the appearance canvas (the previous path) puts the skeleton
        in the keyframe's geometry and then scores against the later frame's
        geometry — the model never sees a pose that matches the target.

        Each image keeps its own aspect and is fitted with ``fit_to_canvas``.
        Same-size inputs (self-reconstruction) still share one layout.
        """
        canvas_width, canvas_height = self.canvas_size(params)
        appearance = as_hwc(conditioning.appearance)
        src_h, src_w = appearance.shape[:2]
        appearance_box = letterbox_from_bbox(
            conditioning.bbox, src_w, src_h, canvas_width, canvas_height
        )
        pose = as_hwc(conditioning.pose)
        pose_h, pose_w = pose.shape[:2]
        pose_box = letterbox_from_bbox(None, pose_w, pose_h, canvas_width, canvas_height)
        return {
            "appearance": letterbox_image(appearance, appearance_box),
            "pose": letterbox_image(pose, pose_box),
            "letterbox": appearance_box,
            "pose_letterbox": pose_box,
            "caveat": TENNIS_MATCH_FINETUNE_CAVEAT,
        }

    def _run_runtime(
        self,
        conditioning: Sequence[ConditioningBundle],
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> Sequence[np.ndarray]:
        import time

        prepared = [self._prepare(bundle, params) for bundle in conditioning]
        width, height = self.canvas_size(params)
        steps = params.steps if params.steps is not None else self.steps
        guidance = params.guidance_scale if params.guidance_scale is not None else self.guidance

        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError("Pillow is required for the animate-anyone runtime.") from exc

        resolved_device = _resolve_torch_device(device)
        _reset_cuda_peak(resolved_device)
        started = time.perf_counter()
        pipeline = self._load_pipeline(device)
        reference_rgb = prepared[0]["appearance"][..., :3]
        pose_rgb = [item["pose"][..., :3] for item in prepared]
        reference = Image.fromarray(reference_rgb)
        pose_images = [Image.fromarray(item) for item in pose_rgb]
        feed = _reference_feed_stats(reference_rgb, pose_rgb[0])
        ref_unet_probe = _install_reference_unet_probe(pipeline)
        restore_unet = ref_unet_probe.pop("_restore", None)

        import torch

        try:
            generator = torch.Generator(device=resolved_device).manual_seed(int(seed))
            with torch.no_grad():
                output = pipeline(
                    reference,
                    pose_images,
                    width,
                    height,
                    len(pose_images),
                    int(steps),
                    float(guidance),
                    generator=generator,
                ).videos
        finally:
            if restore_unet is not None:
                restore_unet()
        wall_s = time.perf_counter() - started
        frames = _videos_to_chw(output)
        scheduler_info = _scheduler_info(pipeline)
        self.last_run = {
            "wall_s": wall_s,
            "peak_vram_bytes": (
                _cuda_peak_bytes(resolved_device)
                if str(resolved_device).startswith("cuda")
                else None
            ),
            "n_frames": len(frames),
            "width": width,
            "height": height,
            "steps": int(steps),
            "guidance": float(guidance),
            "caveat": TENNIS_MATCH_FINETUNE_CAVEAT,
            "checkpoint": str(resolve_checkpoint(self.checkpoint)),
            "scheduler": scheduler_info,
            "reference_feed": feed,
            "reference_unet": ref_unet_probe,
            "appearance_letterbox": _letterbox_info(prepared[0]["letterbox"]),
            "pose_letterbox": _letterbox_info(prepared[0]["pose_letterbox"]),
        }
        if len(frames) != len(conditioning):
            raise RuntimeError(
                "animate-anyone returned "
                f"{len(frames)} frames for {len(conditioning)} bundles. "
                + TENNIS_MATCH_FINETUNE_CAVEAT
            )
        return tuple(frames)

    def _load_pipeline(self, device: Device) -> Any:
        resolved_device = _resolve_torch_device(device)
        if self._pipeline is not None and self._pipeline_device == resolved_device:
            return self._pipeline

        model_root = resolve_checkpoint(self.checkpoint)
        import torch

        try:
            import diffusers.utils
            from diffusers import AutoencoderKL, DDIMScheduler
            from omegaconf import OmegaConf
            from transformers import CLIPVisionModelWithProjection

            diffusers.utils.USE_PEFT_BACKEND = True
            from animate_anyone.models.pose_guider import PoseGuider
            from animate_anyone.models.unet_2d_condition import UNet2DConditionModel
            from animate_anyone.models.unet_3d import UNet3DConditionModel
            from animate_anyone.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "animate-anyone runtime dependencies are missing. "
                "Moore-AnimateAnyone must be importable in this environment. "
                + TENNIS_MATCH_FINETUNE_CAVEAT
            ) from exc

        import animate_anyone as animate_anyone_pkg

        package_root = Path(str(animate_anyone_pkg.__file__)).resolve().parent
        config_path = package_root / "configs" / "inference" / "inference_v2.yaml"
        infer_config = OmegaConf.load(str(config_path))

        dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32
        vae = AutoencoderKL.from_pretrained(str(model_root / "sd-vae-ft-mse")).to(
            device=resolved_device, dtype=dtype
        )
        reference_unet = UNet2DConditionModel.from_pretrained(
            str(model_root / "stable-diffusion-v1-5"),
            subfolder="unet",
        ).to(device=resolved_device, dtype=dtype)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            str(model_root / "stable-diffusion-v1-5"),
            str(model_root / "motion_module.pth"),
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
        ).to(device=resolved_device, dtype=dtype)
        pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
            device=resolved_device, dtype=dtype
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(model_root / "image_encoder")
        ).to(device=resolved_device, dtype=dtype)

        sched_container = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        if not isinstance(sched_container, dict):
            raise ValueError("animate-anyone inference config 'noise_scheduler_kwargs' must be a mapping")
        sched_kwargs = cast(dict[str, Any], sched_container)
        scheduler = DDIMScheduler(**sched_kwargs)

        denoising_unet.load_state_dict(
            torch.load(str(model_root / "denoising_unet.pth"), map_location="cpu", weights_only=True),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(str(model_root / "reference_unet.pth"), map_location="cpu", weights_only=True),
            strict=False,
        )
        pose_guider.load_state_dict(
            torch.load(str(model_root / "pose_guider.pth"), map_location="cpu", weights_only=True),
            strict=False,
        )

        pipe = Pose2VideoPipeline(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        pipe = pipe.to(device=resolved_device, dtype=dtype)
        self._pipeline = pipe
        self._pipeline_device = resolved_device
        return pipe


def _resolve_torch_device(device: Device) -> str:
    import torch

    text = "cpu" if device is None else str(device)
    if text.startswith("cuda") and torch.cuda.is_available():
        return text if ":" in text else "cuda"
    return "cpu"


def _cuda_index(device: str) -> int:
    import torch

    index = torch.device(device).index
    return 0 if index is None else int(index)


def _reset_cuda_peak(device: str) -> None:
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    # reset_peak_memory_stats(0) raises "Invalid device argument 0: did you
    # call init?" if nothing has touched CUDA yet in this process.
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(_cuda_index(device))


def _cuda_peak_bytes(device: str) -> int:
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(_cuda_index(device)))


def _letterbox_info(box: Any) -> dict[str, Any]:
    return {
        "source_width": int(box.source_width),
        "source_height": int(box.source_height),
        "scaled_width": int(box.scaled_width),
        "scaled_height": int(box.scaled_height),
        "offset_x": int(box.offset_x),
        "offset_y": int(box.offset_y),
        "scale": float(box.scale),
    }


def _reference_feed_stats(reference_rgb: np.ndarray, pose_rgb: np.ndarray) -> dict[str, Any]:
    """What the pipeline is about to be handed, before any model call."""
    ref = np.asarray(reference_rgb)
    pose = np.asarray(pose_rgb)
    return {
        "reference_mean": float(ref.mean()),
        "reference_max": int(ref.max()) if ref.size else 0,
        "pose_mean": float(pose.mean()),
        "pose_max": int(pose.max()) if pose.size else 0,
        "reference_equals_pose": bool(ref.shape == pose.shape and np.array_equal(ref, pose)),
        "reference_is_blank": bool(ref.size == 0 or int(ref.max()) == 0),
        "pose_is_blank": bool(pose.size == 0 or int(pose.max()) == 0),
        "reference_shape": [int(n) for n in ref.shape],
        "pose_shape": [int(n) for n in pose.shape],
    }


def _scheduler_info(pipeline: Any) -> dict[str, Any]:
    scheduler = getattr(pipeline, "scheduler", None)
    if scheduler is None:
        return {"class": None}
    config = getattr(scheduler, "config", scheduler)
    return {
        "class": type(scheduler).__name__,
        "prediction_type": getattr(config, "prediction_type", None),
        "timestep_spacing": getattr(config, "timestep_spacing", None),
        "rescale_betas_zero_snr": getattr(config, "rescale_betas_zero_snr", None),
        "steps_offset": getattr(config, "steps_offset", None),
        "beta_schedule": getattr(config, "beta_schedule", None),
        "clip_sample": getattr(config, "clip_sample", None),
    }


def _install_reference_unet_probe(pipeline: Any) -> dict[str, Any]:
    """Record that ReferenceNet saw a non-zero latent, then restore the forward.

    A blank or dropped reference shows up as a near-zero latent. We hook one
    forward, not the training loop — this is a feed check, not a tuner.
    """
    unet = getattr(pipeline, "reference_unet", None)
    captured: dict[str, Any] = {"called": False}
    if unet is None or not callable(getattr(unet, "forward", None)):
        captured["error"] = "reference_unet missing"
        return captured
    original = unet.forward

    def _probed(*args: Any, **kwargs: Any) -> Any:
        sample = kwargs.get("sample", args[0] if args else None)
        captured["called"] = True
        if sample is not None:
            import torch

            tensor = sample.detach() if isinstance(sample, torch.Tensor) else torch.as_tensor(sample)
            captured["sample_shape"] = [int(n) for n in tensor.shape]
            captured["sample_abs_mean"] = float(tensor.float().abs().mean().cpu())
            captured["sample_is_zero"] = bool(float(tensor.float().abs().max().cpu()) == 0.0)
        return original(*args, **kwargs)

    unet.forward = _probed
    captured["_restore"] = lambda: setattr(unet, "forward", original)
    return captured


def _videos_to_chw(video_tensor: Any) -> list[np.ndarray]:
    import torch

    if not isinstance(video_tensor, torch.Tensor):
        video_tensor = torch.as_tensor(video_tensor)
    if video_tensor.ndim == 5:
        video_tensor = video_tensor[0]
    if video_tensor.ndim != 4:
        raise ValueError(f"Expected a 4-D video tensor, got shape {tuple(video_tensor.shape)}.")
    if int(video_tensor.shape[0]) in {1, 3}:
        frames = video_tensor.permute(1, 2, 3, 0)
    else:
        frames = video_tensor.permute(0, 2, 3, 1)
    if float(torch.min(frames)) < 0.0:
        frames = (frames + 1.0) * 0.5
    frames = frames.clamp(0.0, 1.0)
    rgb = (frames.detach().cpu().numpy() * 255.0).astype(np.uint8)
    return [as_chw(frame) for frame in rgb]
