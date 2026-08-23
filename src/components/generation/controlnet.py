"""ControlNet variants: canny, seg, pose, ip-adapter, multi, trajectory.

Each variant is the same class with a different ``variant`` default in the
registry. Conditioning is read from named bundle fields; the compositor no
longer string-matches the backend name to decide whether the overloaded
parameter was a pose, a mask, or a tuple.

Weights are loaded lazily on the first ``generate`` that has no injected
pipeline. Fine-tuned variants load a named checkpoint epoch — never the
last-sorted ``checkpoint-epoch-*`` directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.components.generation._numpy import as_chw, as_hwc, prepare_letterboxed
from src.components.generation.base import BaseFrameGenerator
from src.contracts.capabilities import (
    CONDITION_APPEARANCE,
    CONDITION_CANNY,
    CONDITION_MASK,
    CONDITION_MOTION_FIELD,
    CONDITION_POSE,
)
from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams

_LOGGER = logging.getLogger(__name__)

VARIANT_REQUIRES: dict[str, tuple[str, ...]] = {
    "canny": (CONDITION_CANNY, CONDITION_APPEARANCE),
    "seg": (CONDITION_MASK, CONDITION_APPEARANCE),
    "pose": (CONDITION_POSE, CONDITION_APPEARANCE),
    "pose-ref": (CONDITION_POSE, CONDITION_APPEARANCE),
    "ip-adapter": (CONDITION_APPEARANCE, CONDITION_POSE),
    "multi": (CONDITION_POSE, CONDITION_MASK, CONDITION_APPEARANCE),
    "trajectory": (CONDITION_MOTION_FIELD, CONDITION_APPEARANCE),
}

_KNOWN = frozenset(VARIANT_REQUIRES)

#: Last trained epoch of each fine-tune campaign. Hard-coded so an extra
#: checkpoint written later cannot silently change which weights we load.
#: ``ip-adapter`` is not here: the directory ``ip-adapter-controlnet`` is a
#: mislabelled segmentation ControlNet (``scripts/train_controlnet.py`` line 82
#: puts ``"ip-adapter"`` on the ``seg`` branch with ``cond_dir = None``). A
#: real IP-Adapter is the stock SD-1.5 backbone plus ``h94/IP-Adapter`` plus
#: stock OpenPose ControlNet — no tennis ControlNet epoch.
FINAL_EPOCH: Final[dict[str, int]] = {
    "pose": 10,
    "seg": 7,
    "trajectory": 10,  # same OpenPose ControlNet as the keypoint arm
}

_CONTROLNET_DIR: Final[dict[str, str]] = {
    "pose": "pose-controlnet",
    "pose-ref": "pose-ref-controlnet",
    "seg": "seg-controlnet",
    "ip-adapter": "control_v11p_sd15_openpose",
    "trajectory": "pose-controlnet",
    "canny": "control_v11p_sd15_canny",
}

_SD15_DIR: Final = "stable-diffusion-v1-5"
_IP_ADAPTER_REPO: Final = "h94/IP-Adapter"
_IP_ADAPTER_SUBFOLDER: Final = "models"
_IP_ADAPTER_WEIGHT: Final = "ip-adapter_sd15.bin"

_PROMPT: Final = "photorealistic tennis player, broadcast sports shot"


def resolve_prompt(caption: str | None) -> tuple[str, str]:
    """Return ``(prompt, source)``. ``source`` is ``caption`` or ``fallback``.

    Training read the per-track BLIP caption when it existed. Inference
    hardcoded the fallback, so every published ControlNet number ran with
    that text channel off. Empty strings count as missing.
    """
    text = (caption or "").strip()
    if text:
        return text, "caption"
    return _PROMPT, "fallback"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def weights_dir(root: Path | None = None) -> Path:
    return (root or _repo_root()) / "assets" / "weights"


def resolve_controlnet_checkpoint(
    variant: str,
    checkpoint: str | None = None,
    epoch: int | None = None,
    *,
    root: Path | None = None,
) -> tuple[Path, int | None]:
    """Return ``(controlnet_dir, epoch)`` without globbing for the newest epoch.

    Fine-tuned variants resolve to ``<base>/checkpoint-epoch-<N>``. ``N`` is
    ``epoch`` when given, otherwise ``FINAL_EPOCH[variant]``. A directory
    already named ``checkpoint-epoch-N`` is used as-is. Missing epochs raise
    with the epochs that *are* on disk — they are never replaced by a sort.

    Stock (non-finetuned) variants return the directory itself and ``epoch=None``.
    """
    if variant not in _KNOWN:
        raise ValueError(
            f"Unknown ControlNet variant {variant!r}. Known: {', '.join(sorted(_KNOWN))}."
        )
    base = Path(checkpoint) if checkpoint else weights_dir(root) / _CONTROLNET_DIR.get(
        variant, f"{variant}-controlnet"
    )
    if not base.is_absolute():
        planted = weights_dir(root) / checkpoint if checkpoint else base
        if planted.exists() or planted.is_symlink():
            base = planted
        elif not base.exists():
            base = weights_dir(root) / base.name if checkpoint is None else (weights_dir(root) / checkpoint)

    if base.name.startswith("checkpoint-epoch-"):
        parsed = int(base.name.rsplit("-", maxsplit=1)[-1])
        if epoch is not None and epoch != parsed:
            raise ValueError(
                f"checkpoint path {base} is epoch {parsed}, but epoch={epoch} was requested."
            )
        if not base.is_dir():
            raise FileNotFoundError(f"ControlNet checkpoint directory missing: {base}")
        return base.resolve(), parsed

    chosen = epoch if epoch is not None else FINAL_EPOCH.get(variant)
    if chosen is None:
        if not base.exists():
            raise FileNotFoundError(
                f"{variant}-controlnet weights not found at {base}. "
                f"Place them under {weights_dir(root)}."
            )
        return base.resolve(), None

    epoch_dir = base / f"checkpoint-epoch-{chosen}"
    if not epoch_dir.is_dir():
        available = _available_epochs(base)
        avail_txt = ", ".join(str(n) for n in available) if available else "(none)"
        raise FileNotFoundError(
            f"{variant}-controlnet epoch {chosen} not found at {epoch_dir}. "
            f"Epochs on disk: {avail_txt}. Refusing to pick another by sort order."
        )
    return epoch_dir.resolve(), chosen


def _available_epochs(base: Path) -> list[int]:
    found: list[int] = []
    if not base.is_dir():
        return found
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-epoch-"):
            try:
                found.append(int(child.name.rsplit("-", maxsplit=1)[-1]))
            except ValueError:
                continue
    return sorted(found)


def compose_pose_on_appearance(
    pose: np.ndarray,
    appearance: np.ndarray,
    *,
    threshold: int = 8,
) -> np.ndarray:
    """3-channel control: appearance where the pose canvas is black, skeleton on top.

    Keeps the OpenPose ControlNet backbone (3-channel condition) while making a
    same-track reference actually enter the condition image. Both inputs must
    already share a canvas; this function does not resize.
    """
    pose_hwc = as_hwc(pose)[..., :3]
    appearance_hwc = as_hwc(appearance)[..., :3]
    if pose_hwc.shape[:2] != appearance_hwc.shape[:2]:
        raise ValueError(
            "compose_pose_on_appearance needs pose and appearance on the same canvas, "
            f"got pose {pose_hwc.shape[:2]} vs appearance {appearance_hwc.shape[:2]}."
        )
    skeleton = pose_hwc.max(axis=2, keepdims=True) > threshold
    return np.where(skeleton, pose_hwc, appearance_hwc)


def render_trajectory_control(
    motion_field: Any,
    *,
    width: int,
    height: int,
    stride: int = 8,
) -> np.ndarray:
    """Draw sparse displacement sticks onto a black RGB canvas.

    ``motion_field`` is ``(2, H, W)`` float32, dx/dy in source pixels. A grid
    of starting points is sampled; near-zero vectors are skipped so the image
    stays sparse. Colour encodes direction (HSV hue from angle). This is the
    control image fed to the *same* OpenPose ControlNet as the keypoint arm.
    """
    import cv2

    field = np.asarray(motion_field, dtype=np.float32)
    if field.ndim != 3 or field.shape[0] != 2:
        raise ValueError(f"motion_field must be (2, H, W); got shape {tuple(field.shape)}.")
    _, src_h, src_w = field.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale_x = width / max(1, src_w)
    scale_y = height / max(1, src_h)
    mags = np.hypot(field[0], field[1])
    ys, xs = np.nonzero(mags >= 0.5)
    if ys.size == 0:
        return canvas
    grid_budget = max(32, (max(1, src_h // stride)) * (max(1, src_w // stride)))
    if ys.size > grid_budget:
        points = [
            (y, x)
            for y in range(0, src_h, stride)
            for x in range(0, src_w, stride)
            if mags[y, x] >= 0.5
        ]
    else:
        points = list(zip(ys.tolist(), xs.tolist(), strict=True))
    for y, x in points:
        dx = float(field[0, y, x])
        dy = float(field[1, y, x])
        x0 = int(round(x * scale_x))
        y0 = int(round(y * scale_y))
        x1 = int(round((x + dx) * scale_x))
        y1 = int(round((y + dy) * scale_y))
        angle = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
        hue = int(np.clip(angle * 179.0, 0, 179))
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        colour = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        rgb = (int(colour[0]), int(colour[1]), int(colour[2]))
        cv2.line(canvas, (x0, y0), (x1, y1), rgb, 2, cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 2, rgb, -1)
    return canvas


class ControlNetGenerator(BaseFrameGenerator):
    """SD-ControlNet, one class per variant via registry defaults."""

    def __init__(
        self,
        variant: str = "pose",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        strength: float = 0.65,
        guidance: float = 7.0,
        checkpoint: str | None = None,
        pipeline: Any = None,
        epoch: int | None = None,
        sd_model: str | None = None,
        ip_adapter_scale: float = 0.5,
    ) -> None:
        if variant not in _KNOWN:
            raise ValueError(
                f"Unknown ControlNet variant {variant!r}. Known: {', '.join(sorted(_KNOWN))}."
            )
        self.variant = variant
        self.required = VARIANT_REQUIRES[variant]
        self.width = width
        self.height = height
        self.steps = steps
        self.strength = strength
        self.guidance = guidance
        self.checkpoint = checkpoint
        self.epoch = epoch
        self.sd_model = sd_model
        self.ip_adapter_scale = ip_adapter_scale
        self._pipeline = pipeline
        self.loaded_checkpoint: str | None = None
        self.loaded_epoch: int | None = None
        self.last_seed: int | None = None
        self.last_prompt: str | None = None
        self.last_prompt_source: str | None = None

    def prepare(
        self, conditioning: ConditioningBundle, params: GenerationParams
    ) -> dict[str, Any]:
        """Letterbox appearance and every declared condition onto one canvas.

        Public so the rescale fix is testable without loading diffusers.
        """
        canvas_width, canvas_height = self.canvas_size(params)
        extras: dict[str, Any] = {
            "pose": conditioning.pose,
            "mask": conditioning.mask,
            "canny": conditioning.canny,
        }
        if self.variant == "trajectory" and conditioning.motion_field is not None:
            appearance = as_hwc(conditioning.appearance)
            src_h, src_w = appearance.shape[:2]
            extras["trajectory"] = render_trajectory_control(
                conditioning.motion_field, width=src_w, height=src_h
            )
        return prepare_letterboxed(
            conditioning.appearance,
            conditioning.bbox,
            canvas_width,
            canvas_height,
            extras=extras,
        )

    def _generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        prepared = self.prepare(conditioning, params)
        pipeline = self._pipeline if self._pipeline is not None else self._load_pipeline(device)
        steps = params.steps if params.steps is not None else self.steps
        strength = params.strength if params.strength is not None else self.strength
        guidance = params.guidance_scale if params.guidance_scale is not None else self.guidance
        width, height = self.canvas_size(params)

        appearance = prepared["appearance"]
        init = params.init_image
        init_hwc = as_hwc(init) if init is not None else appearance
        control = self._control_image(prepared)
        self.last_seed = int(seed)
        prompt, source = resolve_prompt(conditioning.caption)
        self.last_prompt = prompt
        self.last_prompt_source = source

        output = self._call_pipeline(
            pipeline,
            init_hwc=init_hwc,
            control=control,
            appearance=appearance,
            seed=seed,
            device=device,
            steps=steps,
            strength=strength,
            guidance=guidance,
            width=width,
            height=height,
            prompt=prompt,
        )
        return as_chw(_coerce_output(output))

    def _call_pipeline(
        self,
        pipeline: Any,
        *,
        init_hwc: np.ndarray,
        control: np.ndarray | list[np.ndarray],
        appearance: np.ndarray,
        seed: int,
        device: Device,
        steps: int,
        strength: float,
        guidance: float,
        width: int,
        height: int,
        prompt: str,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator_seed": seed,
        }
        if self.variant == "ip-adapter":
            kwargs["image"] = control
            kwargs["ip_adapter_image"] = appearance
        else:
            kwargs["image"] = init_hwc
            kwargs["control_image"] = control
            kwargs["strength"] = strength

        if not _looks_like_diffusers(pipeline):
            return pipeline(**kwargs)

        from PIL import Image

        def to_pil(image: np.ndarray) -> Image.Image:
            array = as_hwc(image)
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            elif array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            return Image.fromarray(array)

        if self.variant == "ip-adapter":
            kwargs["image"] = to_pil(control) if not isinstance(control, list) else [
                to_pil(item) for item in control
            ]
            kwargs["ip_adapter_image"] = to_pil(appearance)
            kwargs.pop("control_image", None)
            kwargs.pop("strength", None)
        else:
            kwargs["image"] = to_pil(init_hwc)
            if isinstance(control, list):
                kwargs["control_image"] = [to_pil(item) for item in control]
            else:
                kwargs["control_image"] = to_pil(control)
        kwargs.pop("generator_seed", None)
        kwargs["generator"] = _torch_generator(device, seed)
        return pipeline(**kwargs)

    def _control_image(self, prepared: dict[str, Any]) -> np.ndarray | list[np.ndarray]:
        if self.variant == "canny":
            return prepared["canny"]
        if self.variant == "seg":
            return prepared["mask"]
        if self.variant == "multi":
            return [prepared["pose"], prepared["mask"]]
        if self.variant == "trajectory":
            return prepared["trajectory"]
        if self.variant == "pose-ref":
            return compose_pose_on_appearance(prepared["pose"], prepared["appearance"])
        return prepared["pose"]

    def _load_pipeline(self, device: Device) -> Any:
        try:
            from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"{self.variant}-controlnet has no pipeline loaded and diffusers "
                f"is not installed. Pass a test double as pipeline=... "
                f"Requested device={device!r}, checkpoint={self.checkpoint!r}."
            ) from exc

        control_path, loaded_epoch = resolve_controlnet_checkpoint(
            self.variant, self.checkpoint, self.epoch
        )
        sd_path = self._resolve_sd()
        dtype = _dtype_for(device)
        _LOGGER.info(
            "Loading %s-controlnet from %s (epoch=%s, sd=%s, dtype=%s, device=%s)",
            self.variant,
            control_path,
            loaded_epoch,
            sd_path,
            dtype,
            device,
        )

        if self.variant == "multi":
            pose_path, pose_epoch = resolve_controlnet_checkpoint("pose", None, self.epoch)
            seg_path, seg_epoch = resolve_controlnet_checkpoint("seg", None, None)
            controlnet = [
                ControlNetModel.from_pretrained(
                    str(pose_path), torch_dtype=dtype, local_files_only=True
                ),
                ControlNetModel.from_pretrained(
                    str(seg_path), torch_dtype=dtype, local_files_only=True
                ),
            ]
            self.loaded_checkpoint = f"{pose_path}+{seg_path}"
            self.loaded_epoch = pose_epoch
            _LOGGER.info("multi-controlnet epochs pose=%s seg=%s", pose_epoch, seg_epoch)
        else:
            controlnet = ControlNetModel.from_pretrained(
                str(control_path), torch_dtype=dtype, local_files_only=True
            )
            self.loaded_checkpoint = str(control_path)
            self.loaded_epoch = loaded_epoch

        if self.variant == "ip-adapter":
            from diffusers import StableDiffusionControlNetPipeline

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                str(sd_path),
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                local_files_only=True,
            )
            pipe.set_progress_bar_config(disable=True)
            # Attention slicing replaces attn processors; load IP-Adapter after
            # placement so those processors stay installed.
            pipe = _place_pipeline(pipe, device, attention_slicing=False)
            pipe.load_ip_adapter(
                _IP_ADAPTER_REPO,
                subfolder=_IP_ADAPTER_SUBFOLDER,
                weight_name=_IP_ADAPTER_WEIGHT,
                local_files_only=True,
            )
            pipe.set_ip_adapter_scale(self.ip_adapter_scale)
        else:
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                str(sd_path),
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                local_files_only=True,
            )
            pipe.set_progress_bar_config(disable=True)
            pipe = _place_pipeline(pipe, device)
        self._pipeline = pipe
        return pipe

    def _resolve_sd(self) -> Path:
        if self.sd_model:
            path = Path(self.sd_model)
            if not path.is_absolute():
                planted = weights_dir() / self.sd_model
                path = planted if planted.exists() else path
            if not path.exists():
                raise FileNotFoundError(f"Stable Diffusion weights not found at {path}.")
            return path.resolve()
        path = weights_dir() / _SD15_DIR
        if not path.exists():
            raise FileNotFoundError(
                f"stable-diffusion-v1-5 not found at {path}. "
                f"{self.variant}-controlnet needs it as the shared backbone."
            )
        return path.resolve()


def _looks_like_diffusers(pipeline: Any) -> bool:
    return hasattr(pipeline, "unet") and callable(pipeline)


def _torch_generator(device: Device, seed: int) -> Any:
    import torch

    resolved = torch.device(device) if not isinstance(device, torch.device) else device
    if resolved.type == "cuda":
        return torch.Generator(device=resolved).manual_seed(int(seed))
    return torch.Generator().manual_seed(int(seed))


def _dtype_for(device: Device) -> Any:
    import torch

    from src.components.generation.torch_dtype import resolve_torch_dtype_for_device

    return resolve_torch_dtype_for_device(
        device, default_cuda=torch.float16, allowed_cuda={torch.float16, torch.bfloat16, torch.float32}
    )


def _place_pipeline(pipe: Any, device: Device, *, attention_slicing: bool = True) -> Any:
    """Move a pipeline onto ``device`` without grabbing a busy GPU whole.

    Shared GPUs on this host often have ~10 GB leftover. ``to(cuda)`` of SD-1.5
    + ControlNet in fp16 fits that; if it does not, fall back to CPU offload
    rather than OOM a neighbour. CPU stays on CPU.

    Attention slicing is optional: IP-Adapter installs its own attn processors,
    and slicing after that load replaces them and crashes at step 0.
    """
    import torch

    resolved = torch.device(device) if not isinstance(device, torch.device) else device
    if attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if resolved.type != "cuda":
        return pipe.to(resolved)
    try:
        return pipe.to(resolved)
    except torch.cuda.OutOfMemoryError:
        _LOGGER.warning(
            "ControlNet pipeline did not fit on %s; enabling model CPU offload.", resolved
        )
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            return pipe
        raise


def _coerce_output(output: Any) -> np.ndarray:
    if isinstance(output, np.ndarray):
        return output
    if hasattr(output, "mode") and hasattr(output, "size"):
        return np.asarray(output)
    images = getattr(output, "images", None)
    if images:
        return np.asarray(images[0])
    if isinstance(output, (tuple, list)) and output:
        return np.asarray(output[0])
    raise TypeError(f"ControlNet pipeline returned unusable output: {type(output)!r}.")
