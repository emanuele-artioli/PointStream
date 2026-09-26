"""DINOv3 raw patch-token maps for the maps gallery.

These are ViT-S/16 patch features, not a depth map and not an MPEG-FCM
payload. PCA is a human preview only; the counted payload is pooled int8
zstd. Linear probes are out of scope.

Load order (local files only — never a Hub id or auto-download):
  1. `dinov3` package / local facebookresearch hub → ViT-S/16 + load_state_dict
  2. transformers AutoModel if a config.json sits next to the raw .pth
  3. CLI exit 2: pth present, architecture code missing

Pack/PCA helpers operate on a plain (Hp, Wp, C) array and do not need the
backbone. Unit tests use a fake tensor.

Examples:
  PYTHONPATH=. python -m demo.pipeline.maps.dinov3_features --clip clip.mp4 --out demo/outputs/maps/clip/dino_feat/
  PYTHONPATH=. python -m demo.pipeline.maps.dinov3_features --clip clip.mp4 --out demo/outputs/maps/clip/dino_feat/ --max-frames 8
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from demo.evaluation.profile_map import profile_map
from demo.pipeline.maps.contract import MapStream, write_sidecar
from demo.pipeline.maps.encode import write_zstd_blob
from demo.pipeline.maps.model_paths import require

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None
    import zlib

MAP_NAME = "dino_feat"
POOLED_HW = (32, 32)
INT8_SCALE = 64.0
PATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 512
PAYLOAD_NAME = "feat_32x32_int8.bin"
PREVIEW_MP4_NAME = "preview_pca.mp4"
SIDECAR_NAME = "sidecar.json"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_STATE_PREFIXES = ("module.", "backbone.", "model.", "teacher.", "student.", "module.backbone.")
_PACK_HEADER = struct.Struct("<4sBIIIIf")
_PACK_MAGIC = b"DIN8"
_PACK_VERSION = 1


class Dinov3CodeMissing(RuntimeError):
    """Raw .pth is on disk but ViT-S/16 construction code is not importable."""

    exit_code = 2


# ---------------------------------------------------------------------------
# Bytes (no torch)
# ---------------------------------------------------------------------------


def _compress(raw: bytes) -> bytes:
    if zstd is not None:
        return zstd.ZstdCompressor(level=3).compress(raw)
    return zlib.compress(raw, level=6)


def _decompress(payload: bytes) -> bytes:
    if zstd is not None:
        return zstd.ZstdDecompressor().decompress(payload)
    return zlib.decompress(payload)


def spatial_pool(feat: np.ndarray, out_h: int = 32, out_w: int = 32) -> np.ndarray:
    """Average-pool (or upsample) (Hp, Wp, C) to (out_h, out_w, C).

    OpenCV resize only handles 1/3/4 channels, so this stays in numpy and
    works for ViT-S C=384 (and any fake C used in tests).
    """
    if feat.ndim != 3:
        raise ValueError(f"expected (Hp, Wp, C), got shape {feat.shape}")
    feat = np.ascontiguousarray(feat, dtype=np.float32)
    hp, wp, c = feat.shape
    if (hp, wp) == (out_h, out_w):
        return feat
    if hp >= out_h and wp >= out_w and hp % out_h == 0 and wp % out_w == 0:
        bh, bw = hp // out_h, wp // out_w
        return feat.reshape(out_h, bh, out_w, bw, c).mean(axis=(1, 3))
    return _resize_feature_grid(feat, out_h, out_w)


def _resize_feature_grid(feat: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    hp, wp, c = feat.shape
    ys = np.linspace(0.0, hp, out_h + 1)
    xs = np.linspace(0.0, wp, out_w + 1)
    out = np.empty((out_h, out_w, c), dtype=np.float32)
    for i in range(out_h):
        y0 = min(hp - 1, int(np.floor(ys[i])))
        y1 = min(hp, max(int(np.ceil(ys[i + 1])), y0 + 1))
        for j in range(out_w):
            x0 = min(wp - 1, int(np.floor(xs[j])))
            x1 = min(wp, max(int(np.ceil(xs[j + 1])), x0 + 1))
            out[i, j] = feat[y0:y1, x0:x1].mean(axis=(0, 1))
    return out


def zscore_per_channel(feat: np.ndarray) -> np.ndarray:
    """Per-channel z-score over all non-channel axes."""
    feat = np.asarray(feat, dtype=np.float32)
    axes = tuple(range(feat.ndim - 1))
    mean = feat.mean(axis=axes, keepdims=True)
    std = feat.std(axis=axes, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (feat - mean) / std


def quantize_int8(zscored: np.ndarray, scale: float = INT8_SCALE) -> np.ndarray:
    """q = clip(round(z * scale), -127, 127) as int8. Not a depth encoding."""
    q = np.clip(np.rint(np.asarray(zscored, dtype=np.float32) * scale), -127, 127)
    return q.astype(np.int8)


def dequantize_int8(quantized: np.ndarray, scale: float = INT8_SCALE) -> np.ndarray:
    return quantized.astype(np.float32) / float(scale)


def pack_int8_features(feat: np.ndarray, scale: float = INT8_SCALE) -> bytes:
    """Spatial-pool to 32x32, per-channel z-score, int8, zstd.

    `feat` is (Hp, Wp, C) or (T, Hp, Wp, C). Returns a self-describing blob.
    """
    arr = np.asarray(feat, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"expected (Hp,Wp,C) or (T,Hp,Wp,C), got {arr.shape}")
    pooled = np.stack([spatial_pool(frame, *POOLED_HW) for frame in arr], axis=0)
    quantized = quantize_int8(zscore_per_channel(pooled), scale=scale)
    t, h, w, c = quantized.shape
    header = _PACK_HEADER.pack(_PACK_MAGIC, _PACK_VERSION, t, h, w, c, float(scale))
    return _compress(header + quantized.tobytes(order="C"))


def unpack_int8_features(payload: bytes) -> tuple[np.ndarray, float]:
    """Inverse of pack_int8_features. Returns ((T, 32, 32, C) int8, scale)."""
    raw = _decompress(payload)
    header_size = _PACK_HEADER.size
    magic, version, t, h, w, c, scale = _PACK_HEADER.unpack(raw[:header_size])
    if magic != _PACK_MAGIC:
        raise ValueError(f"bad feature blob magic {magic!r}")
    if version != _PACK_VERSION:
        raise ValueError(f"unsupported feature blob version {version}")
    body = np.frombuffer(raw[header_size:], dtype=np.int8)
    expected = t * h * w * c
    if body.size != expected:
        raise ValueError(f"blob length {body.size} != T*H*W*C {expected}")
    return body.reshape(t, h, w, c), float(scale)


def pca_to_rgb(
    feat: np.ndarray,
    *,
    components: np.ndarray | None = None,
    mean: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project last axis C→3 and stretch each PC to uint8 RGB.

    `feat` is (..., C). Returns (rgb uint8, components (3, C), mean (C,)).
    PCA is a preview, not a learned probe.
    """
    arr = np.asarray(feat, dtype=np.float32)
    if arr.shape[-1] < 1:
        raise ValueError("C must be >= 1")
    flat = arr.reshape(-1, arr.shape[-1])
    if mean is None:
        mean = flat.mean(axis=0)
    centered = flat - mean
    n_comp = min(3, centered.shape[1], max(1, centered.shape[0] - 1))
    if components is None:
        if centered.shape[0] == 1 or n_comp == 0:
            components = np.zeros((3, centered.shape[1]), dtype=np.float32)
            components[: min(3, centered.shape[1]), : min(3, centered.shape[1])] = np.eye(
                min(3, centered.shape[1]), dtype=np.float32
            )
        else:
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
            basis = np.asarray(vt[:n_comp], dtype=np.float32)
            if n_comp < 3:
                padded = np.zeros((3, centered.shape[1]), dtype=np.float32)
                padded[:n_comp] = basis
                components = padded
            else:
                components = basis
    proj = centered @ components.T
    lo = proj.min(axis=0)
    hi = proj.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    rgb = ((proj - lo) / span * 255.0).clip(0, 255).astype(np.uint8)
    return rgb.reshape(*arr.shape[:-1], 3), components, np.asarray(mean, dtype=np.float32)


def upsample_rgb(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Backbone load (lazy torch; never a Hub id)
# ---------------------------------------------------------------------------


def missing_code_message(pth: Path) -> str:
    return (
        f"Error: checkpoint is present at {pth} but the dinov3 code is not.\n"
        "This is a raw .pth (facebookresearch ViT-S/16), not a HuggingFace folder.\n"
        "Install the local `dinov3` package, set DINOV3_REPO to a clone that has "
        "hubconf.py, or place config.json next to the pth.\n"
        "Do not download from the Hub.\n"
        "  PYTHONPATH=. python -m demo.pipeline.maps.dinov3_features "
        "--clip <clip.mp4> --out demo/outputs/maps/<stem>/dino_feat/\n"
        "Pack/PCA helpers still run on a fake (Hp, Wp, C) tensor without the model."
    )


def _strip_state_dict(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a dict, got {type(state)!r}")
    for nested in ("state_dict", "model", "teacher"):
        inner = state.get(nested)
        if isinstance(inner, dict) and inner and not any(
            k in inner for k in ("state_dict", "epoch", "optimizer")
        ):
            if any(hasattr(v, "shape") for v in inner.values()):
                state = inner
                break
    out: dict[str, Any] = {}
    for key, value in state.items():
        if not hasattr(value, "shape"):
            continue
        name = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in _STATE_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    changed = True
        out[name] = value
    return out


def _torch_load(pth: Path) -> Any:
    import torch

    kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        return torch.load(str(pth), weights_only=True, **kwargs)
    except TypeError:
        return torch.load(str(pth), **kwargs)


def local_hf_config_dir(pth: Path) -> Path | None:
    """A HuggingFace config next to the raw pth — never a remote repo id."""
    for candidate in (pth.parent, pth.with_suffix("")):
        if (candidate / "config.json").is_file():
            return candidate
    return None


def _dinov3_vits16_constructor() -> Callable[..., Any] | None:
    try:
        from dinov3.hub.backbones import dinov3_vits16

        return dinov3_vits16
    except ImportError:
        pass
    try:
        from dinov3.models.vision_transformer import vit_small

        def _vit_small(**kwargs: Any) -> Any:
            kwargs.pop("pretrained", None)
            kwargs.setdefault("patch_size", PATCH_SIZE)
            kwargs.setdefault("n_storage_tokens", 4)
            kwargs.setdefault("img_size", 224)
            return vit_small(**kwargs)

        return _vit_small
    except ImportError:
        return None


def _local_dinov3_repo() -> Path | None:
    env = os.environ.get("DINOV3_REPO")
    if not env:
        return None
    repo = Path(env).expanduser()
    if (repo / "hubconf.py").is_file():
        return repo
    return None


def _construct_vits16() -> tuple[Any, str] | None:
    ctor = _dinov3_vits16_constructor()
    if ctor is not None:
        model = ctor(pretrained=False)
        return model, ctor.__module__ + "." + getattr(ctor, "__name__", "dinov3_vits16")

    repo = _local_dinov3_repo()
    if repo is not None:
        import torch

        model = torch.hub.load(
            str(repo),
            "dinov3_vits16",
            source="local",
            pretrained=False,
            trust_repo=True,
        )
        return model, f"torch.hub.local:{repo}"
    return None


def _load_with_facebook_code(pth: Path, device: str) -> tuple[Any, str]:
    built = _construct_vits16()
    if built is None:
        raise Dinov3CodeMissing(missing_code_message(pth))
    model, source = built
    state = _strip_state_dict(_torch_load(pth))
    model.load_state_dict(state, strict=False)
    import torch

    model.to(device)
    model.eval()
    return model, source


def _load_with_transformers(pth: Path, device: str) -> tuple[Any, str]:
    cfg_dir = local_hf_config_dir(pth)
    if cfg_dir is None:
        raise Dinov3CodeMissing(missing_code_message(pth))
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(str(cfg_dir), local_files_only=True)
    model = AutoModel.from_config(config)
    try:
        state = _strip_state_dict(_torch_load(pth))
        model.load_state_dict(state, strict=False)
    except Exception:
        # Folder may already hold converted HF weights.
        model = AutoModel.from_pretrained(str(cfg_dir), local_files_only=True)
    import torch

    model.to(device)
    model.eval()
    return model, f"transformers:{cfg_dir}"


def _facebook_code_available() -> bool:
    return _dinov3_vits16_constructor() is not None or _local_dinov3_repo() is not None


def load_backbone(pth: Path, device: str | None = None) -> tuple[Any, str, str]:
    """Return (model, load_source, device). Never downloads."""
    facebook = _facebook_code_available()
    hf = local_hf_config_dir(pth) is not None
    if not facebook and not hf:
        raise Dinov3CodeMissing(missing_code_message(pth))
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if facebook:
        model, source = _load_with_facebook_code(pth, device)
        return model, source, device
    model, source = _load_with_transformers(pth, device)
    return model, source, device


def _patch_size(model: Any) -> int:
    if hasattr(model, "patch_size"):
        ps = model.patch_size
        return int(ps[0] if isinstance(ps, (tuple, list)) else ps)
    embed = getattr(model, "patch_embed", None)
    if embed is not None and hasattr(embed, "patch_size"):
        ps = embed.patch_size
        return int(ps[0] if isinstance(ps, (tuple, list)) else ps)
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "patch_size"):
        ps = config.patch_size
        return int(ps[0] if isinstance(ps, (tuple, list)) else ps)
    return PATCH_SIZE


def _n_register_tokens(model: Any) -> int:
    for attr in ("n_storage_tokens", "num_register_tokens"):
        if hasattr(model, attr):
            return int(getattr(model, attr))
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "num_register_tokens"):
        return int(config.num_register_tokens)
    return 4


def preprocess_rgb(rgb: np.ndarray, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """uint8 RGB HWC → float32 NCHW ImageNet-normalized, size multiple of 16."""
    size = max(PATCH_SIZE, int(image_size) // PATCH_SIZE * PATCH_SIZE)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1))[None]


def tokens_to_patch_map(
    tokens: Any,
    height: int,
    width: int,
    patch: int,
    n_register: int,
) -> np.ndarray:
    """Drop CLS + register tokens and reshape to (Hp, Wp, C)."""
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().float().cpu().numpy()
    arr = np.asarray(tokens, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected token matrix (N, C), got {arr.shape}")
    n_tokens = int(arr.shape[0])
    hp, wp = height // patch, width // patch
    n_special = 1 + int(n_register)
    if n_tokens == hp * wp:
        patches = arr
    elif n_tokens >= n_special + hp * wp:
        patches = arr[n_special : n_special + hp * wp]
    else:
        patches = arr[n_special:] if n_tokens > n_special else arr
        n = int(patches.shape[0])
        side = int(round(n**0.5))
        if side * side != n:
            raise RuntimeError(
                f"cannot reshape {n} patch tokens (Hp,Wp expected {hp}x{wp} from {height}x{width})"
            )
        hp = wp = side
    return np.ascontiguousarray(patches.reshape(hp, wp, -1), dtype=np.float32)


def forward_patch_map(model: Any, rgb: np.ndarray, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """RGB uint8 → (Hp, Wp, C) float32 patch tokens. Not a depth map."""
    import torch

    nchw = preprocess_rgb(rgb, image_size=image_size)
    pixel = torch.from_numpy(nchw)
    device = next(model.parameters()).device
    pixel = pixel.to(device)
    patch = _patch_size(model)
    n_reg = _n_register_tokens(model)
    h, w = int(pixel.shape[-2]), int(pixel.shape[-1])
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            out = model.forward_features(pixel)
            if isinstance(out, dict) and "x_norm_patchtokens" in out:
                tokens = out["x_norm_patchtokens"]
                n_reg = 0  # already dropped
            elif hasattr(out, "last_hidden_state"):
                tokens = out.last_hidden_state
            else:
                tokens = out
        else:
            try:
                out = model(pixel_values=pixel)
            except TypeError:
                out = model(pixel)
            if hasattr(out, "last_hidden_state"):
                tokens = out.last_hidden_state
            elif isinstance(out, dict) and "last_hidden_state" in out:
                tokens = out["last_hidden_state"]
            else:
                tokens = out
        return tokens_to_patch_map(tokens, h, w, patch, n_reg)


# ---------------------------------------------------------------------------
# Clip IO / preview
# ---------------------------------------------------------------------------


def iter_clip_frames(path: Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot open clip {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return [rgb], 30.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"no frames in {path}")
    return frames, fps


def write_preview_pca(
    patch_maps: list[np.ndarray],
    out_dir: Path,
    src_hw: tuple[int, int],
    fps: float,
) -> Path:
    stacked = np.stack(patch_maps, axis=0)
    rgb_seq, _components, _mean = pca_to_rgb(stacked)
    src_h, src_w = src_hw
    preview = [upsample_rgb(frame, src_h, src_w) for frame in rgb_seq]
    seq_dir = out_dir / "preview_pca"
    seq_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(preview):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = np.clip(luma.astype(np.int32) * 2, 0, 200).astype(np.uint8)
        cv2.imwrite(str(seq_dir / f"{i:06d}.png"), bgra)
    return seq_dir


def extra_from_features(patch_maps: list[np.ndarray]) -> dict[str, Any]:
    hp, wp, c = patch_maps[0].shape
    return {
        "C": int(c),
        "Hp": int(hp),
        "Wp": int(wp),
        "pooled": [POOLED_HW[0], POOLED_HW[1]],
        "dtype": "int8",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 ViT-S/16 patch features (not depth) as a maps-gallery stream.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  PYTHONPATH=. python -m demo.pipeline.maps.dinov3_features "
            "--clip clip.mp4 --out demo/outputs/maps/clip/dino_feat/\n"
            "  PYTHONPATH=. python -m demo.pipeline.maps.dinov3_features "
            "--clip clip.mp4 --out demo/outputs/maps/clip/dino_feat/ --max-frames 8\n"
        ),
    )
    parser.add_argument("--clip", type=Path, required=True, help="Source video (or still image).")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory, typically demo/outputs/maps/<stem>/dino_feat/",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on decoded frames.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square resize before the ViT (multiple of 16). Default 512 → 32x32 patches.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Override MODELS['dinov3_vits']. Must be a local .pth, never a Hub id.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pth = Path(args.weights) if args.weights is not None else require("dinov3_vits")
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "  PYTHONPATH=. python -m demo.pipeline.maps.dinov3_features "
            "--clip <clip.mp4> --out demo/outputs/maps/<stem>/dino_feat/",
            file=sys.stderr,
        )
        return 2
    if not pth.is_file():
        print(f"Error: MODELS['dinov3_vits'] path {pth} is not a file.", file=sys.stderr)
        return 2

    try:
        model, load_source, device = load_backbone(pth)
    except Dinov3CodeMissing as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: failed to load {pth}: {exc}", file=sys.stderr)
        print(missing_code_message(pth), file=sys.stderr)
        return 2

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = iter_clip_frames(args.clip, max_frames=args.max_frames)
    image_size = int(args.image_size)

    def extract_fn(frame: np.ndarray) -> np.ndarray:
        return forward_patch_map(model, frame, image_size=image_size)

    patch_maps = [extract_fn(frame) for frame in frames]
    payload = pack_int8_features(np.stack(patch_maps, axis=0))
    payload_path = out_dir / PAYLOAD_NAME
    write_zstd_blob(payload, payload_path)

    src_h, src_w = frames[0].shape[:2]
    preview_path = write_preview_pca(patch_maps, out_dir, (src_h, src_w), fps)

    stats = profile_map(
        extract_fn,
        frames[0],
        pack_fn=pack_int8_features,
        decode_fn=lambda blob: unpack_int8_features(blob)[0],
        codec_fn=None,
    )

    extra = extra_from_features(patch_maps)
    extra["load_source"] = load_source
    extra["image_size"] = image_size
    extra["n_register_tokens"] = _n_register_tokens(model)
    extra["overlay"] = "rgba"

    n_frames = len(frames)
    duration_s = n_frames / float(fps)
    stream = MapStream(
        map=MAP_NAME,
        backend=pth.name,
        payload_path=str(payload_path),
        payload_bytes=payload_path.stat().st_size,
        preview_path=str(preview_path),
        preview_bytes=_preview_bytes(preview_path),
        duration_s=duration_s,
        n_frames=n_frames,
        fps=float(fps),
        extract_ms_p50=float(stats["extract_ms_p50"]),
        extract_ms_p95=float(stats["extract_ms_p95"]),
        pack_ms_p50=float(stats["pack_ms_p50"]),
        codec_ms_p50=float(stats["codec_ms_p50"]),
        decode_ms_p50=float(stats["decode_ms_p50"]),
        gpu=str(stats.get("gpu") or _device_name()),
        kind="native",
        extra=extra,
    )
    sidecar_path = write_sidecar(stream, out_dir / SIDECAR_NAME)
    print(
        f"wrote {payload_path} ({payload_path.stat().st_size} bytes) "
        f"preview={preview_path} sidecar={sidecar_path} "
        f"backend={pth.name} load={load_source} device={device}"
    )
    return 0


def _preview_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
