"""Frozen HNeRV: encode once, quantize, package, decode from packets alone.

No training and no latent fitting. The architecture comes from the saved
tensor shapes (embedding channels and decoder width), never from the
evaluation frame count, and the state loads strictly from bytes whose
SHA-256 was just verified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from demo.experiments.background_smoke_core import HNERV_ROOT, unit_float_to_uint8

ENC_STRIDES = [5, 3, 2, 2, 2]
DEC_STRIDES = [5, 3, 2, 2, 2]
EMBED_KEY = "decoder.0.conv.downconv.weight"
ENCODER_STEM_KEY = "encoder.downsample_layers.0.0.weight"
MODEL_QUANT_BITS = 8


def write_import_stub(root: Path) -> dict[str, Any]:
    """Modules HNeRV imports but never uses on this path.

    pytorchvideo/decord serve video files only; pytorch_msssim provides SSIM
    losses that frozen decoding never calls. A stub is written only for a
    module the interpreter lacks, and every stub is recorded.
    """
    import importlib.util

    root.mkdir(parents=True, exist_ok=True)
    stubbed = []
    if importlib.util.find_spec("pytorchvideo") is None:
        package = root / "pytorchvideo" / "data"
        package.mkdir(parents=True, exist_ok=True)
        (root / "pytorchvideo" / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "encoded_video.py").write_text("class EncodedVideo:\n    pass\n")
        stubbed.append("pytorchvideo")
    if importlib.util.find_spec("decord") is None:
        (root / "decord.py").write_text(
            "class _Bridge:\n    def set_bridge(self, _name):\n        return None\n"
            "bridge = _Bridge()\n"
            "class VideoReader:\n    def __init__(self, *_args, **_kwargs):\n"
            "        raise RuntimeError('frozen HNeRV decoding does not read video files')\n"
        )
        stubbed.append("decord")
    if importlib.util.find_spec("pytorch_msssim") is None:
        (root / "pytorch_msssim.py").write_text(
            "def _unused(*_args, **_kwargs):\n"
            "    raise RuntimeError('SSIM is not part of frozen HNeRV decoding')\n"
            "ms_ssim = ssim = _unused\n"
        )
        stubbed.append("pytorch_msssim")
    return {"root": str(root), "stubbed_modules": stubbed}


def enable_imports(stub_root: Path) -> dict[str, Any]:
    record = write_import_stub(stub_root)
    for entry in (str(HNERV_ROOT), str(stub_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import model_all  # noqa: F401
    import hnerv_utils  # noqa: F401

    record["model_all"] = str(Path(sys.modules["model_all"].__file__))
    record["hnerv_utils"] = str(Path(sys.modules["hnerv_utils"].__file__))
    return record


def load_verified_state(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"checkpoint bytes {actual} differ from verified {expected_sha256}")
    try:
        saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        loader = "torch.load(weights_only=True)"
    except Exception:
        # Same verified bytes; HNeRV saves optimizer state beside the weights.
        saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        loader = "torch.load(weights_only=False) on hash-verified bytes"
    state = saved["state_dict"] if "state_dict" in saved else saved
    canonical = {}
    for key, value in state.items():
        name = key[len("module."):] if key.startswith("module.") else key
        if name in canonical:
            raise ValueError(f"state-dict key collision after module-prefix removal: {name}")
        canonical[name] = value
    epoch = saved.get("epoch") if isinstance(saved, dict) else None
    meta = {"path": str(path), "sha256": actual, "bytes": len(data), "epoch": int(epoch) if epoch is not None else None, "loader": loader}
    return canonical, meta


def architecture_args(state: dict[str, Any]) -> tuple[argparse.Namespace, dict[str, int]]:
    """Training configuration with widths read from the saved tensors."""
    if EMBED_KEY not in state or ENCODER_STEM_KEY not in state:
        raise ValueError("checkpoint lacks the HNeRV embedding/encoder tensors needed to recover its architecture")
    fc_dim, embed_dim, kh, kw = (int(v) for v in state[EMBED_KEY].shape)
    if (kh, kw) != (1, 1):
        raise ValueError(f"unexpected first decoder kernel {kh}x{kw}")
    enc_dim1 = int(state[ENCODER_STEM_KEY].shape[0])
    args = argparse.Namespace(
        embed="", ks="0_1_5", num_blks="1_1", enc_strds=ENC_STRIDES, dec_strds=DEC_STRIDES,
        enc_dim=f"{enc_dim1}_{embed_dim}", conv_type=["convnext", "pshuffel"], norm="none",
        act="gelu", reduce=1.2, lower_width=12, fc_dim=fc_dim, fc_hw="9_16", out_bias="tanh",
        quant_model_bit=MODEL_QUANT_BITS, quant_embed_bit=6, vid="frozen",
    )
    return args, {"embed_dim": embed_dim, "fc_dim": fc_dim, "enc_dim1": enc_dim1}


def build_model(state: dict[str, Any], device: str = "cuda"):
    import torch
    from model_all import HNeRV

    args, dims = architecture_args(state)
    model = HNeRV(args)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if (model.fc_h, model.fc_w) != (1, 1):
        raise ValueError("recovered architecture does not have a 1x1 decoder grid")
    dims["parameters"] = sum(p.numel() for p in model.parameters())
    dims["decoder_parameters"] = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder"))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return model, args, dims


def quantize_decoder(model):
    """train_nerv_all.quant_model: every non-encoder tensor at 8 bits."""
    from copy import deepcopy
    from hnerv_utils import quant_tensor

    quantized = deepcopy(model)
    current = quantized.state_dict()
    codes = {}
    for key, value in list(current.items()):
        if "encoder" in key:
            continue
        quant, new_value = quant_tensor(value, MODEL_QUANT_BITS)
        codes[key] = quant
        current[key] = new_value
    quantized.load_state_dict(current, strict=True)
    return quantized.eval(), codes


def decoder_of(model):
    from model_all import HNeRVDecoder

    return HNeRVDecoder(model).eval()


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy()


def write_setup_package(codes: dict[str, Any], path: Path) -> dict[str, Any]:
    """Decoder setup: 8-bit codes and exact min/scale arrays, zip-deflated."""
    arrays = {}
    for index, key in enumerate(sorted(codes)):
        arrays[f"{index:03d}_quant"] = _numpy(codes[key]["quant"]).astype(np.uint8)
        arrays[f"{index:03d}_min"] = _numpy(codes[key]["min"])
        arrays[f"{index:03d}_scale"] = _numpy(codes[key]["scale"])
    arrays["keys"] = np.asarray(sorted(codes))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    data = buffer.getvalue()
    with Path(path).open("xb") as stream:
        stream.write(data)
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "tensors": len(codes)}


def decoder_from_setup(state: dict[str, Any], path: Path, device: str = "cuda"):
    """A fresh decoder holding only what the setup package carries."""
    import torch
    from model_all import HNeRV, HNeRVDecoder

    args, _dims = architecture_args(state)
    with np.load(path, allow_pickle=False) as package:
        keys = [str(key) for key in package["keys"]]
        restored = {}
        for index, key in enumerate(keys):
            quant = torch.from_numpy(package[f"{index:03d}_quant"]).to(device)
            minimum = torch.from_numpy(package[f"{index:03d}_min"]).to(device)
            scale = torch.from_numpy(package[f"{index:03d}_scale"]).to(device)
            restored[key] = minimum.expand_as(quant) + scale.expand_as(quant) * quant
    decoder = HNeRVDecoder(HNeRV(args)).to(device).eval()
    decoder.load_state_dict(restored, strict=True)
    return decoder, restored


def encode_frames(model, frames: list[np.ndarray], device: str = "cuda"):
    """Float embeddings [N, C, 9, 16] from uint8 RGB frames, one pass each."""
    import torch

    embeds = []
    with torch.inference_mode():
        for pixels in frames:
            image = torch.from_numpy(np.ascontiguousarray(pixels)).to(device).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            embeds.append(model.encoder(image))
    return torch.cat(embeds, 0)


def quantize_embeddings(embed, bits: int) -> tuple[np.ndarray, dict[str, np.ndarray], Any]:
    """hnerv_utils.quant_tensor on one independent segment."""
    from hnerv_utils import quant_tensor

    quant, new_value = quant_tensor(embed, bits)
    codes = _numpy(quant["quant"]).astype(np.uint8)
    return codes, {"min": _numpy(quant["min"]), "scale": _numpy(quant["scale"])}, new_value


def decode_pixels(decoder, embed) -> list[np.ndarray]:
    import torch

    with torch.inference_mode():
        output = decoder(embed)
    return [unit_float_to_uint8(_numpy(frame), layout="CHW", value_range="unit") for frame in output]


def source_independence(model, decoder, embed, sources: list[np.ndarray], device: str = "cuda") -> dict[str, Any]:
    """Compare decoder-only output with full-model calls given real, zero and random sources."""
    import torch

    with torch.inference_mode():
        reference = decoder(embed)
        repeat = decoder(embed)
        gt = torch.stack([torch.from_numpy(np.ascontiguousarray(p)).permute(2, 0, 1) for p in sources]).to(device).float() / 255.0
        variants = {"real_source": gt, "zero_source": torch.zeros_like(gt), "random_source": torch.rand_like(gt)}
        differences = {}
        for name, tensor in variants.items():
            output, _embeds, _seconds = model(tensor, input_embed=embed)
            differences[name] = float((output - reference).abs().max())
    determinism = float((repeat - reference).abs().max())
    return {
        "decoder_inputs": "embedding only (HNeRVDecoder.forward(img_embed))",
        "repeat_max_abs_diff": determinism,
        "full_model_max_abs_diff": differences,
        "independent": all(value <= determinism for value in differences.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--stub", type=Path, required=True)
    args = parser.parse_args(argv)
    record = enable_imports(args.stub)
    import torch

    record.update(torch=torch.__version__, torch_cuda=torch.version.cuda, python=sys.version.split()[0])
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
