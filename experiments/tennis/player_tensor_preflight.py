"""CPU-only comparison of the fit's input tensors to pinned pipeline processors."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture() -> np.ndarray:
    """An asymmetric, colored 784x512 crop with distinct top and bottom bands."""
    y, x = np.indices((784, 512), dtype=np.uint32)
    a = np.stack(((x + 3 * y) % 256, (2 * x + y) % 256, (5 * x + 7 * y) % 256), axis=-1).astype(np.uint8)
    a[:96] = (241, 37, 13)
    a[-96:] = (9, 67, 229)
    return a


def audit(upstream_pipeline: Path | None = None) -> dict[str, Any]:
    import torch
    from diffusers.image_processor import VaeImageProcessor
    from PIL import Image
    from transformers import CLIPImageProcessor

    from experiments.tennis.k6b_fit import Trainer

    assert not torch.cuda.is_initialized()
    pipeline_sha = "480971c58c845c8636a9b0e616708f07bd7d7a5c19ae3c20b9439c0c9ac90a7f"
    if upstream_pipeline is not None and digest(upstream_pipeline.read_bytes()) != pipeline_sha:
        raise ValueError("upstream pipeline source SHA256 does not match the pinned version")
    array = fixture()
    image = Image.fromarray(array, mode="RGB")
    reference = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True).preprocess(image, height=784, width=512)
    pose = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True, do_normalize=False).preprocess(
        image, height=784, width=512
    ).unsqueeze(2)
    processor = CLIPImageProcessor()
    upstream_clip = processor.preprocess(image.resize((224, 224)), return_tensors="pt").pixel_values
    old_clip = processor.preprocess(image, return_tensors="pt").pixel_values

    class VAE:
        def encode(self, value: Any) -> Any:
            self.input = value.clone()
            return SimpleNamespace(latent_dist=SimpleNamespace(mean=value, std=value))

    class Encoder:
        def __call__(self, value: Any) -> Any:
            self.input = value.clone()
            return SimpleNamespace(image_embeds=value)

    trainer = Trainer.__new__(Trainer)
    trainer.device, trainer.dtype = torch.device("cpu"), torch.float32
    trainer.vae, trainer.encoder = VAE(), Encoder()
    trainer.encode(array[None])
    fit_pose = torch.from_numpy(array[None]).permute(0, 3, 1, 2).float().div(255).unsqueeze(2)

    def stats(a: Any, b: Any) -> dict[str, Any]:
        delta = (a.float() - b.float()).abs()
        return {"max_abs": float(delta.max()), "different_values": int((delta != 0).sum()),
                "sha256_a": digest(a.contiguous().numpy().tobytes()),
                "sha256_b": digest(b.contiguous().numpy().tobytes())}

    comparisons = {"vae_float32": stats(trainer.vae.input, reference),
                   "pose_float32": stats(fit_pose, pose),
                   "clip_float32": stats(trainer.encoder.input, upstream_clip),
                   "vae_post_fp16": stats(trainer.vae.input.half(), reference.half()),
                   "pose_post_fp16": stats(fit_pose.half(), pose.half()),
                   "clip_post_fp16": stats(trainer.encoder.input.half(), upstream_clip.half()),
                   "old_direct_clip_vs_square": stats(old_clip, upstream_clip)}
    assert all(t.device.type == "cpu" for t in (reference, pose, upstream_clip, trainer.vae.input, trainer.encoder.input))
    assert reference.shape == (1, 3, 784, 512) and pose.shape == (1, 3, 1, 784, 512)
    assert upstream_clip.shape == (1, 3, 224, 224)
    # After CLIP normalization, these distinct bands remain at the image edges.
    assert upstream_clip[0, 0, 0, 0] > upstream_clip[0, 2, 0, 0]
    assert upstream_clip[0, 2, -1, 0] > upstream_clip[0, 0, -1, 0]
    assert float(reference.min()) >= -1 and float(reference.max()) <= 1
    assert float(pose.min()) >= 0 and float(pose.max()) <= 1
    assert all(comparisons[k]["max_abs"] <= 1e-6 for k in ("vae_float32", "pose_float32", "clip_float32"))
    assert all(comparisons[k]["max_abs"] == 0 for k in ("vae_post_fp16", "pose_post_fp16", "clip_post_fp16"))
    assert comparisons["old_direct_clip_vs_square"]["max_abs"] > 0
    assert not torch.cuda.is_initialized()
    modules = {"vae_processor": VaeImageProcessor, "clip_processor": CLIPImageProcessor, "trainer": Trainer}
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "diffusers", "transformers", "huggingface-hub", "Pillow", "numpy")}
    return {"scope": "CPU processor boundaries only; no weights, inference, training, GPU parity or quality claim",
            "pinned_pipeline_sha256": pipeline_sha,
            "pipeline_source_verification": "verified" if upstream_pipeline is not None else "declared-only",
            "versions": versions,
            "physical_module_sha256": {key: digest(Path(inspect.getfile(obj)).read_bytes()) for key, obj in modules.items()},
            "fixture": {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": digest(array.tobytes()),
                        "top_rgb": array[0, 0].tolist(), "bottom_rgb": array[-1, 0].tolist()},
            "shapes": {"vae": list(reference.shape), "pose": list(pose.shape), "clip": list(upstream_clip.shape)},
            "ranges": {"vae": [float(reference.min()), float(reference.max())],
                       "pose": [float(pose.min()), float(pose.max())],
                       "clip": [float(upstream_clip.min()), float(upstream_clip.max())]},
            "float32_absolute_tolerance": 1e-6, "comparisons": comparisons,
            "cuda_initialized": torch.cuda.is_initialized()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-pipeline", type=Path, help="verify the pinned upstream pipeline source bytes")
    args = parser.parse_args()
    # CLI isolation is set before importing torch or model libraries.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = audit(args.upstream_pipeline)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "max_abs": {k: v["max_abs"] for k, v in result["comparisons"].items()}}))


if __name__ == "__main__":
    main()
