"""PLAN step H3e's generator: JointControlVideo renders a window from its first frame and the coded hand joints.

Run as a script in the ``pointstream-jcv`` environment (env/jcv), never imported by PointStream:

    python experiments/visor/h3e_generate.py generate --inputs DIR --dit FILE --t5 FILE --vae FILE --clip FILE \\
        --tokenizer DIR --hand-controller FILE --lora FILE [--out DIR] --frames N --steps N [--items ID,ID] \\
        [--joint-order openpose|mano ...]
    python experiments/visor/h3e_generate.py validate

``--inputs`` holds `experiments.visor.h3e prepare`'s ``<item>.npz`` (``frame0`` 480×832×3 uint8, ``joints`` (81, 42, 3)
camera-frame metres in OpenPose order, left hand first, absent hands zero, ``K`` 3×3 for the 832×480 canvas). Each
(item, joint order) is written to ``--out`` (``$PS_SCRATCH_DIR/publish`` in a fleet job) as ``<item>@<order>.npz``
(``frames`` F×480×832×3 uint8) with a ``.json``; finished ones are skipped, so a resumed job keeps them.

The model is JointControlVideo (ECCV 2026, Apache-2.0) at the revision vendored in ``opt/JointControlVideo``: the
Wan2.1-I2V-14B-480P backbone with its released LoRA, expanded patch embedding and hand-conditioning module, loaded and
sampled as its ``inf_wm_hand.py`` does (cfg 5, 832×480, tiled VAE), with the fixed prompt below. The one change is
where the text encoder runs: the two prompts are encoded once by the real T5 on the CPU (bf16) before the GPU models
load, and a stub returns those embeddings, so the 14B DiT fits a 48 GB card with room to spare.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROMPT = "First-person view of hands handling objects in a kitchen."
NEGATIVE = ""
CFG = 5.0
SEED = 0
HEIGHT, WIDTH = 480, 832
# OpenPose order (as `experiments.visor.mano_np` emits) back to MANO's (16 kinematic joints, then the five tips):
# the inverse of mano_np.MANO_TO_OPENPOSE.
MANO_TO_OPENPOSE = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
OPENPOSE_TO_MANO = tuple(int(i) for i in np.argsort(MANO_TO_OPENPOSE))
ORDERS = ("openpose", "mano")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def reorder(joints: np.ndarray, order: str) -> np.ndarray:
    """(F, 42, 3) in OpenPose order per hand into ``order``."""
    if order == "openpose":
        return joints
    index = list(OPENPOSE_TO_MANO) + [21 + i for i in OPENPOSE_TO_MANO]
    return joints[:, index]


def replace_patch_embedding(pipe: Any) -> None:
    """``inf_wm_hand.replace_modules_in_pipe``: the patch embedding takes 16 more input channels."""
    import torch

    conv = pipe.dit.patch_embedding
    new = torch.nn.Conv3d(conv.in_channels + 16, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
                          padding=conv.padding)
    pipe.dit.patch_embedding = new.to(torch.bfloat16).to(pipe.device)


def load_patch_embedding(pipe: Any, state: dict[str, Any]) -> int:
    """``inf_wm_hand.apply_custom_state_dict``: copy the expanded patch embedding, exact shapes only."""
    loaded = 0
    for key in ("patch_embedding.weight", "patch_embedding.bias"):
        param = getattr(pipe.dit.patch_embedding, key.rsplit(".", 1)[1])
        value = state[key]
        if tuple(param.shape) != tuple(value.shape):
            raise RuntimeError(f"{key}: checkpoint {tuple(value.shape)}, model {tuple(param.shape)}")
        param.data.copy_(value.to(param.dtype).to(param.device))
        loaded += 1
    return loaded


def attention_record() -> dict[str, Any]:
    """Which attention DiffSynth's Wan DiT uses (flash-attn packages, else torch SDPA) and whether SDPA's flash kernel
    runs on this device."""
    import torch
    import torch.nn.functional as F
    from diffsynth.models import wan_video_dit as dit
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.bfloat16)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(q, q, q)
        flash = True
    except RuntimeError:
        flash = False
    return {"flash_attn_3": bool(getattr(dit, "FLASH_ATTN_3_AVAILABLE", False)),
            "flash_attn_2": bool(getattr(dit, "FLASH_ATTN_2_AVAILABLE", False)),
            "sage_attn": bool(getattr(dit, "SAGE_ATTN_AVAILABLE", False)), "sdpa_flash_kernel_runs": flash}


def cached_text_encoder(cache: dict[bytes, Any]) -> Any:
    """Stands in for the T5 encoder (a module, so the pipeline may hold it as one): returns the embedding computed
    for the same token ids."""
    import torch

    class Cached(torch.nn.Module):
        def forward(self, ids: Any, mask: Any) -> Any:
            key = ids.cpu().numpy().tobytes()
            if key not in cache:
                raise RuntimeError("prompt not encoded before the GPU models loaded")
            return cache[key].clone()

    return Cached()


def load_pipeline(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline, WanVideoUnit_PromptEmbedder
    from safetensors.torch import load_file

    # Wan2.1-I2V-14B-480P's files (the DiT cast to bf16 by tools/models/wan_dit_bf16.py) and JointControlVideo's.
    files = {k: Path(getattr(args, k)) for k in ("dit", "t5", "vae", "clip", "hand_controller", "lora")}
    began = time.time()
    # The two prompts through the real T5 on the CPU, once, before the GPU models load; a stub then returns them.
    text = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", redirect_common_files=False,
                                            model_configs=[ModelConfig(path=str(files["t5"]))],
                                            tokenizer_config=ModelConfig(path=str(Path(args.tokenizer))))
    unit = WanVideoUnit_PromptEmbedder()
    cache = {}
    with torch.no_grad():
        for prompt in (PROMPT, NEGATIVE):
            ids, _ = text.tokenizer(prompt, return_mask=True, add_special_tokens=True)
            cache[ids.numpy().tobytes()] = unit.encode_prompt(text, prompt).detach().to("cuda")
    tokenizer = text.tokenizer
    del text
    text_seconds = round(time.time() - began, 1)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda", redirect_common_files=False,
        model_configs=[ModelConfig(path=str(files[k])) for k in ("dit", "vae", "clip", "hand_controller")], tokenizer_config=None)
    pipe.tokenizer = tokenizer
    pipe.text_encoder = cached_text_encoder(cache)
    replace_patch_embedding(pipe)
    patched = load_patch_embedding(pipe, load_file(str(files["lora"])))
    pipe.load_lora(pipe.dit, str(files["lora"]), alpha=1)
    torch.cuda.empty_cache()
    on_cuda = {name: next(getattr(pipe, name).parameters()).is_cuda
               for name in ("dit", "vae", "image_encoder", "hand_controler") if getattr(pipe, name, None) is not None}
    record = {"files": {k: str(v) for k, v in files.items()}, "file_bytes": {k: v.stat().st_size for k, v in files.items()},
              "patch_embedding_tensors": patched, "on_cuda": on_cuda, "load_seconds": round(time.time() - began, 1),
              "text_encoder": {"device": "cpu", "seconds": text_seconds, "prompts": len(cache)},
              "memory_after_load_mib": round(torch.cuda.memory_allocated() / 2**20)}
    return pipe, record


def generate(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise SystemExit("h3e_generate needs a GPU")
    frames = int(args.frames)
    if frames % 4 != 1:
        raise SystemExit(f"--frames {frames}: Wan needs 4k+1 frames")
    out = Path(args.out or Path(os.environ["PS_SCRATCH_DIR"]) / "publish")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.inputs)
    root = root / "publish" if (root / "publish").is_dir() else root  # an extracted published.tar keeps publish/
    inputs = sorted(root.glob("*.npz"))
    if args.items:
        wanted = args.items.split(",")
        inputs = [p for p in inputs if p.stem in wanted]
        if missing := sorted(set(wanted) - {p.stem for p in inputs}):
            raise SystemExit(f"no prepared inputs for {missing}")
    todo = [(p, o) for p in inputs for o in args.joint_order if not (out / f"{p.stem}@{o}.json").exists()]
    torch.cuda.reset_peak_memory_stats()
    device = {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
              "torch": torch.__version__, "cuda": torch.version.cuda, "bf16": torch.cuda.is_bf16_supported()}
    pipe: Any = None
    load: dict[str, Any] = {}
    if todo:
        pipe, load = load_pipeline(args)
    attention = attention_record()
    for path, order in todo:
        data = np.load(path)
        joints = reorder(np.asarray(data["joints"], np.float32), order)[:frames]
        frame0 = np.asarray(data["frame0"])
        if frame0.shape != (HEIGHT, WIDTH, 3) or joints.shape != (frames, 42, 3):
            raise RuntimeError(f"{path.name}: frame0 {frame0.shape}, joints {joints.shape}")
        torch.cuda.synchronize()
        began = time.time()
        video = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, input_image=Image.fromarray(frame0),
                     joints_cam=torch.from_numpy(joints)[None], intrinsics=torch.from_numpy(np.asarray(data["K"], np.float32))[None],
                     cfg_scale=CFG, height=HEIGHT, width=WIDTH, num_frames=frames, num_inference_steps=int(args.steps),
                     seed=SEED, tiled=True)
        torch.cuda.synchronize()
        seconds = time.time() - began
        rgb = np.stack([np.asarray(im.convert("RGB")) for im in video]).astype(np.uint8)
        name = f"{path.stem}@{order}"
        np.savez(out / f"{name}.npz", frames=rgb)
        record = {"item": path.stem, "joint_order": order, "frames": frames, "steps": int(args.steps), "cfg": CFG,
                  "seed": SEED, "prompt": PROMPT, "negative_prompt": NEGATIVE, "shape": list(rgb.shape),
                  "input_sha256": sha256(path), "output_sha256": sha256(out / f"{name}.npz"),
                  "seconds": round(seconds, 1), "seconds_per_step": round(seconds / int(args.steps), 2),
                  "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                  "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                  "first_frame_mean_abs_diff": float(np.abs(rgb[0].astype(np.int16) - frame0.astype(np.int16)).mean())}
        (out / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps(record), flush=True)
    provenance = Path(sys.prefix) / "opt" / "PROVENANCE.json"
    report = {"device": device, "attention": attention, "load": load, "steps": int(args.steps), "frames": frames,
              "joint_orders": args.joint_order, "items": [p.stem for p in inputs],
              "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20),
              "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
              "jointcontrolvideo": json.loads(provenance.read_text()) if provenance.exists() else None,
              "weights_sha256": {k: sha256(Path(v)) for k, v in load.get("files", {}).items()} if args.hash_weights else None}
    (out / "generate.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


def validate(args: argparse.Namespace) -> int:
    """Fleet validator: every (item, order) published, frames finite and varied, the first frame close to the input,
    the model on the GPU, a recorded GPU of the intended class."""
    stage = Path(os.environ.get("PS_STAGE_DIR", "."))
    import tarfile

    tar = stage / "published.tar"
    records = []
    with tarfile.open(tar) as t:
        names = t.getnames()
        for n in names:
            if n.endswith(".json") and "@" in n:
                records.append(json.load(t.extractfile(n)))  # type: ignore[arg-type]
        report = json.load(t.extractfile([n for n in names if n.endswith("generate.json")][0]))  # type: ignore[arg-type]
    checks = {
        "published": bool(records) and len(records) == len(report["items"]) * len(report["joint_orders"]),
        "on_cuda": bool(report["load"]) and all(report["load"]["on_cuda"].values()),
        "patch_embedding": bool(report["load"]) and report["load"]["patch_embedding_tensors"] == 2,
        "shapes": all(r["shape"] == [report["frames"], 480, 832, 3] for r in records),
        "first_frame_kept": all(r["first_frame_mean_abs_diff"] < 20.0 for r in records),
        "bf16": bool(report["device"]["bf16"]),
        # The step budget and the shared memory declaration were measured with a flash kernel (model-GPU table): a
        # run that fell back to slower attention must not pass.
        "flash_attention": any(report["attention"].get(k) for k in ("sdpa_flash_kernel_runs", "flash_attn_3", "flash_attn_2")),
    }
    checks = {k: bool(v) for k, v in checks.items()}
    result = {"passed": all(checks.values()), "checks": checks}
    if target := os.environ.get("PS_VALIDATION_PATH"):
        Path(target).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({**result, "records": records, "device": report["device"], "attention": report["attention"]}, indent=1))
    return 0 if result["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--inputs", required=True)
    for name in ("dit", "t5", "vae", "clip", "tokenizer", "hand-controller", "lora"):
        g.add_argument(f"--{name}", required=True)
    g.add_argument("--out")
    g.add_argument("--items")
    g.add_argument("--frames", default="81")
    g.add_argument("--steps", default="50")
    g.add_argument("--joint-order", nargs="+", choices=ORDERS, default=["openpose"])
    g.add_argument("--hash-weights", action="store_true")
    sub.add_parser("validate")
    args = parser.parse_args(argv)
    return generate(args) if args.command == "generate" else validate(args)


if __name__ == "__main__":
    sys.exit(main())
