"""K6b (players) 2's generator: Moore-AnimateAnyone renders each player's runs from one reference crop and the coded
pose maps. Run in the ``pointstream-jcv`` environment (env/jcv/extend-animateanyone.sh), never imported by PointStream:

    python -m experiments.tennis.k6b_generate generate --prepared TAR --weight NAME PATH ... [--profiles P,P]
        [--units U,U] --max-units N --max-frames N --steps N
    python -m experiments.tennis.k6b_generate validate

``--prepared`` is `experiments.tennis.k6b prepare`'s published.tar: per unit (window, player) ``reference``
(784x512x3 uint8), per frame ``t``, ``run``, ``pose`` (784x512x3 uint8, the DWPose body map) and ``judged``. Each run
is rendered as one video with the released long-video sampler (Pose2VideoPipeline, 24-frame contexts, overlap 4), at
the released settings (512x784, DDIM with zero-SNR v-prediction, guidance 3.5, fp16). Each (unit, profile) is written
to ``$PS_SCRATCH_DIR/publish/<unit>@<profile>.npz`` (``t``, ``crops``: the judged frames) with a ``.json``; finished
ones are skipped, so a resumed job keeps them.

Weights (``--weight``, staged one file each): ``vae/config.json``, ``vae/diffusion_pytorch_model.safetensors``
(sd-vae-ft-mse), ``image_encoder/config.json``, ``image_encoder/pytorch_model.bin`` (lambdalabs' CLIP image encoder),
and per profile ``<profile>/{denoising_unet,reference_unet,pose_guider,motion_module}.pth``. The UNets are built from
SD 1.5's UNet config (below; it is all the base model contributes, since every weight is overwritten) and each state
dict is loaded with its missing and unexpected keys recorded; a missing key stops the job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

PROFILES = ("original", "finetuned_tennis")
GEN_W, GEN_H = 512, 784
CFG = 3.5
CONTEXT_FRAMES, CONTEXT_OVERLAP = 24, 4
#: runwayml/stable-diffusion-v1-5 unet/config.json (Models/stable-diffusion-v1-5/unet/config.json).
SD15_UNET: dict[str, Any] = {
    "act_fn": "silu", "attention_head_dim": 8, "block_out_channels": [320, 640, 1280, 1280], "center_input_sample": False,
    "cross_attention_dim": 768, "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
    "downsample_padding": 1, "flip_sin_to_cos": True, "freq_shift": 0, "in_channels": 4, "layers_per_block": 2,
    "mid_block_scale_factor": 1, "norm_eps": 1e-05, "norm_num_groups": 32, "out_channels": 4, "sample_size": 64,
    "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
}
#: configs/inference/inference_v2.yaml at the fork's revision.
UNET_ADDITIONAL: dict[str, Any] = {
    "use_inflated_groupnorm": True, "unet_use_cross_frame_attention": False, "unet_use_temporal_attention": False,
    "use_motion_module": True, "motion_module_resolutions": [1, 2, 4, 8], "motion_module_mid_block": True,
    "motion_module_decoder_only": False, "motion_module_type": "Vanilla",
    "motion_module_kwargs": {"num_attention_heads": 8, "num_transformer_block": 1, "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                             "temporal_position_encoding": True, "temporal_position_encoding_max_len": 32, "temporal_attention_dim_div": 1},
}
SCHEDULER: dict[str, Any] = {"beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "linear", "clip_sample": False,
                             "steps_offset": 1, "prediction_type": "v_prediction", "rescale_betas_zero_snr": True,
                             "timestep_spacing": "trailing"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 2**20):
            digest.update(block)
    return digest.hexdigest()


def stage_dir() -> Path:
    path = Path(os.environ.get("PS_STAGE_DIR") or ".")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")


#: The reference UNet's output head, absent from reference_unet.pth: AnimateAnyone reads the ReferenceNet's attention
#: features (ReferenceAttentionControl), never its output, so upstream leaves these four at SD 1.5's values and they
#: cannot change a rendering. They stay at their initialization here and are recorded.
UNUSED_REFERENCE_HEAD = ("conv_norm_out.weight", "conv_norm_out.bias", "conv_out.weight", "conv_out.bias")


def load_checked(module: Any, state: dict[str, Any], name: str, allowed_missing: tuple[str, ...] = ()) -> dict[str, Any]:
    missing, unexpected = module.load_state_dict(state, strict=False)
    allowed = [k for k in missing if k in allowed_missing]
    missing = [k for k in missing if k not in allowed_missing]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} missing keys, e.g. {missing[:5]}")
    return {"keys": len(state), "missing": 0, "missing_allowed": allowed, "unexpected": len(unexpected),
            "unexpected_examples": unexpected[:5]}


def folder(weights: dict[str, Path], prefix: str, scratch: Path) -> Path:
    """A folder whose files link the staged weights named ``prefix/<file>`` (from_pretrained reads folders)."""
    out = scratch / "folders" / prefix
    out.mkdir(parents=True, exist_ok=True)
    for name, path in weights.items():
        if name.startswith(prefix + "/"):
            link = out / name.split("/", 1)[1]
            if not link.exists():
                link.symlink_to(path.resolve())
    return out


def loading_record(info: dict[str, Any], name: str) -> dict[str, Any]:
    missing = list(info.get("missing_keys", []))
    if missing:
        raise SystemExit(f"{name}: {len(missing)} missing keys, e.g. {missing[:5]}")
    unexpected = list(info.get("unexpected_keys", []))
    return {"missing": 0, "unexpected": len(unexpected), "unexpected_examples": unexpected[:5],
            "mismatched": len(info.get("mismatched_keys", []))}


class Renderer:
    """One profile's models in fp16 on the GPU, and the released long-video pipeline."""

    def __init__(self, weights: dict[str, Path], profile: str, device: str, scratch: Path, fit: Path | None = None):
        import torch
        from animate_anyone.models.pose_guider import PoseGuider
        from animate_anyone.models.unet_2d_condition import UNet2DConditionModel
        from animate_anyone.models.unet_3d import UNet3DConditionModel
        from animate_anyone.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
        from diffusers import AutoencoderKL, DDIMScheduler
        from transformers import CLIPVisionModelWithProjection

        dtype = torch.float16  # animation.yaml's weight_dtype

        def pth(key: str) -> dict[str, Any]:
            return torch.load(weights[f"{profile}/{key}.pth"], map_location="cpu", weights_only=True)

        record: dict[str, Any] = {"profile": profile, "dtype": "float16"}
        # The VAE and the image encoder load through their libraries' own from_pretrained, as pose2vid.py does:
        # sd-vae-ft-mse stores diffusers' deprecated attention names (query/key/value/proj_attn), which only
        # from_pretrained renames. Each sees a folder linking the staged files, and its loading info is checked.
        vae, info = AutoencoderKL.from_pretrained(str(folder(weights, "vae", scratch)), output_loading_info=True)
        record["vae"] = loading_record(info, "vae")
        encoder, info = CLIPVisionModelWithProjection.from_pretrained(str(folder(weights, "image_encoder", scratch)),
                                                                      output_loading_info=True)
        record["image_encoder"] = loading_record(info, "image_encoder")
        reference = UNet2DConditionModel.from_config({**SD15_UNET, "_class_name": "UNet2DConditionModel"})
        record["reference_unet"] = load_checked(reference, pth("reference_unet"), "reference_unet", UNUSED_REFERENCE_HEAD)
        config3d = {**SD15_UNET, "_class_name": "UNet3DConditionModel",
                    "down_block_types": ["CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"],
                    "up_block_types": ["UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"],
                    "mid_block_type": "UNetMidBlock3DCrossAttn"}
        denoising = UNet3DConditionModel.from_config(config3d, **UNET_ADDITIONAL)
        state = pth("denoising_unet")
        motion = pth("motion_module")
        record["denoising_unet"] = load_checked(denoising, {**state, **motion}, "denoising_unet+motion_module")
        record["motion_module_keys"] = len(motion)
        guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256))  # pose2vid.py's construction
        record["pose_guider"] = load_checked(guider, pth("pose_guider"), "pose_guider")
        del state, motion
        if fit is not None:  # K6b (players) 3: a LoRA fitted on the window's frames, merged into `original`
            from experiments.tennis.k6b_fit import apply_fit

            record["fit"] = {"checkpoint_sha256": sha256(fit), **apply_fit(reference, denoising, guider, fit)}
        self.pipe = Pose2VideoPipeline(vae=vae, image_encoder=encoder, reference_unet=reference, denoising_unet=denoising,
                                       pose_guider=guider, scheduler=DDIMScheduler(**SCHEDULER)).to(device, dtype=dtype)
        self.modules = {"vae": vae, "image_encoder": encoder, "reference_unet": reference, "denoising_unet": denoising, "pose_guider": guider}
        self.device = device
        self.record = record

    def render(self, reference: np.ndarray, poses: np.ndarray, steps: int, seed: int) -> np.ndarray:
        """One run: ``poses`` (n, H, W, 3) uint8 -> (n, H, W, 3) uint8."""
        import torch
        from PIL import Image

        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode():
            video = self.pipe(Image.fromarray(reference), [Image.fromarray(p) for p in poses], GEN_W, GEN_H, len(poses), steps, CFG,
                              generator=generator, context_frames=CONTEXT_FRAMES, context_overlap=CONTEXT_OVERLAP).videos
        frames = video[0].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy()  # (n, H, W, 3)
        return (frames * 255).round().astype(np.uint8)

    def probe(self, reference: np.ndarray, poses: np.ndarray) -> dict[str, Any]:
        """The first-run check (AGENTS.md): one sampling step on 8 frames under the profiler (kernels, attention)."""
        from experiments.audit.env_smoke import profile_cuda

        _, kernels = profile_cuda(lambda: self.render(reference, poses[:8], 1, 0))
        return kernels


def command_generate(args: argparse.Namespace) -> int:
    import torch

    from experiments.audit.env_smoke import device_record, model_on_cuda
    from experiments.jobs.monitor import publish_progress

    os.environ["HF_HUB_OFFLINE"] = "1"  # code never downloads (AGENTS.md); every file is staged
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    began = time.time()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    weights = {name: Path(path) for name, path in args.weight}
    root = scratch / "prepared"
    if not root.is_dir():
        root.mkdir(parents=True)
        with tarfile.open(args.prepared) as archive:
            archive.extractall(root, filter="data")
    prepared = json.loads((root / "publish" / "prepare.json").read_text())
    names = [u["unit"] for w in prepared["windows"] for u in w["units"]]
    if args.units:
        names = [n for n in names if n in args.units.split(",")]
    names = names[:args.max_units or None]
    profiles = [p for p in args.profiles.split(",") if p]  # empty: only the --fit profiles
    if args.fit_archive:  # k6b_fit's published.tar files: --fit names a member of one (placeholders must be whole arguments)
        members: dict[str, Path] = {}
        for n, tar in enumerate(args.fit_archive):
            fit_root = scratch / "fit-archive" / str(n)
            if not fit_root.is_dir():
                fit_root.mkdir(parents=True)
                with tarfile.open(tar) as archive:
                    archive.extractall(fit_root, filter="data")
            for path in (fit_root / "publish").glob("*.pt"):
                if path.name in members:
                    raise SystemExit(f"{path.name} is in two fit archives")
                members[path.name] = path
        fits = {name: members[member] for name, member in args.fit}
    else:
        fits = {name: Path(path) for name, path in args.fit}
    profiles += [name for name in fits if name not in profiles]
    device = "cuda"
    limit = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "24000"))
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    records: dict[str, Any] = {}
    timings: list[dict[str, Any]] = []
    for profile in profiles:
        todo = [n for n in names if not (publish / f"{n}@{profile}.json").exists()]
        if not todo:
            continue
        loaded = time.time()
        renderer = Renderer(weights, "original" if profile in fits else profile, device, scratch, fits.get(profile))
        record = {**renderer.record, "load_seconds": round(time.time() - loaded, 1),
                  "on_cuda": {k: model_on_cuda(m) for k, m in renderer.modules.items()},
                  "params": {k: int(sum(p.numel() for p in m.parameters())) for k, m in renderer.modules.items()}}
        for name in todo:
            unit = np.load(root / "publish" / "units" / f"{name}.npz")
            seed = int(hashlib.sha256(name.encode()).hexdigest()[:7], 16)  # k6b.unit_seed
            if "kernels" not in record:
                record["kernels"] = renderer.probe(unit["reference"], unit["pose"])
            t, run, judged = unit["t"], unit["run"], unit["judged"]
            keep_t, keep_crops = [], []
            frames = 0
            started = time.time()
            for r in np.unique(run):
                rows = np.nonzero((run == r) & (t >= args.from_t))[0][:args.max_frames or None]  # step 3: held-out frames
                if not len(rows):
                    continue
                if args.max_frames and frames >= args.max_frames:
                    break
                video = renderer.render(unit["reference"], unit["pose"][rows], args.steps, seed * 1009 + int(r))
                frames += len(rows)
                for i, row in enumerate(rows):
                    if judged[row] or args.max_frames:
                        keep_t.append(int(t[row]))
                        keep_crops.append(video[i])
                print(f"phase=generate profile={profile} unit={name} run={int(r)} frames={len(rows)}", flush=True)
            seconds = time.time() - started
            path = publish / f"{name}@{profile}.npz"
            crops = np.stack(keep_crops) if keep_crops else np.zeros((0, GEN_H, GEN_W, 3), np.uint8)
            np.savez_compressed(path, t=np.asarray(keep_t, np.int64), crops=crops)
            write_json(path.with_suffix(".json"), {"unit": name, "profile": profile, "frames_generated": frames, "kept": len(keep_t),
                                                   "seconds": round(seconds, 1), "seconds_per_frame": seconds / max(1, frames),
                                                   "steps": args.steps, "seed": seed, "npz_sha256": sha256(path),
                                                   "crop_std_min": float(min((c.std() for c in crops), default=0.0))})
            timings.append({"unit": name, "profile": profile, "frames": frames, "seconds": round(seconds, 1)})
            publish_progress(os.environ.get("PS_STAGE", "local"), len(timings))  # one finished (unit, profile)
        records[profile] = record
        del renderer
        torch.cuda.empty_cache()
    with tarfile.open(stage_dir() / "published.tar", "w") as tar:
        tar.add(publish, arcname="publish")
    write_json(stage_dir() / "k6b_generate.json", {
        "command": "generate", "stage": os.environ.get("PS_STAGE"), "units": names, "profiles": profiles,
        "steps": args.steps, "max_frames": args.max_frames, "from_t": args.from_t,
        "fits": {k: {"path": str(v), "sha256": sha256(v)} for k, v in fits.items()}, "device": device_record(args.gpu_class), "models": records,
        "timings": timings, "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3), "gpu_memory_limit_mib": limit,
        "inputs": {"prepared": sha256(Path(args.prepared)), "weights": {k: sha256(v) for k, v in sorted(weights.items())}},
        "settings": {"size": [GEN_W, GEN_H], "cfg": CFG, "context": [CONTEXT_FRAMES, CONTEXT_OVERLAP], "scheduler": SCHEDULER},
        "versions": versions(), "seconds": round(time.time() - began, 2)})
    return 0


def versions() -> dict[str, str]:
    import diffusers
    import torch
    import transformers

    return {"torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__}


def command_validate(args: argparse.Namespace) -> int:
    import tempfile

    stage = Path(args.stage) if args.stage else stage_dir()
    result = json.loads((stage / "k6b_generate.json").read_text())
    checks: dict[str, bool] = {"device_class": bool(result["device"]["class_matches"]),
                               "peak_within_declaration": result["gpu_peak_gib"] * 1024 <= result["gpu_memory_limit_mib"]}
    for profile, rec in result["models"].items():
        checks[f"{profile}:on_cuda"] = all(rec["on_cuda"].values())
        checks[f"{profile}:no_missing_keys"] = all(rec[k]["missing"] == 0 for k in ("vae", "image_encoder", "reference_unet", "denoising_unet", "pose_guider"))
        checks[f"{profile}:motion_module_loaded"] = rec["motion_module_keys"] > 100
        checks[f"{profile}:kernels_on_gpu"] = rec["kernels"]["kernel_launches"] > 0
    with tempfile.TemporaryDirectory(prefix="k6bg-validate-", dir="/tmp") as temp:
        with tarfile.open(stage / "published.tar") as archive:
            archive.extractall(temp, filter="data")
        for name in result["units"]:
            for profile in result["profiles"]:
                path = Path(temp) / "publish" / f"{name}@{profile}.npz"
                meta = json.loads(path.with_suffix(".json").read_text())
                a = np.load(path)
                checks[f"{name}@{profile}:hash"] = sha256(path) == meta["npz_sha256"]
                checks[f"{name}@{profile}:shape"] = a["crops"].shape[1:] == (GEN_H, GEN_W, 3) and len(a["t"]) == len(a["crops"]) > 0
                checks[f"{name}@{profile}:not_flat"] = meta["crop_std_min"] > 5.0
    report = {"passed": all(checks.values()), "checks": checks}
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--prepared", required=True)
    gen.add_argument("--weight", action="append", nargs=2, required=True, metavar=("NAME", "PATH"),
                     help="NAME as in the docstring (two arguments: fleet placeholders must be whole arguments)")
    gen.add_argument("--profiles", default=",".join(PROFILES))
    gen.add_argument("--units", default="", help="comma-separated unit names (default: every prepared unit)")
    gen.add_argument("--max-units", type=int, default=0)
    gen.add_argument("--max-frames", type=int, default=0, help="the smoke: render only this many frames of a unit")
    gen.add_argument("--steps", type=int, default=30, help="DDIM steps (pose2vid.py's default: 30)")
    gen.add_argument("--fit", action="append", nargs=2, default=[], metavar=("PROFILE", "CHECKPOINT"),
                     help="render a profile named PROFILE: `original` with k6b_fit's checkpoint merged")
    gen.add_argument("--fit-archive", action="append", default=[], help="k6b_fit's published.tar (repeatable); --fit names a member")
    gen.add_argument("--from-t", type=int, default=0, help="render only frames from here (step 3's held-out frames)")
    gen.add_argument("--gpu-class", default="RTX")
    val = sub.add_parser("validate")
    val.add_argument("--stage", default="")
    args = ap.parse_args(argv)
    return {"generate": command_generate, "validate": command_validate}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
