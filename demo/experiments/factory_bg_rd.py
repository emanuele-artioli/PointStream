"""Fine-tune one background codec per factory and score it like AV1.

Training uses every kept inpainted second. Factory 001 is clip 1 plus the
clip 3 press seconds. Clip 3 at 210s, 240s, and 420s stays out. Factory 002
is its own room. DCVC-UF LD, HT-S, and HT-L fine-tune from the CVPR 2026
checkpoints across the lambda range that keeps all 64 QPs. HNeRV fits the
same frames. DCVC-FM and DCVC-RT have no trainer; they are scored frozen.

Rate is the bytes of one segment. AV1 and the neural bitstream see the same
cut. A short segment is a shorter delay and a higher rate, because prediction
and the sequence header stop at the cut. Quantization (QP, or HNeRV embedding
bits) is the rate knob inside a segment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from demo.pipeline.maps.av1_crf import AV1_CRF, AV1_LADDER, AV1_PRESET, av1_output_args

DEMO_ROOT = Path("/home/itec/emanuele/Datasets/pointstream-demo")
WORK_ROOT = Path("/home/itec/emanuele/pointstream-data/jobs/factory-bg-rd")
DCVC_ROOT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/src/DCVC")
HNERV_ROOT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/src/HNeRV")
DCVC_PYTHON = Path("/home/itec/emanuele/.conda/envs/pointstream-dcvc/bin/python")
HNERV_PYTHON = Path("/home/itec/emanuele/.conda/envs/pointstream/bin/python")
FFMPEG = "/opt/local/bin/ffmpeg"
FPS = 30
CLIP3_DROP_S = frozenset({210.0, 240.0, 420.0})
# One intra at the start of a segment. -1 is not "every frame" in DCVC-UF.
ONE_INTRA = -1
SEGMENT_FRAMES = (1, 8, 32, 300)
# HT codes 8-frame chunks. 300 is not a multiple of 8; 296 is the longest
# span inside the 10-second hold-out that is.
HT_LONG_SEGMENT = 296
UF_STRUCTURES = ("ld", "hts", "htl")
UF_LAMBDAS = (1.0, 768.0)
# Schedule lengths in train_video.get_training_strategy, stage0 then stage1.
# Stage2/3 move to 512 patches and are a later memory check, not this train.
UF_EPOCHS = {
    "ld": (56, 38),
    "hts": (46, 11),
    "htl": (46, 11),
}
SCORE_QPS = (0, 21, 42, 63)
# 1080x1920 divided by the encoder stride 5*3*2*2*2. The training command
# uses enc_dim 64_0.5, so half of the 1.5M budget becomes this grid.
HNERV_EMBED_HW = 144
HNERV_MODELSIZE = 1.5
HNERV_EMBED_RATIO = 0.5
UF_IMAGE = DCVC_ROOT / "checkpoints" / "cvpr2026_image.pth.tar"
UF_VIDEO = {
    "ld": DCVC_ROOT / "checkpoints" / "cvpr2026_video_ld.pth.tar",
    "hts": DCVC_ROOT / "checkpoints" / "cvpr2026_video_hts.pth.tar",
    "htl": DCVC_ROOT / "checkpoints" / "cvpr2026_video_htl.pth.tar",
}
SOURCES = {
    "factory001": (
        ("clip_01_factory001_worker001_00001", "f000000-f035129", frozenset()),
        ("clip_03_factory001_worker001_00000", "f000000-f012629", CLIP3_DROP_S),
    ),
    "factory002": (
        ("factory002_worker001_00000", "f000000-f035129", frozenset()),
    ),
}


def source_frame_names(start_s: float, frames: int = FPS, fps: int = FPS) -> list[str]:
    origin = int(round(start_s * fps))
    return [f"{origin + offset:06d}.jpg" for offset in range(frames)]


def kept_starts(batches: list[dict], drop_s: frozenset[float]) -> list[float]:
    starts = []
    for batch in batches:
        start = float(batch["start_s"])
        if batch.get("train") is False or start in drop_s:
            continue
        starts.append(start)
    return starts


def segments_for(structure: str) -> tuple[int, ...]:
    if structure in {"hts", "htl"}:
        return (1, 8, 32, HT_LONG_SEGMENT)
    return SEGMENT_FRAMES


def hnerv_embed_dim(train_frames: int) -> int:
    """Embedding channels implied by the 64_0.5, 1.5M training command."""
    return int(HNERV_EMBED_RATIO * HNERV_MODELSIZE * 1e6 / train_frames / HNERV_EMBED_HW)


def hnerv_decoder_params(train_frames: int) -> float:
    embed = hnerv_embed_dim(train_frames)
    return HNERV_MODELSIZE * 1e6 - embed * HNERV_EMBED_HW * train_frames


def hnerv_modelsize_for_segment(train_frames: int, length: int) -> float:
    """Keep the trained decoder size when the segment is shorter than the fit.

    HNeRV rebuilds the decoder from ``modelsize`` minus the embedding budget,
    and the embedding budget grows with the frame count. Passing the trained
    embedding width as an integer ratio (``64_<width>``) freezes that width,
    and this modelsize puts the saved decoder back on the same parameter count.
    """
    embed = hnerv_embed_dim(train_frames)
    return (hnerv_decoder_params(train_frames) + embed * HNERV_EMBED_HW * length) / 1e6


def segment_starts(n_frames: int, length: int) -> list[int]:
    """Non-overlapping cuts. A tail shorter than ``length`` is left out."""
    if length <= 0 or n_frames < length:
        return []
    return list(range(0, n_frames - length + 1, length))


def write_video_folder(
    dest: Path,
    sequences: list[tuple[str, list[Path]]],
) -> Path:
    """Symlink each second under the shared 000000.jpg–000029.jpg names."""
    dest.mkdir(parents=True, exist_ok=True)
    seqs = []
    for name, sources in sequences:
        if len(sources) != FPS:
            raise ValueError(f"{name} has {len(sources)} frames, expected {FPS}")
        folder = dest / name
        folder.mkdir(parents=True, exist_ok=True)
        for index, src in enumerate(sources):
            if not src.is_file():
                raise FileNotFoundError(src)
            link = folder / f"{index:06d}.jpg"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
        seqs.append({"path": name, "height": 1080, "width": 1920, "seq_length": FPS})
    description = {
        "seqs": seqs,
        "frames": [f"{index:06d}.jpg" for index in range(FPS)],
    }
    (dest / "description.json").write_text(json.dumps(description))
    return dest


def factory_sequences(demo_root: Path, factory: str) -> list[tuple[str, list[Path]]]:
    sequences = []
    for stem, span, drop_s in SOURCES[factory]:
        base = demo_root / stem / span
        batches = json.loads((base / "batches.json").read_text())["batches"]
        inpainted = base / "inpainted"
        for start in kept_starts(batches, drop_s):
            sources = [inpainted / name for name in source_frame_names(start)]
            sequences.append((f"{stem}_t{int(start):06d}", sources))
    if not sequences:
        raise RuntimeError(f"{factory} has no training seconds")
    return sequences


def prepare_factory(demo_root: Path, factory: str, dest: Path) -> Path:
    description = dest / "description.json"
    if description.is_file():
        return dest
    return write_video_folder(dest, factory_sequences(demo_root, factory))


def _dcvc_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DCVC_ROOT)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def train_uf(
    factory: str,
    structure: str,
    dataset: Path,
    save_root: Path,
    *,
    stage0_epochs: int | None = None,
    stage1_epochs: int | None = None,
    batch_size: int = 8,
    stage1_batch_size: int = 4,
) -> Path:
    if structure not in UF_STRUCTURES:
        raise ValueError(structure)
    if not DCVC_PYTHON.is_file():
        raise FileNotFoundError(DCVC_PYTHON)
    default0, default1 = UF_EPOCHS[structure]
    stage0_epochs = default0 if stage0_epochs is None else stage0_epochs
    stage1_epochs = default1 if stage1_epochs is None else stage1_epochs
    image = UF_IMAGE
    video = UF_VIDEO[structure]
    if not image.is_file() or not video.is_file():
        raise FileNotFoundError(f"missing UF checkpoint for {structure}")
    stage0 = save_root / structure / "s0"
    stage1 = save_root / structure / "s1"
    _run_uf_stage(
        structure, dataset, stage0, image, video,
        epochs=stage0_epochs, batch_size=batch_size, scheduling="stage0",
    )
    if stage1_epochs <= 0:
        return stage0 / "ckpt.pth.tar"
    _run_uf_stage(
        structure, dataset, stage1, image, stage0 / "ckpt.pth.tar",
        epochs=stage1_epochs, batch_size=stage1_batch_size, scheduling="stage1",
    )
    return stage1 / "ckpt.pth.tar"


def _run_uf_stage(
    structure: str,
    dataset: Path,
    save_dir: Path,
    image: Path,
    pretrain: Path,
    *,
    epochs: int,
    batch_size: int,
    scheduling: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(DCVC_PYTHON), "train_video.py",
        "--model_structure", structure,
        "--model_path_i", str(image),
        "--train_dataset", str(dataset),
        "--save_dir", str(save_dir),
        "--lambdas", str(UF_LAMBDAS[0]), str(UF_LAMBDAS[1]),
        "--training_scheduling", scheduling,
        "--pretrain_path", str(pretrain),
        "--batch_size", str(batch_size),
        "-n", "2",
        "-e", str(epochs),
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(DCVC_ROOT), env=_dcvc_env(), check=True)


def hnerv_import_stub(root: Path) -> Path:
    """HNeRV imports EncodedVideo and never uses it. No local env ships pytorchvideo."""
    package = root / "pytorchvideo" / "data"
    package.mkdir(parents=True, exist_ok=True)
    for parent in (root / "pytorchvideo", package):
        init = parent / "__init__.py"
        if not init.is_file():
            init.write_text("")
    encoded = package / "encoded_video.py"
    if not encoded.is_file():
        encoded.write_text("class EncodedVideo:\n    pass\n")
    # decord is only constructed for a video file. Image folders use torchvision.
    decord = root / "decord.py"
    if not decord.is_file():
        decord.write_text(
            "class _Bridge:\n"
            "    def set_bridge(self, _name):\n"
            "        return None\n"
            "bridge = _Bridge()\n"
            "class VideoReader:\n"
            "    def __init__(self, *_args, **_kwargs):\n"
            "        raise RuntimeError('image folders do not use decord')\n"
        )
    return root


def _ensure_target_package(python: Path, stub: Path, module: str, package: str) -> None:
    """Install a pure-Python extra beside the job. --no-deps keeps the env torch."""
    probe = subprocess.run(
        [str(python), "-c", f"import {module}"],
        env={**os.environ, "PYTHONPATH": str(stub)},
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        return
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "--target", str(stub), package],
        check=True,
    )


def train_hnerv(factory: str, frames: Path, save_dir: Path, *, epochs: int = 30) -> None:
    """Fit HNeRV to the factory frames at native 1080p. The decoder is the setup cost."""
    if not HNERV_PYTHON.is_file():
        raise FileNotFoundError(HNERV_PYTHON)
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(HNERV_PYTHON), "train_nerv_all.py",
        "--data_path", str(frames),
        "--vid", factory,
        "--data_split", "1_1_1",
        "--crop_list", "-1",
        "--resize_list", "-1",
        # 1080x1920 stays divisible by 5,3,2,2,2, so the embedding is 9x16.
        "--enc_strds", "5", "3", "2", "2", "2",
        "--dec_strds", "5", "3", "2", "2", "2",
        # 16-d embeddings of 9x16 over a factory of frames exceed 1.5M.
        # A ratio below 1 assigns that fraction of modelsize to the embeddings.
        "--enc_dim", "64_0.5",
        "--ks", "0_1_5",
        "--lower_width", "12",
        "--modelsize", "1.5",
        "--epochs", str(epochs),
        "--batchSize", "1",
        "--loss", "Fusion6",
        "--quant_model_bit", "8",
        "--quant_embed_bit", "6",
        "--outf", str(save_dir),
        "--workers", "2",
    ]
    env = os.environ.copy()
    stub = hnerv_import_stub(save_dir.parent.parent.parent / "hnerv-stub")
    _ensure_target_package(HNERV_PYTHON, stub, "dahuffman", "dahuffman")
    _ensure_target_package(HNERV_PYTHON, stub, "pytorch_msssim", "pytorch-msssim")
    env["PYTHONPATH"] = os.pathsep.join((str(stub), str(HNERV_ROOT)))
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(HNERV_ROOT), env=env, check=True)


def av1_segment_cmd(
    src: Path,
    dest: Path,
    *,
    scale: str | None,
    gop: int,
    ffmpeg: str = FFMPEG,
) -> list[str]:
    """CRF 63 preset 7, with the keyframe interval equal to the segment."""
    return [
        ffmpeg, "-y", "-i", str(src),
        *av1_output_args(scale),
        "-g", str(gop),
        "-svtav1-params", f"keyint={gop}:keyint-min={gop}",
        str(dest),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--factory", required=True, choices=sorted(SOURCES))
    prepare.add_argument("--demo-root", type=Path, default=DEMO_ROOT)
    prepare.add_argument("--dest", type=Path)

    train = commands.add_parser("train-uf")
    train.add_argument("--factory", required=True, choices=sorted(SOURCES))
    train.add_argument("--structure", required=True, choices=UF_STRUCTURES)
    train.add_argument("--demo-root", type=Path, default=DEMO_ROOT)
    train.add_argument("--work-root", type=Path, default=WORK_ROOT)
    train.add_argument("--stage0-epochs", type=int)
    train.add_argument("--stage1-epochs", type=int)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--stage1-batch-size", type=int, default=4)

    hnerv = commands.add_parser("train-hnerv")
    hnerv.add_argument("--factory", required=True, choices=sorted(SOURCES))
    hnerv.add_argument("--demo-root", type=Path, default=DEMO_ROOT)
    hnerv.add_argument("--work-root", type=Path, default=WORK_ROOT)
    hnerv.add_argument("--epochs", type=int, default=30)

    fill = commands.add_parser("fill-holdout")
    fill.add_argument("--work-root", type=Path, default=WORK_ROOT)
    fill.add_argument("--holdouts", type=Path, default=DEMO_ROOT / "holdouts" / "holdouts.json")
    fill.add_argument("--max-chunks", type=int)

    score_cmd = commands.add_parser("score")
    score_cmd.add_argument("--work-root", type=Path, default=WORK_ROOT)
    score_cmd.add_argument("--factory", choices=sorted(SOURCES))
    score_cmd.add_argument("--codecs", nargs="+", required=True)
    score_cmd.add_argument("--limit", type=int)

    args = parser.parse_args(argv)
    if args.action == "prepare":
        dest = args.dest or (WORK_ROOT / "datasets" / args.factory)
        path = prepare_factory(args.demo_root, args.factory, dest)
        print(path)
        return 0
    if args.action == "train-uf":
        dataset = prepare_factory(
            args.demo_root, args.factory, args.work_root / "datasets" / args.factory,
        )
        ckpt = train_uf(
            args.factory, args.structure, dataset,
            args.work_root / "checkpoints" / args.factory,
            stage0_epochs=args.stage0_epochs,
            stage1_epochs=args.stage1_epochs,
            batch_size=args.batch_size,
            stage1_batch_size=args.stage1_batch_size,
        )
        print(ckpt)
        return 0
    if args.action == "train-hnerv":
        dataset = prepare_factory(
            args.demo_root, args.factory, args.work_root / "datasets" / args.factory,
        )
        # One directory of canonical names cannot hold every second, because
        # VideoFolder shares filenames and HNeRV lists a single folder. Point
        # HNeRV at a concatenated view written beside the dataset.
        flat = args.work_root / "datasets" / f"{args.factory}-flat"
        _flatten(dataset, flat)
        train_hnerv(args.factory, flat, args.work_root / "checkpoints" / args.factory / "hnerv", epochs=args.epochs)
        return 0
    if args.action == "fill-holdout":
        fill_holdouts(args.work_root, args.holdouts, max_chunks=args.max_chunks)
        return 0
    if args.action == "score":
        score(
            args.work_root,
            args.factory,
            tuple(args.codecs),
            limit=args.limit,
        )
        return 0
    raise AssertionError(args.action)


def sam_union_allowing_empty(builder, frames: Path, work: Path):
    """Arm/hand union. A chunk where SAM 3.1 finds nothing is an empty mask."""
    work.mkdir(parents=True, exist_ok=True)
    cached = work / "union.npy"
    if cached.is_file():
        return list(builder.np.load(cached))
    masks = builder.build("sam31").segment(frames, builder.load_domain("egocentric"))
    masks.save(work)
    union = builder.np.stack([masks.foreground(index) for index in range(len(masks))])
    builder.np.save(cached, union)
    return list(union)


def fill_holdouts(work: Path, holdouts: Path, *, max_chunks: int | None) -> None:
    """Fill hold-out backgrounds with the same hand/arm DiffuEraser path as training."""
    import importlib.util

    builder_path = DEMO_ROOT / "build_sampled_background.py"
    spec = importlib.util.spec_from_file_location("sampled_background", builder_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    items = json.loads(holdouts.read_text())
    done = 0
    for item in items:
        src = Path(item["path"])
        root = work / "holdouts" / src.stem
        frames_dir = root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        if not any(frames_dir.glob("*.jpg")):
            subprocess.run(
                [FFMPEG, "-y", "-i", str(src), "-q:v", "2", "-start_number", "0", str(frames_dir / "%05d.jpg")],
                check=True,
            )
        frames = sorted(frames_dir.glob("*.jpg"))
        for start, end in chunk_spans(len(frames)):
            if max_chunks is not None and done >= max_chunks:
                print(f"FILL smoke stopped after {done}", flush=True)
                return
            chunk = root / f"chunk_{start:05d}"
            staged = chunk / "frames"
            staged.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(frames[start:end]):
                link = staged / f"{index:05d}.jpg"
                if not link.is_symlink():
                    if link.exists():
                        link.unlink()
                    link.symlink_to(frame)
            kept_dir = chunk / "fill" / "kept"
            if len(list(kept_dir.glob("*.jpg"))) >= end - start:
                print(f"FILL skip {src.stem} {start}:{end}", flush=True)
                done += 1
                continue
            print(f"FILL {src.stem} {start}:{end}", flush=True)
            masks = sam_union_allowing_empty(builder, staged, chunk / "sam")
            fill_dir = chunk / "fill"
            fill_dir.mkdir(parents=True, exist_ok=True)
            if not any(mask.any() for mask in masks):
                kept_dir.mkdir(parents=True, exist_ok=True)
                for index, frame in enumerate(frames[start:end]):
                    link = kept_dir / f"{index:05d}.jpg"
                    if not link.exists():
                        link.symlink_to(frame)
                print(f"FILL empty {src.stem} {start}:{end}", flush=True)
                done += 1
                continue
            kept = builder.fill_batch(staged, masks, fill_dir)
            if len(kept) < end - start:
                raise RuntimeError(f"{src.stem} chunk {start} kept {len(kept)} frames")
            done += 1
    print(f"FILL done {done}", flush=True)


def chunk_spans(n_frames: int, size: int = 30, minimum: int = 23) -> list[tuple[int, int]]:
    """Non-overlapping DiffuEraser chunks. A tail shorter than 23 frames is dropped."""
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n_frames:
        remain = n_frames - start
        if remain < minimum:
            break
        end = start + min(size, remain)
        spans.append((start, end))
        if end == n_frames:
            break
        start += size
    return spans


def _flatten(dataset: Path, dest: Path) -> None:
    description = json.loads((dataset / "description.json").read_text())
    expected = len(description["seqs"]) * len(description["frames"])
    dest.mkdir(parents=True, exist_ok=True)
    existing = list(dest.glob("*.jpg"))
    if len(existing) == expected and all(path.is_symlink() for path in existing):
        print(f"FLAT {dest} {expected}", flush=True)
        return
    index = 0
    for seq in description["seqs"]:
        folder = dataset / seq["path"]
        for name in description["frames"]:
            src = folder / name
            link = dest / f"{index:06d}.jpg"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
            index += 1
    print(f"FLAT {dest} {index}", flush=True)


def holdout_stems(factory: str) -> tuple[str, ...]:
    return tuple(f"{clip}_last10s" for clip, _, _ in SOURCES[factory])


def kept_frames(work: Path, stem: str) -> list[Path]:
    frames: list[Path] = []
    root = work / "holdouts" / stem
    for chunk in sorted(path for path in root.glob("chunk_*") if path.is_dir()):
        frames.extend(sorted((chunk / "fill" / "kept").glob("*.jpg")))
    return frames


def uf_test_config(root: Path, sequences: dict[str, dict]) -> dict:
    """PNG sequences. intra_period -1 is one I-frame for the segment."""
    return {
        "root_path": str(root),
        "test_classes": {
            "holdout": {
                "test": 1,
                "src_type": "png",
                "base_path": ".",
                "sequences": sequences,
            }
        },
    }


def score(work: Path, factory: str | None, codecs: tuple[str, ...], *, limit: int | None) -> None:
    """Score finished checkpoints and AV1 on the filled hold-outs."""
    stems = []
    if "av1" in codecs or factory is None:
        stems = sorted(path.name for path in (work / "holdouts").iterdir() if path.is_dir())
    if factory is not None:
        stems = list(holdout_stems(factory))
    done = 0
    for stem in stems:
        frames = kept_frames(work, stem)
        if not frames:
            raise FileNotFoundError(f"no filled frames for {stem}")
        structures = [codec for codec in codecs if codec in UF_STRUCTURES]
        if "av1" in codecs:
            done = _score_av1(work, stem, frames, done, limit)
            if limit is not None and done >= limit:
                print(f"SCORE smoke stopped after {done}", flush=True)
                return
        for structure in structures:
            ckpt = work / "checkpoints" / (factory or "") / structure / "s1" / "ckpt.pth.tar"
            if factory is None or not ckpt.is_file():
                print(f"SCORE skip {structure} {stem}", flush=True)
                continue
            done = _score_uf(work, factory, structure, ckpt, stem, frames, done, limit)
            if limit is not None and done >= limit:
                print(f"SCORE smoke stopped after {done}", flush=True)
                return
        if "hnerv" in codecs:
            if factory is None:
                print(f"SCORE skip hnerv {stem}", flush=True)
            else:
                done = _score_hnerv(work, factory, stem, frames, done, limit)
                if limit is not None and done >= limit:
                    print(f"SCORE smoke stopped after {done}", flush=True)
                    return
    print(f"SCORE done {done}", flush=True)


def _score_av1(work: Path, stem: str, frames: list[Path], done: int, limit: int | None) -> int:
    from PIL import Image

    for length in SEGMENT_FRAMES:
        for start in segment_starts(len(frames), length):
            if limit is not None and done >= limit:
                return done
            segment = frames[start:start + length]
            staged = work / "scores" / "staged" / stem / f"{start:05d}_{length}"
            staged.mkdir(parents=True, exist_ok=True)
            for index, src in enumerate(segment, start=1):
                dest = staged / f"im{index:05d}.png"
                if not dest.is_file():
                    Image.open(src).convert("RGB").save(dest)
            for name, scale in AV1_LADDER:
                out = work / "scores" / "av1" / stem / f"{start:05d}_{length}_{name}.mp4"
                record = out.with_suffix(".json")
                if record.is_file():
                    done += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    FFMPEG, "-y", "-framerate", str(FPS), "-i", str(staged / "im%05d.png"),
                    *av1_output_args(scale),
                    "-g", str(length),
                    "-svtav1-params", f"keyint={length}:keyint-min={length}",
                    str(out),
                ]
                print("RUN", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)
                payload = {
                    "codec": "av1", "stem": stem, "start": start, "frames": length,
                    "rung": name, "bytes": out.stat().st_size, "rgb_psnr_1080": _av1_rgb_psnr(staged, out),
                }
                record.write_text(json.dumps(payload) + "\n")
                print("SCORE", json.dumps(payload), flush=True)
                done += 1
    return done


def _av1_rgb_psnr(staged: Path, encoded: Path) -> float:
    import tempfile

    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        subprocess.run(
            [FFMPEG, "-y", "-i", str(encoded), "-vf", "scale=1920:1080:flags=bicubic", str(folder / "dec%05d.png")],
            check=True, capture_output=True,
        )
        sources = sorted(staged.glob("im*.png"))
        decoded = sorted(folder.glob("dec*.png"))
        if len(sources) != len(decoded) or not sources:
            raise RuntimeError(f"decoded {len(decoded)} frames from {encoded.name}, source {len(sources)}")
        total = 0.0
        for src, rec in zip(sources, decoded):
            a = np.asarray(Image.open(src).convert("RGB"), dtype=np.float64)
            b = np.asarray(Image.open(rec).convert("RGB"), dtype=np.float64)
            mse = float(np.mean((a - b) ** 2))
            total += 99.0 if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)
        return total / len(sources)


def _score_uf(
    work: Path, factory: str, structure: str, ckpt: Path, stem: str,
    frames: list[Path], done: int, limit: int | None,
) -> int:
    from PIL import Image

    for length in segments_for(structure):
        starts = segment_starts(len(frames), length)
        if limit is not None:
            starts = starts[: max(0, limit - done)]
        if not starts:
            continue
        stream = work / "scores" / "uf" / factory / structure / stem / str(length)
        record = stream / "out.json"
        if limit is None and record.is_file():
            print(f"SCORE uf {factory} {structure} {stem} length {length} exists", flush=True)
            done += len(starts)
            continue
        root = work / "scores" / "uf-src" / factory / structure / stem / str(length)
        sequences = {}
        for start in starts:
            name = f"s{start:05d}"
            folder = root / name
            folder.mkdir(parents=True, exist_ok=True)
            for index, src in enumerate(frames[start:start + length], start=1):
                dest = folder / f"im{index:05d}.png"
                if not dest.is_file():
                    Image.open(src).convert("RGB").save(dest)
            sequences[name] = {
                "height": 1080, "width": 1920,
                "intra_period": -1, "frames": length,
            }
        config_path = root / "test.json"
        config_path.write_text(json.dumps(uf_test_config(root, sequences)))
        stream.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(DCVC_PYTHON), "test_video.py",
            "--model_path_i", str(UF_IMAGE),
            "--model_path_p", str(ckpt),
            "--model_structure", structure,
            "--test_config", str(config_path),
            "--rate_num", str(len(SCORE_QPS)),
            "--qp_i", *[str(qp) for qp in SCORE_QPS],
            "--qp_p", *[str(qp) for qp in SCORE_QPS],
            "--reset_interval", "0",
            "--stream_path", str(stream),
            "--output_path", str(stream / "out.json"),
            "--worker", "1",
        ]
        if length == 1:
            cmd.extend(["--force_intra", "True"])
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(DCVC_ROOT), env=_dcvc_env(), check=True)
        done += len(starts)
        print(f"SCORE uf {factory} {structure} {stem} length {length}", flush=True)
    return done


def hnerv_checkpoint(work: Path, factory: str) -> Path:
    root = work / "checkpoints" / factory / "hnerv"
    preferred = sorted(path for path in root.glob("**/model_latest.pth") if "Dim64_0.5" in str(path))
    if not preferred:
        raise FileNotFoundError(f"no Dim64_0.5 HNeRV checkpoint under {root}")
    return preferred[-1]


def _score_hnerv(
    work: Path, factory: str, stem: str, frames: list[Path], done: int, limit: int | None,
) -> int:
    train_frames = len(list((work / "datasets" / f"{factory}-flat").iterdir()))
    ckpt = hnerv_checkpoint(work, factory)
    embed = hnerv_embed_dim(train_frames)
    for length in SEGMENT_FRAMES:
        for start in segment_starts(len(frames), length):
            if limit is not None and done >= limit:
                return done
            record = work / "scores" / "hnerv" / factory / stem / f"{start:05d}_{length}.json"
            if limit is None and record.is_file():
                done += 1
                continue
            staged = work / "scores" / "hnerv-src" / factory / stem / f"{start:05d}_{length}"
            staged.mkdir(parents=True, exist_ok=True)
            for path in staged.iterdir():
                path.unlink()
            for index, src in enumerate(frames[start:start + length]):
                (staged / f"{index:05d}{src.suffix}").symlink_to(src)
            outf = work / "scores" / "hnerv-run" / factory / stem / f"{start:05d}_{length}"
            modelsize = hnerv_modelsize_for_segment(train_frames, length)
            cmd = [
                str(HNERV_PYTHON), "train_nerv_all.py",
                "--data_path", str(staged),
                "--vid", f"{factory}-holdout",
                "--data_split", "1_1_1",
                "--crop_list", "-1",
                "--resize_list", "-1",
                "--enc_strds", "5", "3", "2", "2", "2",
                "--dec_strds", "5", "3", "2", "2", "2",
                "--enc_dim", f"64_{embed}",
                "--ks", "0_1_5",
                "--lower_width", "12",
                "--modelsize", f"{modelsize:.6f}",
                "--epochs", "30",
                "--batchSize", "1",
                "--loss", "Fusion6",
                "--quant_model_bit", "8",
                "--quant_embed_bit", "6",
                "--eval_only",
                "--not_resume",
                "--weight", str(ckpt),
                "--outf", str(outf),
                "--workers", "0",
            ]
            env = os.environ.copy()
            stub = hnerv_import_stub(work / "hnerv-stub")
            _ensure_target_package(HNERV_PYTHON, stub, "dahuffman", "dahuffman")
            _ensure_target_package(HNERV_PYTHON, stub, "pytorch_msssim", "pytorch-msssim")
            env["PYTHONPATH"] = os.pathsep.join((str(stub), str(HNERV_ROOT)))
            print("RUN", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(HNERV_ROOT), env=env, check=True)
            eval_txt = sorted(outf.rglob("eval.txt"))
            if len(eval_txt) != 1:
                raise RuntimeError(f"expected one HNeRV eval.txt under {outf}, found {len(eval_txt)}")
            text = eval_txt[0].read_text()
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps({
                "codec": "hnerv", "factory": factory, "stem": stem,
                "start": start, "frames": length, "embed_dim": embed,
                "modelsize": modelsize, "quant_embed_bit": 6,
                "eval": text,
            }) + "\n")
            print(f"SCORE hnerv {factory} {stem} {start} {length}", flush=True)
            done += 1
    return done


# Imported by tests that check the AV1 command stays on the shared CRF recipe.
_ = (AV1_CRF, AV1_PRESET, AV1_LADDER)


if __name__ == "__main__":
    raise SystemExit(main())
