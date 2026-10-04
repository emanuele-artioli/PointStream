"""Render a compact visual comparison from the saved background hold-outs."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from demo.experiments.factory_bg_rd import (
    DCVC_PYTHON,
    DCVC_ROOT,
    FFMPEG,
    HNERV_PYTHON,
    HNERV_ROOT,
    UF_IMAGE,
    UF_VIDEO,
    WORK_ROOT,
    _dcvc_env,
    _ensure_target_package,
    hnerv_checkpoint,
    hnerv_import_stub,
    kept_frames,
    uf_test_config,
)


START = 120
LENGTH = 8
SAMPLE_INDICES = (0, 2, 4, 6)
QP = 21
FPS = 30
METHODS = (
    ("filled", "Filled source"),
    ("av1_240p", "AV1 · 240p"),
    ("av1_1080p", "AV1 · 1080p"),
    ("hnerv", "HNeRV"),
    ("uf_ld", "DCVC-UF · LD"),
    ("uf_hts", "DCVC-UF · HT-S"),
    ("uf_htl", "DCVC-UF · HT-L"),
)
CLIPS = (
    ("factory001", "clip_03_factory001_worker001_00000_last10s"),
    ("factory002", "factory002_worker001_00000_last10s"),
)


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _psnr(reference: Path, reconstruction: Path) -> float:
    import numpy as np

    a = np.asarray(Image.open(reference).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(reconstruction).convert("RGB"), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10.0 * math.log10((255.0**2) / mse)


def _average_psnr(reference: list[Path], reconstruction: list[Path]) -> float:
    if len(reference) != LENGTH or len(reconstruction) != LENGTH:
        raise RuntimeError("visual comparison requires the same eight frames for both arms")
    return sum(_psnr(a, b) for a, b in zip(reference, reconstruction)) / LENGTH


def _decode_av1(source: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", "scale=1920:1080:flags=bicubic", "-fps_mode", "passthrough",
            str(output / "frame_%05d.png"),
        ],
        check=True,
    )
    frames = sorted(output.glob("frame_*.png"), key=_natural_key)
    if len(frames) != LENGTH:
        raise RuntimeError(f"decoded {len(frames)} frames from {source}, expected {LENGTH}")
    return frames


def _render_dcvc(
    root: Path,
    factory: str,
    stem: str,
    frames: list[Path],
    structure: str,
) -> tuple[list[Path], Path]:
    input_root = root / "inputs" / factory / stem
    sequence = f"s{START:05d}"
    sequence_dir = input_root / sequence
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(frames, start=1):
        Image.open(source).convert("RGB").save(sequence_dir / f"im{index:05d}.png")

    config_path = input_root / "test.json"
    config_path.write_text(
        json.dumps(
            uf_test_config(
                input_root,
                {
                    sequence: {
                        "height": 1080,
                        "width": 1920,
                        "intra_period": -1,
                        "frames": LENGTH,
                    }
                },
            )
        )
    )
    checkpoint = WORK_ROOT / "checkpoints" / factory / structure / "s1" / "ckpt.pth.tar"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stream = root / "streams" / factory / structure / stem
    stream.mkdir(parents=True, exist_ok=True)
    output_json = stream / "out.json"
    command = [
        str(DCVC_PYTHON), "test_video.py",
        "--model_path_i", str(UF_IMAGE),
        "--model_path_p", str(checkpoint),
        "--model_structure", structure,
        "--test_config", str(config_path),
        "--rate_num", "1",
        "--qp_i", str(QP),
        "--qp_p", str(QP),
        "--reset_interval", "0",
        "--stream_path", str(stream),
        "--output_path", str(output_json),
        "--worker", "1",
        "--save_decoded_frame", "True",
    ]
    env = _dcvc_env()
    log_path = stream / "command.log"
    with log_path.open("w") as log:
        subprocess.run(command, cwd=str(DCVC_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    bitstreams = sorted(stream.rglob("*.bin"), key=_natural_key)
    if len(bitstreams) != 1:
        raise RuntimeError(f"expected one DCVC bitstream, found {len(bitstreams)} under {stream}")
    decoded = sorted(stream.rglob("*.png"), key=_natural_key)
    decoded = [path for path in decoded if path.parent != sequence_dir]
    if len(decoded) != LENGTH:
        raise RuntimeError(f"decoded {len(decoded)} DCVC frames for {factory}/{structure}/{stem}")
    return decoded, bitstreams[0]


def _render_hnerv(root: Path, factory: str, frames: list[Path]) -> tuple[list[Path], int]:
    import numpy as np
    import torch
    from torchvision.io import read_image

    from demo.experiments.hnerv_holdout_eval import _architecture, _load

    stub = hnerv_import_stub(root / "hnerv-stub")
    _ensure_target_package(HNERV_PYTHON, stub, "dahuffman", "dahuffman")
    _ensure_target_package(HNERV_PYTHON, stub, "pytorch_msssim", "pytorch-msssim")
    sys.path[:0] = [str(stub), str(HNERV_ROOT)]
    from hnerv_utils import quant_tensor
    from model_all import TransformInput
    from train_nerv_all import quant_model as quantize_model

    torch.set_num_threads(4)
    train_frames = len(list((WORK_ROOT / "datasets" / f"{factory}-flat").iterdir()))
    checkpoint = hnerv_checkpoint(WORK_ROOT, factory)
    model, args, _embed_dim = _architecture(train_frames)
    _load(model, checkpoint)
    model.eval()
    quantized, _quant_ckt = quantize_model(model, args)
    quant_net = quantized[-1].cuda().eval()
    transform = TransformInput(args)
    embeddings = []
    ground_truths = []
    with torch.inference_mode():
        for source in frames:
            image = read_image(str(source)).float().cuda() / 255.0
            model_input, gt, _mask = transform(image.unsqueeze(0))
            _prediction, embed_list, _decoder_time = model(model_input)
            embeddings.append(embed_list[0].detach())
            ground_truths.append(gt)
    quantized_embed, dequantized = quant_tensor(torch.cat(embeddings, 0), args.quant_embed_bit)
    del quantized_embed
    output_dir = root / "hnerv" / factory
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    with torch.inference_mode():
        for index, gt in enumerate(ground_truths):
            decoded, _embed_list, _decoder_time = quant_net(gt, dequantized[index:index + 1])
            image = decoded.detach().float().squeeze(0).permute(1, 2, 0).cpu().numpy()
            if float(np.nanmin(image)) < -0.05 or float(np.nanmax(image)) > 1.05:
                image = image * 0.5 + 0.5
            image = np.clip(image, 0.0, 1.0)
            path = output_dir / f"frame_{index:05d}.png"
            Image.fromarray(np.round(image * 255.0).astype(np.uint8), mode="RGB").save(path)
            outputs.append(path)
    score_path = WORK_ROOT / "scores" / "hnerv" / factory / CLIPS[0 if factory == "factory001" else 1][1] / f"{START:05d}_{LENGTH}.json"
    scored_bits = None
    if score_path.is_file():
        saved = json.loads(score_path.read_text())
        scored_bits = saved.get("total_bits")
    if not isinstance(scored_bits, int):
        raise RuntimeError(f"no saved HNeRV rate record at {score_path}")
    del model, quant_net, quantized, ground_truths, embeddings, dequantized
    torch.cuda.empty_cache()
    return outputs, scored_bits


def _font(size: int, bold: bool = False):
    candidates = (
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf")
        if bold else ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf")
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _contact_sheet(
    path: Path,
    clip_label: str,
    sample_index: int,
    images: dict[str, list[Path]],
    metrics: dict[str, dict[str, float | str | None]],
) -> None:
    tile_w, image_h, header_h = 384, 216, 36
    gap = 12
    title_h = 66
    cols, rows = 4, 2
    cell_h = image_h + header_h
    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * gap, title_h + rows * cell_h + (rows + 1) * gap), "#101923")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 13), f"{clip_label} · source frame offset {sample_index} of an 8-frame segment", font=_font(22, True), fill="#edf4fa")
    draw.text((gap, 41), "Same filled input across methods · all decoded previews scaled to 1080p", font=_font(14), fill="#aebdcc")

    for position, (key, label) in enumerate(METHODS):
        col, row = position % cols, position // cols
        x, y = gap + col * (tile_w + gap), title_h + gap + row * (cell_h + gap)
        draw.rounded_rectangle((x, y, x + tile_w, y + cell_h), radius=8, fill="#182431", outline="#314254", width=1)
        draw.rectangle((x + 1, y + 1, x + tile_w - 1, y + header_h), fill="#223346")
        metric = metrics[key]
        detail = metric.get("detail") or "filled input"
        draw.text((x + 9, y + 7), label, font=_font(15, True), fill="#f1f5fa")
        draw.text((x + tile_w - 9, y + 8), str(detail), font=_font(12), fill="#b5c7d8", anchor="ra")
        image = Image.open(images[key][sample_index]).convert("RGB")
        image = image.resize((tile_w, image_h), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y + header_h))

    notes_col, notes_row = 3, 1
    x, y = gap + notes_col * (tile_w + gap), title_h + gap + notes_row * (cell_h + gap)
    draw.rounded_rectangle((x, y, x + tile_w, y + cell_h), radius=8, fill="#1d2a37", outline="#3b5164", width=1)
    lines = (
        "Reading the comparison",
        "• Filled source is the inpainted codec input.",
        "• PSNR is mean RGB over all 8 decoded frames versus that input.",
        "• Rates use each saved stream; HNeRV includes model + embedding bits.",
        "• DCVC-UF uses QP 21 for all three structures; rates are not matched.",
        "• AV1 240p and 1080p are both shown decoded at 1080p.",
        "Qualitative sample only; this is not a paper result.",
    )
    for line_no, line in enumerate(lines):
        draw.text((x + 12, y + 12 + line_no * 25), line, font=_font(14, line_no == 0), fill="#e8eef5" if line_no == 0 else "#c1ceda")

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "JPEG", quality=84, optimize=True, progressive=True)


def render_clip(root: Path, factory: str, stem: str, clip_label: str) -> dict:
    kept = kept_frames(WORK_ROOT, stem)
    selected = kept[START:START + LENGTH]
    if len(selected) != LENGTH or not all(path.is_file() for path in selected):
        raise RuntimeError(f"missing filled source frames {START}:{START + LENGTH} for {stem}")

    images: dict[str, list[Path]] = {"filled": selected}
    rates_kbps: dict[str, float | None] = {"filled": None}
    clip_output = root / "previews" / factory

    for rung in ("240p", "1080p"):
        video = WORK_ROOT / "scores" / "av1" / stem / f"{START:05d}_{LENGTH}_{rung}.mp4"
        images[f"av1_{rung}"] = _decode_av1(video, clip_output / f"av1_{rung}")
        rates_kbps[f"av1_{rung}"] = video.stat().st_size * 8 * FPS / LENGTH / 1000.0

    for structure in ("ld", "hts", "htl"):
        key = f"uf_{structure}"
        decoded, bitstream = _render_dcvc(root, factory, stem, selected, structure)
        images[key] = decoded
        rates_kbps[key] = bitstream.stat().st_size * 8 * FPS / LENGTH / 1000.0

    hnerv_frames, hnerv_bits = _render_hnerv(root, factory, selected)
    images["hnerv"] = hnerv_frames
    rates_kbps["hnerv"] = hnerv_bits * FPS / LENGTH / 1000.0

    metrics: dict[str, dict[str, float | str | None]] = {}
    for key, _label in METHODS:
        psnr = None if key == "filled" else _average_psnr(selected, images[key])
        rate = rates_kbps[key]
        if key == "filled":
            detail = "reference"
        elif key.startswith("uf_"):
            detail = f"QP {QP} · {rate:.0f} kbps · {psnr:.1f} dB"
        elif key == "hnerv":
            detail = f"{rate:.0f} kbps · {psnr:.1f} dB"
        else:
            detail = f"{rate:.0f} kbps · {psnr:.1f} dB"
        metrics[key] = {"rate_kbps": rate, "rgb_psnr_vs_filled_db": psnr, "detail": detail}

    panel_dir = root / "canvas"
    for sample_index in SAMPLE_INDICES:
        _contact_sheet(
            panel_dir / f"{factory}_frame_{sample_index:02d}.jpg",
            clip_label,
            sample_index,
            images,
            metrics,
        )
    return {
        "factory": factory,
        "stem": stem,
        "label": clip_label,
        "segment_start_in_holdout": START,
        "segment_length": LENGTH,
        "sample_indices": list(SAMPLE_INDICES),
        "methods": metrics,
        "panels": {str(index): f"canvas/{factory}_frame_{index:02d}.jpg" for index in SAMPLE_INDICES},
    }


def main() -> int:
    root = Path(os.environ["PS_JOB_DIR"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    clips = [
        render_clip(root, "factory001", CLIPS[0][1], "Factory 001 · clip 3 press"),
        render_clip(root, "factory002", CLIPS[1][1], "Factory 002"),
    ]
    manifest = {
        "scope": "bounded qualitative preview; existing filled hold-out inputs and saved AV1/HNeRV/UF checkpoints",
        "fps": FPS,
        "qp": QP,
        "clips": clips,
    }
    (root / "canvas" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"canvas_dir": str(root / "canvas"), "clips": [clip["factory"] for clip in clips]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
