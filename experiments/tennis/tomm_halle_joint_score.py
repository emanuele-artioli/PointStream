"""Measure 12 fixed visible-ball Halle instants across joint-crf50, AV1, and DCVC-UF on identical source supports."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from fractions import Fraction
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys
import tempfile
import time
from typing import Any

import cv2
import numpy as np
import torch
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure

from experiments.tennis.player_requalification import sha256
from experiments.tennis.test_extension_native_score import select_arm
from src.codecs import quality
from src.pointstream.tracked_patch import yuv_to_rgb

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs/experiments/test-joint-extension-20261122/native"
sys.path.insert(0, str(ROOT / "docs/experiments/test-crop-joint-20261122"))
from score_native import StreamingYUV, get_support, prep, tensor  # noqa: E402

VGG_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
BALL_CSV_SHA256 = "26e07e6f6d825bf42cb439da571c76ea12bfcbf217ca26c00443a20566a168eb"
NT = 240


def image_psnr(source: np.ndarray, decoded: np.ndarray, mask: np.ndarray | None = None) -> float:
    delta = source.astype(np.float32) - decoded.astype(np.float32)
    mse = float(np.mean(np.square(delta if mask is None else delta[mask]), dtype=np.float64))
    return 10.0 * math.log10(255.0**2 / mse) if mse > 0 else math.inf


def ssim_maps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def gaussian(value: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(value, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)

    mx, my = gaussian(x), gaussian(y)
    xx = np.maximum(gaussian(x**2) - mx**2, 0.0)
    yy = np.maximum(gaussian(y**2) - my**2, 0.0)
    xy = gaussian(x * y) - mx * my
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    contrast = (2.0 * xy + c2) / (xx + yy + c2)
    return ((2.0 * mx * my + c1) / (mx**2 + my**2 + c1)) * contrast, contrast


def luma_ssim(source: np.ndarray, decoded: np.ndarray, mask: np.ndarray | None = None) -> float:
    x, y = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) for rgb in (source, decoded))
    similarity, _ = ssim_maps(x, y)
    return float(np.mean(similarity if mask is None else similarity[mask], dtype=np.float64))


def rgb_ms_ssim(source: np.ndarray, decoded: np.ndarray) -> float:
    def downsample(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return cv2.resize(
            image[: h - h % 2, : w - w % 2], (w // 2, h // 2), interpolation=cv2.INTER_AREA
        )

    x, y = source.astype(np.float64), decoded.astype(np.float64)
    weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    scores = []
    for index, weight in enumerate(weights):
        similarity, contrast = ssim_maps(x, y)
        measure = similarity if index == len(weights) - 1 else contrast[5:-5, 5:-5]
        scores.append(max(0.0, float(np.mean(measure, dtype=np.float64))) ** weight)
        x, y = downsample(x), downsample(y)
    return math.prod(scores)


@torch.inference_mode()
def region_scores(
    original: np.ndarray,
    display: np.ndarray,
    mask: np.ndarray,
    dists: Any,
    lpips: Any,
) -> dict[str, float]:
    src_t, dec_t = tensor(original), tensor(display)
    support = torch.from_numpy(np.ascontiguousarray(mask))[None, None]
    lp = lpips(dec_t * 2.0 - 1.0, src_t * 2.0 - 1.0)
    lp = torch.nn.functional.interpolate(lp, size=mask.shape, mode="bilinear", align_corners=False)
    return {
        "dists": float(quality.masked_dists(dists, dec_t, src_t, support).item()),
        "lpips": float(lp[support].mean().item()),
        "luma_ssim": luma_ssim(original, display, mask),
        "rgb_psnr": image_psnr(original, display, mask),
    }


def whole_scores(original: np.ndarray, display: np.ndarray) -> dict[str, float]:
    return {
        "luma_ssim": luma_ssim(original, display),
        "rgb_psnr": image_psnr(original, display),
        "rgb_ms_ssim": rgb_ms_ssim(original, display),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {k: mean(r[k] for r in rows) for k in rows[0]}


def select_official_ticks(ball_csv: Path, count: int) -> tuple[list[int], dict[int, tuple[int, int]]]:
    assert sha256(ball_csv) == BALL_CSV_SHA256
    coords: dict[int, tuple[int, int]] = {}
    with ball_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tick = int(row["Frame"])
            if 0 <= tick < NT and int(row["Visibility"]) == 1:
                x, y = int(row["X"]), int(row["Y"])
                if 32 <= x <= 1920 - 32 and 32 <= y <= 1080 - 32:
                    coords[tick] = (x, y)
    visible = sorted(coords)
    assert len(visible) >= 12
    if count == 1:
        selected = [visible[0]]
    else:
        assert count == 12
        selected = [visible[round(i * (len(visible) - 1) / 11)] for i in range(12)]
        assert len(set(selected)) == 12
    return selected, {t: coords[t] for t in selected}


def run(args: argparse.Namespace) -> None:
    started = time.process_time()
    assert len(os.sched_getaffinity(0)) == 1 and (
        not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)
    assert cv2.getNumThreads() == 1
    expected_ticks = {"smoke": 1, "full": 12}[os.environ["PS_STAGE"]]
    assert args.ticks == expected_ticks
    assert sha256(args.vgg) == VGG_SHA256 and sha256(args.alexnet) == ALEXNET_SHA256

    binding = json.loads((CARD / "match105-binding.json").read_text())
    receipts = CARD / "match105-crop-receipt.json"
    bank = CARD.parents[1] / "cohort-expansion-20261122/test-physical"
    fps = Fraction(binding["fps"])
    assert binding["source_sha256"] == sha256(args.source)
    assert binding["source_features_sha256"] == sha256(args.features)

    supports, support_rle_sha = get_support(
        args.features, receipt=receipts, tracks=[binding["roles"]["near"], binding["roles"]["far"]]
    )
    ticks, ball_coords = select_official_ticks(args.ball, args.ticks)

    arms = [
        select_arm(
            argparse.Namespace(
                event="match105",
                arm="joint-crf50",
                data=args.witness,
                report_sha256=None,
                sender=args.sender,
                receipt=None,
            ),
            binding,
            bank,
        ),
        select_arm(
            argparse.Namespace(
                event="match105",
                arm="av1-320-250",
                data=args.ordinary,
                report_sha256=None,
                sender=None,
                receipt=None,
            ),
            binding,
            bank,
        ),
        select_arm(
            argparse.Namespace(
                event="match105",
                arm="av1-640-280",
                data=args.ordinary,
                report_sha256=None,
                sender=None,
                receipt=None,
            ),
            binding,
            bank,
        ),
        select_arm(
            argparse.Namespace(
                event="match105",
                arm="dcvc-480-q24",
                data=args.neural,
                report_sha256=None,
                sender=None,
                receipt=None,
            ),
            binding,
            bank,
        ),
    ]

    dists_model = quality.load_dists(args.vgg, "cpu")
    lpips_model = quality.load_lpips(args.alexnet, "cpu")
    setup_seconds = time.process_time() - started

    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    stage = Path(os.environ["PS_STAGE_DIR"])
    tick_set = set(ticks)
    results: dict[str, list[dict[str, Any]]] = {a["name"]: [] for a in arms}
    self_control: dict[str, Any] | None = None
    validation_seconds = 0.0
    forecast = 0.0

    with ExitStack() as scope:
        images: dict[str, Any] = {}
        for arm in arms:
            work = Path(scope.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
            images[arm["name"]], _ = prep(
                argparse.Namespace(source=args.source, source_pixels=args.source, frames=NT, fps=fps),
                arm,
                work,
            )
            if isinstance(images[arm["name"]], StreamingYUV):
                scope.callback(images[arm["name"]].close)

        cap = cv2.VideoCapture(str(args.source), cv2.CAP_FFMPEG)
        assert abs(cap.get(cv2.CAP_PROP_FPS) - float(fps)) < 0.001
        evaluated = 0
        for tick in range(NT):
            ok, bgr_src = cap.read()
            if not ok:
                raise RuntimeError(f"failed source frame {tick}")
            arm_raw: dict[str, Any] = {}
            for arm in arms:
                if not isinstance(images[arm["name"]], StreamingYUV) and tick not in tick_set:
                    continue
                arm_raw[arm["name"]] = images[arm["name"]][tick]
            if tick not in tick_set:
                continue

            original = cv2.cvtColor(bgr_src, cv2.COLOR_BGR2RGB)
            bx, by = ball_coords[tick]
            ball_rect = (bx - 32, by - 32, bx + 32, by + 32)
            ball_mask = np.ones((64, 64), dtype=bool)
            tick_supports = [
                ("near", supports[tick][0][0], supports[tick][0][1]),
                ("far", supports[tick][1][0], supports[tick][1][1]),
                ("ball", ball_rect, ball_mask),
            ]
            if evaluated == 0:
                target_ball = original[by - 32 : by + 32, bx - 32 : bx + 32]
                self_control = region_scores(target_ball, target_ball, ball_mask, dists_model, lpips_model)
                assert abs(self_control["dists"]) < 1e-5 and abs(self_control["lpips"]) < 1e-9
                assert abs(self_control["luma_ssim"] - 1.0) < 1e-7 and math.isinf(
                    self_control["rgb_psnr"]
                )
                self_control["rgb_ms_ssim"] = whole_scores(original, original)["rgb_ms_ssim"]
                assert abs(self_control["rgb_ms_ssim"] - 1.0) < 1e-5
                self_control["rgb_psnr"] = "infinite"

            for arm in arms:
                tag = arm["name"]
                fr = arm_raw[tag]
                if arm["kind"] in ("rgb_npy", "rgb_zseq"):
                    display = fr
                else:
                    display = cv2.resize(
                        yuv_to_rgb(fr, arm["width"], arm["height"], False),
                        (1920, 1080),
                        interpolation=cv2.INTER_CUBIC,
                    )
                assert display.shape == original.shape == (1080, 1920, 3)
                values: dict[str, dict[str, float]] = {}
                fg_pixels = np.zeros(original.shape[:2], dtype=bool)
                for name, (x0, y0, x1, y1), mask in tick_supports:
                    fg_pixels[y0:y1, x0:x1] |= mask
                    pair = region_scores(
                        original[y0:y1, x0:x1], display[y0:y1, x0:x1], mask, dists_model, lpips_model
                    )
                    assert all(math.isfinite(v) for v in pair.values())
                    values[name] = pair
                players = aggregate([values["near"], values["far"]])
                whole = whole_scores(original, display)
                assert all(math.isfinite(v) for v in whole.values())
                if evaluated == 0 and tag == "joint-crf50":
                    before = time.process_time()
                    with torch.inference_mode():
                        reference = float(
                            multiscale_structural_similarity_index_measure(
                                tensor(display), tensor(original), data_range=1
                            ).item()
                        )
                    diff = abs(reference - whole["rgb_ms_ssim"])
                    assert diff < 0.00015, (reference, whole["rgb_ms_ssim"])
                    assert self_control is not None
                    self_control["ms_ssim_implementation_difference"] = diff
                    validation_seconds = time.process_time() - before
                results[tag].append(
                    {
                        "tick": tick,
                        "values": values,
                        "foreground": aggregate([players, values["ball"]]),
                        "player": players,
                        "ball": values["ball"],
                        "whole": whole,
                        "background_psnr": image_psnr(original, display, ~fg_pixels),
                    }
                )
            consumed = time.process_time() - started
            if evaluated == 0:
                forecast = (
                    setup_seconds
                    + validation_seconds
                    + 1.5 * (consumed - setup_seconds - validation_seconds) * 12
                    + 30
                )
                assert forecast < args.full_cap, f"CPU forecast {forecast:.1f}s exceeds {args.full_cap}s"
            assert consumed < args.full_cap
            evaluated += 1
        cap.release()

    domains = ("player", "ball", "foreground", "whole")
    summary_arms = {}
    for arm in arms:
        tag = arm["name"]
        rows = results[tag]
        summary_arms[tag] = {
            "bytes": arm["bytes"],
            "kbps": arm["bytes"] * float(fps) / 30000.0,
            "means": {d: aggregate([r[d] for r in rows]) for d in domains},
            "rows": rows,
        }
    record = {
        "event": "match105",
        "source_sha256": binding["source_sha256"],
        "official_ball_sha256": BALL_CSV_SHA256,
        "support_rle_sha256": support_rle_sha,
        "source_ticks": NT,
        "sampled_ticks": len(ticks),
        "ticks": ticks,
        "source_fps": [fps.numerator, fps.denominator],
        "dists_backbone_sha256": VGG_SHA256,
        "alexnet_backbone_sha256": ALEXNET_SHA256,
        "self_control": self_control,
        "forecast_seconds": forecast,
        "cpu_seconds": time.process_time() - started,
        "arms": summary_arms,
    }
    (stage / "halle_joint_scores.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.ticks == 12:
        validate(args, stage, stage / "full_validation.json")


def validate(args: argparse.Namespace, stage: Path, dest: Path) -> None:
    expected = {"smoke": 1, "full": 12}[os.environ["PS_STAGE"]]
    data = json.loads((stage / "halle_joint_scores.json").read_text())
    assert data["sampled_ticks"] == expected and len(data["ticks"]) == expected
    assert abs(data["self_control"]["dists"]) < 1e-5
    assert data["forecast_seconds"] < args.full_cap
    compact_arms = {
        tag: {"bytes": arm["bytes"], "kbps": arm["kbps"], "means": arm["means"]}
        for tag, arm in data["arms"].items()
    }
    dest.write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "sampled_ticks": expected,
                    "ticks": data["ticks"],
                    "forecast_seconds": data["forecast_seconds"],
                },
                "arms": compact_arms,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "features", "ball", "ordinary", "neural", "witness", "sender", "vgg", "alexnet"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    p.add_argument("--ticks", type=int, choices=(1, 12), required=True)
    p.add_argument("--full-cap", type=int, default=900)
    p.add_argument("--validate", action="store_true")
    if sys.argv[1:] == ["--validate"]:
        stage = Path(os.environ["PS_STAGE_DIR"])
        argv = json.loads((stage / "dispatch.json").read_text())["command"]
        args = p.parse_args(argv[3:])
        args.validate = True
        validate(args, stage, Path(os.environ["PS_VALIDATION_PATH"]))
    else:
        run(p.parse_args())


if __name__ == "__main__":
    main()
