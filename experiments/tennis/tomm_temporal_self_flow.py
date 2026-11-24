"""Measure self-flow warping error, source-flow error, context seams, and tLPIPS on Shanghai match102."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

from experiments.tennis.player_requalification import array_sha, sha256
from experiments.tennis.test_joint_native_motion import inputs as load_shanghai_arms
from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
from src.codecs import quality
from src.pointstream.tracked_patch import yuv_to_rgb

CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-crop-joint-20261122"
sys.path.insert(0, str(CARD))
from score_native import FPS, NT, StreamingYUV, get_support, prep  # noqa: E402

ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"


def source_mask(crop: tuple[tuple[int, int, int, int], np.ndarray]) -> np.ndarray:
    (x0, y0, x1, y1), m = crop
    mask = np.zeros((1080, 1920), np.uint8)
    mask[y0:y1, x0:x1] = m
    return cv2.resize(mask, (960, 544), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def is_seam_tick(tick: int) -> bool:
    return tick >= 20 and (tick % 20) in (0, 1, 2, 3, 4)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    expected = {"smoke": 10, "full": NT}[os.environ["PS_STAGE"]]
    if args.frames != expected:
        raise ValueError(f"frames={args.frames} does not match stage {os.environ['PS_STAGE']} ({expected})")

    grid = json.loads((CARD / "match102-source-grid.json").read_text())
    can = json.loads((CARD / "match102-canonical-source.json").read_text())
    if sha256(args.source) != grid["source_sha256"] or sha256(args.pixels) != can["lossless_sha256"]:
        raise ValueError("source/pixels hash mismatch")
    if sha256(args.alexnet) != ALEXNET_SHA256:
        raise ValueError("alexnet backbone hash mismatch")

    arms_path = load_shanghai_arms(args)
    arms = json.loads(arms_path.read_text())
    supports, support_hash = get_support(args.features)
    lpips_net = quality.load_lpips(args.alexnet, "cpu")

    # Evaluate tLPIPS on uniformly spaced transitions (step 5) plus 24-frame context boundaries (mod 20).
    tlpips_ticks = set(range(5, args.frames, 5)) | {t for t in range(20, args.frames, 20)}
    if args.frames <= 10:
        tlpips_ticks = set(range(1, args.frames))

    yy, xx = np.indices((544, 960), dtype=np.float32)
    cap = cv2.VideoCapture(str(args.pixels), cv2.CAP_FFMPEG)
    if abs(cap.get(cv2.CAP_PROP_FPS) - float(FPS)) >= 0.001:
        raise ValueError("unexpected source FPS")

    rows: dict[str, list[dict[str, Any]]] = {r["name"]: [] for r in arms}
    tlpips_rows: dict[str, list[dict[str, Any]]] = {r["name"]: [] for r in arms}
    source_hashes: list[str] = []
    prev_luma: dict[str, np.ndarray] = {}
    prev_rgb: dict[str, np.ndarray] = {}
    prev_gray: dict[str, np.ndarray] = {}
    prev_masks: list[np.ndarray] = []
    prev_boxes: list[tuple[int, int, int, int]] = []

    t0 = time.monotonic()
    with ExitStack() as scope:
        images: dict[str, Any] = {}
        for r in arms:
            work = Path(scope.enter_context(self_tempdir(scratch)))
            images[r["name"]] = prep(
                argparse.Namespace(source=args.source, source_pixels=args.pixels, frames=NT, fps=FPS),
                r,
                work,
            )[0]
            if isinstance(images[r["name"]], StreamingYUV):
                scope.callback(images[r["name"]].close)

        for t in range(args.frames):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"failed reading source frame {t}")
            rgb1080 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            source_hashes.append(array_sha(rgb1080))
            if source_hashes[-1] != grid["native_rgb_typed_sha256"][t]:
                raise ValueError(f"source frame {t} hash mismatch")

            src_960 = cv2.resize(rgb1080, (960, 544), interpolation=cv2.INTER_AREA)
            cur_luma: dict[str, np.ndarray] = {"source": luma(src_960)}
            cur_rgb: dict[str, np.ndarray] = {"source": src_960}
            cur_gray: dict[str, np.ndarray] = {"source": cv2.cvtColor(src_960, cv2.COLOR_RGB2GRAY)}
            masks = [source_mask(v) for v in supports[t]]
            boxes = [v[0] for v in supports[t]]

            for r in arms:
                name = r["name"]
                fr = images[name][t]
                if r["kind"] not in ("rgb_npy", "rgb_zseq"):
                    fr = cv2.resize(
                        yuv_to_rgb(fr, r["width"], r["height"], False),
                        (1920, 1080),
                        interpolation=cv2.INTER_CUBIC,
                    )
                im960 = cv2.resize(fr, (960, 544), interpolation=cv2.INTER_AREA)
                cur_rgb[name] = im960
                cur_luma[name] = luma(im960)
                cur_gray[name] = cv2.cvtColor(im960, cv2.COLOR_RGB2GRAY)

            if t > 0:
                src_flow = dis_flow(cur_gray["source"], prev_gray["source"])
                sx, sy = xx + src_flow[:, :, 0], yy + src_flow[:, :, 1]
                src_inside = (sx >= 0) & (sy >= 0) & (sx < 959) & (sy < 543)
                src_delta = {n: im - warp(prev_luma[n], src_flow) for n, im in cur_luma.items()}

                self_flows = {
                    r["name"]: dis_flow(cur_gray[r["name"]], prev_gray[r["name"]])
                    for r in arms
                }

                for role, idx in (("near", 0), ("far", 1)):
                    reg_src = src_inside & (masks[idx] > 0) & (warp(prev_masks[idx], src_flow) >= 0.5)
                    if not reg_src.any():
                        continue
                    src_d = src_delta["source"][reg_src]
                    # Union bounding box in 960x544 for tLPIPS on player region
                    bx0 = min(boxes[idx][0], prev_boxes[idx][0]) // 2
                    by0 = min(boxes[idx][1], prev_boxes[idx][1]) // 2
                    bx1 = min(960, max(boxes[idx][2], prev_boxes[idx][2]) // 2)
                    by1 = min(544, max(boxes[idx][3], prev_boxes[idx][3]) // 2)

                    for r in arms:
                        name = r["name"]
                        s_err = float(np.abs(src_delta[name][reg_src] - src_d).mean())
                        af = self_flows[name]
                        ax, ay = xx + af[:, :, 0], yy + af[:, :, 1]
                        a_inside = (ax >= 0) & (ay >= 0) & (ax < 959) & (ay < 543)
                        reg_self = a_inside & (masks[idx] > 0) & (warp(prev_masks[idx], af) >= 0.5)
                        self_diff = cur_luma[name] - warp(prev_luma[name], af)
                        self_err = float(np.abs(self_diff[reg_self]).mean()) if reg_self.any() else s_err
                        rows[name].append(
                            {
                                "tick": t,
                                "role": role,
                                "seam": is_seam_tick(t),
                                "source_flow_error": s_err,
                                "self_flow_error": self_err,
                                "source_delta": float(np.abs(src_d).mean()),
                                "decoded_delta": float(np.abs(src_delta[name][reg_src]).mean()),
                            }
                        )

                        if t in tlpips_ticks:
                            c0 = torch.from_numpy(np.ascontiguousarray(prev_rgb[name][by0:by1, bx0:bx1])).permute(2, 0, 1)[None].float() / 255.0
                            c1 = torch.from_numpy(np.ascontiguousarray(cur_rgb[name][by0:by1, bx0:bx1])).permute(2, 0, 1)[None].float() / 255.0
                            # Ensure minimum spatial size 32x32 for AlexNet
                            if c0.shape[-2] < 32 or c0.shape[-1] < 32:
                                c0 = torch.nn.functional.interpolate(c0, size=(max(32, c0.shape[-2]), max(32, c0.shape[-1])), mode="bilinear", align_corners=False)
                                c1 = torch.nn.functional.interpolate(c1, size=c0.shape[-2:], mode="bilinear", align_corners=False)
                            with torch.inference_mode():
                                lp = float(lpips_net(c0 * 2 - 1, c1 * 2 - 1).mean())
                            tlpips_rows[name].append({"tick": t, "role": role, "seam": (t % 20 == 0), "tlpips": lp})

            prev_luma, prev_rgb, prev_gray, prev_masks, prev_boxes = cur_luma, cur_rgb, cur_gray, masks, boxes

    cap.release()
    summary: dict[str, Any] = {}
    for r in arms:
        name = r["name"]
        rr = rows[name]
        seam = [x for x in rr if x["seam"]]
        interior = [x for x in rr if not x["seam"]]
        tr = tlpips_rows[name]
        t_seam = [x for x in tr if x["seam"]]
        t_int = [x for x in tr if not x["seam"]]
        summary[name] = {
            "n": len(rr),
            "source_flow_error": mean(x["source_flow_error"] for x in rr),
            "self_flow_error": mean(x["self_flow_error"] for x in rr),
            "source_delta": mean(x["source_delta"] for x in rr),
            "decoded_delta": mean(x["decoded_delta"] for x in rr),
            "seam_self_flow_error": mean(x["self_flow_error"] for x in seam) if seam else None,
            "interior_self_flow_error": mean(x["self_flow_error"] for x in interior) if interior else None,
            "tlpips_n": len(tr),
            "tlpips_mean": mean(x["tlpips"] for x in tr),
            "tlpips_seam_mean": mean(x["tlpips"] for x in t_seam) if t_seam else None,
            "tlpips_interior_mean": mean(x["tlpips"] for x in t_int) if t_int else None,
        }

    out = {
        "source_sha256": sha256(args.source),
        "source_mask_rle_sha256": support_hash,
        "recorded_ticks": args.frames,
        "elapsed_s": time.monotonic() - t0,
        "summary": summary,
        "rows": rows,
        "tlpips_rows": tlpips_rows,
    }
    (stage / "self_flow.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
    return out


def self_tempdir(root: Path) -> Any:
    import tempfile
    return tempfile.TemporaryDirectory(dir=root)


def validate(args: argparse.Namespace) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    expected = {"smoke": 10, "full": NT}[os.environ["PS_STAGE"]]
    data = json.loads((stage / "self_flow.json").read_text())
    assert data["recorded_ticks"] == expected
    assert len(data["summary"]) == 4
    for name, s in data["summary"].items():
        assert s["n"] == 2 * (expected - 1)
        assert np.isfinite(s["self_flow_error"]) and np.isfinite(s["tlpips_mean"])
        assert s["self_flow_error"] > 0 and s["tlpips_mean"] > 0
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": True, "recorded_ticks": expected, "summary": data["summary"]}, indent=2) + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "pixels", "features", "ordinary", "roi", "neural", "witness", "alexnet"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--frames", type=int, choices=(10, NT), required=True)
    p.add_argument("--validate", action="store_true")
    if sys.argv[1:] == ["--validate"]:
        stage = Path(os.environ["PS_STAGE_DIR"])
        argv = json.loads((stage / "dispatch.json").read_text())["command"]
        validate(p.parse_args(["--validate", *argv[3:]]))
    else:
        args = p.parse_args()
        if args.validate:
            validate(args)
        else:
            run_eval(args)


if __name__ == "__main__":
    main()
