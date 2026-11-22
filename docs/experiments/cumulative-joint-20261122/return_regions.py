"""Predeclared /002 rectangular regional fidelity on actual receiver frames."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import tarfile
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from experiments.tennis import (
    common_player_packet as p,
    common_player_ball as b,
    player_requalification as replay,
)
from src.codecs import quality, svtav1
from src.pointstream import tracked_patch

TICKS = list(range(10, 231, 20))
ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
SOURCE = ROOT / "tennis/w7-tier2-codex-20261031/eval/sources/nadal_kyrgios__002.mp4"
BIND = ROOT / "tennis/cumulative-joint-20261122/return-cached-player-v5"
VGG = ROOT / "visor/h3b-2026-10-11/inputs/dists/vgg16-397923af.pth"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finish", type=Path, required=True)
    ap.add_argument("--finish-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert sha(args.finish) == args.finish_sha256
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    started = time.monotonic()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert (
        sha(SOURCE) == "f0551443eaff4e4b3539f21cd0c6ac66f55601d480ad099a571e8626db735e4a"
        and sha(VGG) == "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
    )
    carrier = (BIND / "receiver.pscp").read_bytes()
    side = (BIND / "ball.sidecar").read_bytes()
    assert (
        hashlib.sha256(carrier).hexdigest()
        == "a0ec63326642130adb5d765a3e7c30aaf564ad5bbf7cd17d56594ffd82dc5b06"
    )
    _, _, players = p.unpack(carrier)
    names = p._embedded_profile(carrier)["order"]
    dec = svtav1.tool("dav1d")
    pos, tile, _ = b._open(carrier, side, dec["sha256"])
    tracked_patch.decode_native(
        tile,
        args.output / "tile.yuv",
        {"width": 64, "height": 64, "fps": [50, 1], "frames": 240},
        dec["real_path"],
    )
    patch = np.memmap(args.output / "tile.yuv", np.uint8, "r", shape=(240, 96, 64))
    with tarfile.open(args.finish) as tar:
        report = json.load(tar.extractfile("publish/finish.json"))
        f = tar.extractfile("publish/frames.npy")
        with (args.output / "frames.npy").open("wb") as dst:
            while x := f.read(1048576):
                dst.write(x)
    frames = np.load(args.output / "frames.npy", mmap_mode="r")
    assert (
        frames.shape == (240, 1080, 1920, 3)
        and report["carrier_sha256"] == hashlib.sha256(carrier).hexdigest()
    )
    with tarfile.open(
        ROOT / "jobs/fleet/inbox/20261121T234041Z-ad3f14db/full/published.tar"
    ) as tar:
        control = tar.extractfile("publish/960-q63.5.ivf").read()
    assert (
        hashlib.sha256(control).hexdigest()
        == "66d3af87818b76c12435a9b2b4f5e9797a219a3d32faf4aface4568af74b8bf8"
    )
    tracked_patch.decode_native(
        control,
        args.output / "control.yuv",
        {"width": 960, "height": 544, "fps": [50, 1], "frames": 240},
        dec["real_path"],
    )
    ctrl = np.memmap(args.output / "control.yuv", np.uint8, "r", shape=(240, 816, 960))
    source_y = args.output / "source-yuv.yuv"
    subprocess.run(
        [
            svtav1.tool("ffmpeg")["path"],
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(SOURCE),
            "-frames:v",
            "240",
            "-vf",
            "scale=480:272:flags=area:out_range=tv,format=yuv420p",
            "-filter_threads",
            "1",
            "-fps_mode",
            "passthrough",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            str(source_y),
        ],
        check=True,
        timeout=120,
    )
    truth = np.memmap(source_y, np.uint8, "r", shape=(240, 408, 480))
    luma_sse = {"ordinary": 0, "pointstream": 0}
    net = quality.load_dists(VGG, "cpu")
    rows = []
    preview_rows = []
    cap = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG)
    assert cap.isOpened()
    with torch.inference_mode():
        for tick in range(240):
            ok, src = cap.read()
            assert ok
            joint = np.array(frames[tick], copy=True)
            assert replay.array_sha(joint) == report["rows"][tick]["frame_sha256"]
            if pos[tick] is not None:
                x, y = pos[tick]
                joint[y : y + 32, x : x + 32] = tracked_patch.yuv_to_rgb(
                    patch[tick], 64, 64, False
                )[16:48, 16:48]
            ordinary = cv2.resize(
                tracked_patch.yuv_to_rgb(ctrl[tick], 960, 544, False),
                (1920, 1080),
                interpolation=cv2.INTER_CUBIC,
            )
            for arm, image in (("ordinary", ordinary), ("pointstream", joint)):
                small = cv2.resize(image, (480, 272), interpolation=cv2.INTER_AREA).astype(
                    np.float64
                )
                luma = np.clip(
                    np.rint(
                        16
                        + 219
                        * (
                            0.2126 * small[:, :, 0]
                            + 0.7152 * small[:, :, 1]
                            + 0.0722 * small[:, :, 2]
                        )
                        / 255
                    ),
                    0,
                    255,
                ).astype(np.int32)
                delta = truth[tick, :272].astype(np.int32) - luma
                luma_sse[arm] += int(np.square(delta).sum(dtype=np.int64))
            if tick not in TICKS:
                continue
            source = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            if tick in (30, 110, 190):
                panels = []
                for label, image in zip(
                    ("Source", "AV1", "PointStream"), (source, ordinary, joint), strict=True
                ):
                    panel = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
                    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        panel,
                        f"{label}  frame {tick}",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    panels.append(panel)
                preview_rows.append(np.hstack(panels))
            for name in names:
                box = players[name][0]["boxes"][tick]
                x, y, w, h = [round(float(v)) for v in box]
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(1920, x + max(1, w))
                y1 = min(1080, y + max(1, h))
                assert x1 > x0 and y1 > y0
                interpolation = (
                    cv2.INTER_AREA if x1 - x0 > 256 or y1 - y0 > 256 else cv2.INTER_LINEAR
                )
                imgs = [
                    cv2.resize(im[y0:y1, x0:x1], (256, 256), interpolation=interpolation)
                    for im in (source, ordinary, joint)
                ]

                def tensor(im):
                    return torch.from_numpy(im.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255

                s = tensor(imgs[0])
                mask = torch.ones((1, 1, 256, 256))
                selfd = float(quality.masked_dists(net, s, s, mask)[0])
                assert abs(selfd) < 1e-5
                vals = []
                for target in imgs[1:]:
                    d = float(quality.masked_dists(net, s, tensor(target), mask)[0])
                    mse = float(np.mean((imgs[0].astype(np.float64) - target) ** 2))
                    assert math.isfinite(d) and mse > 0
                    vals.append({"dists": d, "rgb_psnr": 10 * math.log10(255**2 / mse)})
                rows.append(
                    {
                        "tick": tick,
                        "unit": name,
                        "box": [x0, y0, x1, y1],
                        "ordinary": vals[0],
                        "pointstream": vals[1],
                    }
                )
    cap.release()
    assert len(rows) == 24 and len(preview_rows) == 3
    assert luma_sse["ordinary"] == 415193395
    preview = args.output / "return-preview.png"
    assert cv2.imwrite(str(preview), np.vstack(preview_rows))
    means = {
        arm: {
            metric: float(np.mean([r[arm][metric] for r in rows]))
            for metric in ("dists", "rgb_psnr")
        }
        for arm in ("ordinary", "pointstream")
    }
    by_unit = {
        name: {
            arm: {
                metric: float(np.mean([row[arm][metric] for row in rows if row["unit"] == name]))
                for metric in ("dists", "rgb_psnr")
            }
            for arm in ("ordinary", "pointstream")
        }
        for name in names
    }
    result = {
        "ticks": TICKS,
        "runner_sha256": sha(Path(__file__)),
        "finish_archive_sha256": args.finish_sha256,
        "preview_sha256": sha(preview),
        "preview_ticks": [30, 110, 190],
        "opencv": cv2.__version__,
        "torch": str(torch.__version__),
        "support": "paid expanded player boxes, clipped to source, 256x256 RGB",
        "weights": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
        "dists_network_record": quality.dists_record(VGG),
        "per_player_equal": True,
        "all_frame_luma_sse": luma_sse,
        "all_frame_luma_psnr": {
            k: 10 * math.log10(255**2 * 480 * 272 * 240 / v) for k, v in luma_sse.items()
        },
        "rows": rows,
        "means": means,
        "by_unit": by_unit,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "opened development return, sender-derived rectangles, no independent labels",
    }
    (args.output / "regions.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {"means": means, "rows": len(rows), "elapsed_seconds": result["elapsed_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
