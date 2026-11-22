"""Compose six actual decoded arms on the original reviewed supports."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs/experiments/receiver-replay-20261118"))
sys.path.insert(0, str(ROOT / "docs/experiments/clean-court-regional-screen-20261118"))
from independent_crop_audit import array_frames, file_sha, read_small, require  # noqa: E402
from independent_finish_audit import UNITS, bind_generation, overlay  # noqa: E402
from crops_support import supports  # noqa: E402
from experiments.tennis import common_player_ball as ball  # noqa: E402
from src.pointstream.tracked_patch import yuv_to_rgb  # noqa: E402

SOURCE = Path("/Users/manu/Datasets/pointstream-submission-20261118")
STAGE = Path("/private/tmp/ps-joint-campus-stage-20261118")
AUDIT = Path("/private/tmp/pointstream-independent-idle-nov18")
RUN = Path("/private/tmp/agentctl-artifacts/Emanueles-MacBook-Pro.local/9897d39530594990a4b5cc4ab12ceaf9/output/full-quarter")
LABELS = SOURCE / "foreground-human-reviewed-20261118/9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a.json"
BUNDLE = SOURCE / "ps-development-source-refs-20261118-native/source-only-review.zip"
WEIGHTS = Path("/Users/manu/.codex/visualizations/2026/11/01/01a196fe-ab4d-70f0-a4b4-6843ca289667/atlas-metric-models/checkpoints/vgg16-397923af.pth")
TICKS = list(range(10, 240, 20))


def png(rgb):
    ok, encoded = cv2.imencode(".png", rgb, [cv2.IMWRITE_PNG_COMPRESSION, 5])
    require(ok, "PNG encoder")
    return encoded.tobytes()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--count", type=int, choices=(1,12), required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    start = time.process_time()
    cv2.setNumThreads(1)
    require(file_sha(LABELS) == "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a", "labels")
    require(file_sha(BUNDLE) == "5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb", "source")
    manifest = json.loads((RUN / "report.json").read_text())
    rows = {(r["arm"], r["crf"]): r for r in manifest["outputs"]}
    videos = {}
    for key, row in rows.items():
        name = f"{key[0]}-{key[1]}"
        path = RUN / name / "decoded.yuv"
        require(file_sha(path) == row["decoded_sha256"], "decoded frame identity")
        videos[name] = np.memmap(path, np.uint8, "r", shape=(240,204,240))
    carrier = (RUN / "clean-48/players.pscp").read_bytes()
    sidecar = (RUN / "clean-48/ball.sidecar").read_bytes()
    sha_dec = manifest["tools"][2]["sha256"]
    positions, tile, canvas_size = ball._open(carrier, sidecar, sha_dec)
    require(canvas_size == (1920,1080), "ball canvas")
    raw = subprocess.check_output(["/opt/homebrew/bin/ffmpeg","-v","error","-threads","1","-c:v","libdav1d",
                                   "-f","ivf","-i","pipe:0","-pix_fmt","yuv420p","-f","rawvideo","pipe:1"],
                                  input=tile, timeout=30)
    tiles = np.frombuffer(raw, np.uint8).reshape((240,96,64))
    records = json.loads(LABELS.read_text())["frames"][:args.count]
    require([v["tick"] for v in records] == TICKS[:args.count], "tick selection")
    ticks = {v["tick"]: v for v in records}
    finish = json.loads((STAGE / "finish/finish.json").read_text())
    audit = json.loads((STAGE / "finish-full-audit.json").read_text())
    require(audit["all_frames_exact"] and audit["carrier_sha256"] == finish["carrier_sha256"], "receiver")
    inventory = []
    squared_errors = {name: 0 for name in videos}
    with ExitStack() as stack:
        source = stack.enter_context(zipfile.ZipFile(BUNDLE))
        output = stack.enter_context(zipfile.ZipFile(args.output / "crops.zip", "x"))
        generations = dict(bind_generation(stack, AUDIT / f"player{i}-new-full.tar",
                                             STAGE / f"player{i}-full.json", finish) for i in (1,2))
        masks = {}
        for unit in UNITS:
            path = STAGE / "finish" / (unit + "-masks.npz")
            require(file_sha(path) == finish["mask_files"][unit], "mask identity")
            zipped = stack.enter_context(zipfile.ZipFile(path))
            require(np.array_equal(read_small(zipped, "t"), np.arange(240)), "mask timeline")
            masks[unit] = stack.enter_context(array_frames(zipped, "masks", (240,784,512), np.bool_))
        for tick in range(max(ticks)+1):
            crops = {unit: next(generations[unit][1]) for unit in UNITS}
            cur_masks = {unit: next(masks[unit]) for unit in UNITS}
            if tick not in ticks:
                continue
            label = ticks[tick]
            src = source.read(f"frame-{tick:03d}/source.png")
            require(hashlib.sha256(src).hexdigest() == label["source_png_sha256"], "source tick")
            source_rgb = cv2.imdecode(np.frombuffer(src,np.uint8),cv2.IMREAD_COLOR)[:,:,::-1]
            images = {"source": source_rgb}
            for name, frames in videos.items():
                image = cv2.resize(yuv_to_rgb(frames[tick],240,136,False),(1920,1080),interpolation=cv2.INTER_CUBIC)
                if name.startswith("clean"):
                    support = np.zeros((1080,1920),bool)
                    for unit in UNITS:
                        area, status = overlay(image,support,crops[unit],generations[unit][0][tick],cur_masks[unit])
                        require(area == finish["rows"][tick]["projected_pixels"][unit] and
                                status == finish["rows"][tick]["status"][unit], "player projection")
                    if positions[tick] is not None:
                        x,y = positions[tick]
                        image[y:y+32,x:x+32] = yuv_to_rgb(tiles[tick],64,64,False)[16:48,16:48]
                images[name] = image
                delta = image.astype(np.int32) - source_rgb.astype(np.int32)
                squared_errors[name] += int(np.sum(delta.astype(np.int64) ** 2))
            for who, y, x, mask in supports(label):
                prefix = f"{tick:03d}/{who}/"
                output.writestr(prefix+"mask.png",png(mask))
                for name,image in images.items():
                    output.writestr(prefix+name+".png",png(image[y,x,::-1]))
            inventory.append({"tick":tick, "ball_placement":positions[tick]})
    pixels = len(records) * 1080 * 1920 * 3
    full_frame_psnr = {name: 10*math.log10(255**2*pixels/error) for name,error in squared_errors.items()}
    report = {"full_frame_rgb_psnr":full_frame_psnr, "full_frame_squared_errors":squared_errors, "ticks":inventory, "input_report_sha256":file_sha(RUN/"report.json"),
              "crops_sha256":file_sha(args.output/"crops.zip"),
              "cpu_seconds":time.process_time()-start, "scope":"native source matched regions, six paid decoded streams"}
    if args.count == 1:
        require(report["cpu_seconds"] * 12 * 1.5 + 200 < 900, "crop forecast exceeds budget")
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.output/"metrics.json").write_text(json.dumps({"valid":True,"metrics":{"ticks":len(inventory)},
                                                        "cost":report["cpu_seconds"]})+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
