"""Create a private full-cadence review copy and verify the joint display digest."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import zipfile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("joint_trial", HERE / "run.py")
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)
crops = trial.crops


def view(frame, w, h):
    return cv2.resize(
        trial.yuv_to_rgb(frame, w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
    )


def main(args, on_frame=None):
    args.output.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    meta = json.loads((HERE / "result.json").read_text())
    arm = meta["streams"]["20"]
    plain = next(
        r
        for r in json.loads((args.controls / "20-240/result.json").read_text())["selected"].values()
        if r["mode"] == "plain" and r["q"] == 63
    )
    assert (
        plain["file_bytes"] == 12974
        and crops.file_sha(Path(plain["decoded"])) == plain["decoded_sha256"]
    )
    source_path = crops.SOURCE / "ps-native-ctl-20261118-full/source.yuv"
    assert (
        crops.file_sha(source_path)
        == "6cae4bd7a46bfa4bd2a7a7ec5c926647ea43d9e7a5366eb96e464eaffe1e603c"
    )
    source = np.memmap(source_path, np.uint8, "r", shape=(240, 816, 960))
    standard = np.memmap(plain["decoded"], np.uint8, "r", shape=(240, 204, 240))
    native = args.packet / "decoded-20.yuv"
    assert crops.file_sha(native) == arm["decoded_yuv_sha256"]
    bg = np.memmap(native, np.uint8, "r", shape=(240, 144, 160))
    packet = (args.packet / "joint-20.psw").read_bytes()
    assert trial.sha(packet) == arm["wire_sha256"]
    profile, parts = trial.open_packet(packet)
    assert profile["order"] == ["clean_base", "ball", "far_player", "near_player"]
    pts, looks = trial.ball.decode_frames(parts["trajectory"], 240, 960, 544)
    motion = trial.ball.velocities(pts)
    digest = hashlib.sha256()
    finish = json.loads((crops.STAGE / "finish/finish.json").read_text())
    receipt = json.loads((crops.STAGE / "finish-full-audit.json").read_text())
    assert receipt["all_frames_exact"] and finish["carrier_sha256"] == receipt["carrier_sha256"]
    ffmpeg = meta["tools"]["ffmpeg"]
    assert crops.file_sha(Path(ffmpeg["path"])) == ffmpeg["sha256"]
    w, h = 1536, 694
    video = args.output / "joint-serve-comparison.mp4"
    command = [
        ffmpeg["path"],
        "-v",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        "50",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "12",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video),
    ]
    start = time.monotonic()
    with ExitStack() as stack:
        gens = dict(
            crops.bind_generation(
                stack,
                crops.AUDIT / f"player{i}-new-full.tar",
                crops.STAGE / f"player{i}-full.json",
                finish,
            )
            for i in (1, 2)
        )
        masks = {}
        for unit in crops.UNITS:
            path = crops.STAGE / "finish" / (unit + "-masks.npz")
            assert crops.file_sha(path) == finish["mask_files"][unit]
            z = stack.enter_context(zipfile.ZipFile(path))
            masks[unit] = stack.enter_context(
                crops.array_frames(z, "masks", (240, 784, 512), np.bool_)
            )
        proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for t in range(240):
                scene = cv2.resize(
                    trial.yuv_to_rgb(bg[t], 160, 96, False),
                    (960, 544),
                    interpolation=cv2.INTER_CUBIC,
                )
                joint = cv2.resize(
                    trial.ball.render(scene, t, pts, motion, looks),
                    (1920, 1080),
                    interpolation=cv2.INTER_CUBIC,
                )
                support = np.zeros((1080, 1920), bool)
                for unit in crops.UNITS:
                    area, status = crops.overlay(
                        joint, support, next(gens[unit][1]), gens[unit][0][t], next(masks[unit])
                    )
                    assert (area, status) == (
                        finish["rows"][t]["projected_pixels"][unit],
                        finish["rows"][t]["status"][unit],
                    )
                digest.update(joint)
                if on_frame is not None:
                    on_frame(
                        t,
                        trial.yuv_to_rgb(source[t], 960, 544, False),
                        view(standard[t], 240, 136),
                        joint,
                    )
                board = np.full((h, w, 3), 247, np.uint8)
                for j, (img, title) in enumerate(
                    zip(
                        (view(source[t], 960, 544), view(standard[t], 240, 136), joint),
                        (
                            "Source | 960 x 544",
                            "AV1 | 21.623 kbps",
                            "PointStream | 21.545 kbps cached",
                        ),
                        strict=True,
                    )
                ):
                    x = j * 512
                    cv2.putText(
                        board,
                        title,
                        (x + 12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.67,
                        (25, 35, 45),
                        1,
                        cv2.LINE_AA,
                    )
                    board[40:328, x : x + 512] = cv2.resize(
                        img, (512, 288), interpolation=cv2.INTER_AREA
                    )
                    board[340:650, x + 141 : x + 368] = cv2.resize(
                        img[390:1032, 854:1324], (227, 310), interpolation=cv2.INTER_AREA
                    )
                cv2.putText(
                    board,
                    "4.8 s at 50 fps  |  Same crop in every column  |  PointStream cold start: 428.777 kbps",
                    (12, 680),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.64,
                    (25, 35, 45),
                    1,
                    cv2.LINE_AA,
                )
                proc.stdin.write(board.tobytes())
                if t == 23:
                    assert (time.monotonic() - start) * 10 * 1.5 < 260
            proc.stdin.close()
            assert proc.wait(timeout=60) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    assert digest.hexdigest() == arm["all_240_rgb_sha256"]
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(probe.stdout)["streams"][0]
    assert (
        info["width"],
        info["height"],
        info["r_frame_rate"],
        int(info["nb_frames"]),
        float(info["duration"]),
    ) == (w, h, "50/1", 240, 4.8)
    record = {
        "file_sha256": crops.file_sha(video),
        "original_joint_240_rgb_sha256": digest.hexdigest(),
        "ordinary_yuv_sha256": plain["decoded_sha256"],
        "probe": info,
        "scope": "private review encode; displayed coding rates price the original streams",
        "wall_seconds": time.monotonic() - start,
    }
    (args.output / "motion-receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("packet", type=Path)
    cli.add_argument("controls", type=Path)
    cli.add_argument("output", type=Path)
    main(cli.parse_args())
