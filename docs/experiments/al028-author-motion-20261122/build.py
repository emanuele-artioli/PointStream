"""Build an authenticated private 25-fps four-way author review once joint output exists."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
SOURCE = Path("/private/tmp/ps-al028-review-source-20261122.mp4")
AV1 = Path("/private/tmp/ps-nov22-al028-exact-budget/ordinary-52.75/decoded.yuv")
DCVC = Path("/private/tmp/ps-nov22-al028-q24-full/full.yuv")
EXPECTED = {
    SOURCE: "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934",
    AV1: "c60215a513435b726bf9c3dd398b651b1c9add176245a354eb1076ee21445bf1",
    DCVC: "bac9b77792126c748a7640eee2ce80649f54c2509d5973623d06e1ed35b931bb",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def display(yuv, w, h):
    full = cv2.resize(yuv_to_rgb(yuv, w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(full, (960, 540), interpolation=cv2.INTER_AREA)


def text(board, s, xy, scale=0.57):
    cv2.putText(board, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (28, 39, 47), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joint-dir", type=Path, required=True)
    p.add_argument("--result-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    selected = json.loads((HERE / "source-selection.json").read_text())
    ticks = [v["tick"] for v in selected["ticks"]]
    assert ticks == [40, 120, 200]
    for path, digest in EXPECTED.items():
        assert sha(path) == digest
    resultfile = a.joint_dir / "result.json"
    assert sha(resultfile) == a.result_sha256
    result = json.loads(resultfile.read_text())
    assert result["frames"] == 240 and abs(result["joint_warm_kbps"]["20"] - 20.4075) < 1e-8
    joint = np.load(a.joint_dir / "frames-20.npy", mmap_mode="r")
    assert joint.shape == (240, 1080, 1920, 3) and joint.dtype == np.uint8
    av1 = np.memmap(AV1, np.uint8, "r", shape=(240, 288, 320))
    dcvc = np.memmap(DCVC, np.uint8, "r", shape=(240, 816, 960))
    a.output.mkdir(parents=True, exist_ok=False)
    video = a.output / "four-way-25fps.mp4"
    ffmpeg = Path(subprocess.check_output(["which", "ffmpeg"], text=True).strip())
    ffprobe = Path(subprocess.check_output(["which", "ffprobe"], text=True).strip())
    cmd = [
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        "1920x680",
        "-framerate",
        "25",
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
    cap = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG)
    assert cap.isOpened()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    digest = hashlib.sha256()
    shots = []
    source_checks = []
    try:
        for t in range(240):
            ok, bgr = cap.read()
            assert ok and bgr.shape == (1080, 1920, 3)
            source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if t in ticks:
                expected = next(x for x in selected["ticks"] if x["tick"] == t)[
                    "source_native_rgb_sha256"
                ]
                assert hashlib.sha256(source.tobytes()).hexdigest() == expected
                source_checks.append(t)
            raw_joint = np.asarray(joint[t])
            digest.update(raw_joint.tobytes())
            images = [
                cv2.resize(source, (960, 540), interpolation=cv2.INTER_AREA),
                display(av1[t], 320, 192),
                display(dcvc[t], 960, 544),
                cv2.resize(raw_joint, (960, 540), interpolation=cv2.INTER_AREA),
            ]
            crops = [im[295:510, 300:560] for im in images]
            if t in ticks:
                shots.append((t, [np.array(x, copy=True) for x in crops]))
            board = np.full((680, 1920, 3), 248, np.uint8)
            titles = [
                "SOURCE | native video",
                "AV1 | 20.3608 kbps",
                "DCVC-UF | 20.4117 kbps",
                "POINTSTREAM | 20.4075 kbps warm",
            ]
            for j, (im, crop, title) in enumerate(zip(images, crops, titles, strict=True)):
                x = j * 480
                text(board, title, (x + 9, 25), 0.49)
                board[36:306, x : x + 480] = cv2.resize(
                    im, (480, 270), interpolation=cv2.INTER_AREA
                )
                board[320:630, x + 52 : x + 428] = cv2.resize(
                    crop, (376, 310), interpolation=cv2.INTER_CUBIC
                )
            text(
                board,
                f"Tick {t:03d}/239 | {t / 25:.2f} s | 25 fps | Same fixed near-player crop",
                (13, 655),
                0.55,
            )
            text(
                board,
                "PointStream cold: 151.67 kbps; warm assumes valid paid references",
                (865, 655),
                0.55,
            )
            proc.stdin.write(board.tobytes())
        assert digest.hexdigest() == result["rgb_stream_sha256"]["20"] and source_checks == ticks
        proc.stdin.close()
        assert proc.wait(timeout=120) == 0
    finally:
        cap.release()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    info = json.loads(
        subprocess.run(
            [
                str(ffprobe),
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
        ).stdout
    )["streams"][0]
    assert (info["width"], info["height"], info["r_frame_rate"], int(info["nb_frames"])) == (
        1920,
        680,
        "25/1",
        240,
    ) and abs(float(info["duration"]) - 9.6) < 0.001
    fig, axes = plt.subplots(3, 4, figsize=(10.7, 7), layout="constrained")
    titles = [
        "Source",
        "AV1\n20.3608 kbps",
        "DCVC-UF\n20.4117 kbps",
        "PointStream\n20.4075 kbps warm",
    ]
    for i, (t, images) in enumerate(shots):
        for j, im in enumerate(images):
            axes[i, j].imshow(im)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=10, weight="bold")
            if j == 0:
                axes[i, j].set_ylabel(f"{t / 25:.2f} s\nphase {i + 1}", fontsize=10)
    fig.suptitle("Al028 · three source-selected phases", fontsize=14, weight="bold")
    fig.supxlabel("Same fixed near-player crop · 25 fps · PointStream cold 151.67 kbps", fontsize=9)
    fig.savefig(a.output / "source-selected-crops.pdf")
    fig.savefig(a.output / "source-selected-crops.png", dpi=180)
    plt.close(fig)
    receipt = {
        "source_sha256": EXPECTED[SOURCE],
        "av1_yuv_sha256": EXPECTED[AV1],
        "dcvc_yuv_sha256": EXPECTED[DCVC],
        "joint_result_sha256": a.result_sha256,
        "joint_rgb_sha256": digest.hexdigest(),
        "source_selected_native_ticks_verified": source_checks,
        "native_fps": [25, 1],
        "frames": 240,
        "duration_seconds": 9.6,
        "video_sha256": sha(video),
        "still_pdf_sha256": sha(a.output / "source-selected-crops.pdf"),
        "still_png_sha256": sha(a.output / "source-selected-crops.png"),
        "ffmpeg_path": str(ffmpeg),
        "ffmpeg_sha256": sha(ffmpeg),
        "baseline_kbps": {"av1": 20.360833333333332, "dcvc": 20.411666666666665},
        "joint_kbps": {"warm": 20.4075, "cold": 20.4075 + 157517 * 8 / 9600},
        "fixed_crop_960x540_xyxy": [300, 295, 560, 510],
        "scope": "private 25fps review re-encode; source-selected moments; generated joint bytes checked against complete composed-frame receipt; not a formal subjective evaluation",
    }
    (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
