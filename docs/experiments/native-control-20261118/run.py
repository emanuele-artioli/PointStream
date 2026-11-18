"""Run the registered AV1 controls on an authenticated development source."""
import argparse
import hashlib
import json
import math
import platform
import resource
import struct
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis.w1_roi import roi_text
from src.codecs import svtav1
from src.pointstream.tracked_patch import origin

WIDTH, HEIGHT = 960, 544


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu():
    return sum(r.ru_utime + r.ru_stime for r in (
        resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def confirm(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Input hash mismatch for {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "conditioning", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frames", choices=(24, 240), type=int, required=True)
    parser.add_argument("--ball-track", type=Path, help="Optional authenticated same-rally automatic centers")
    parser.add_argument("--ball-sha256")
    parser.add_argument("--reuse-plain", type=Path, help="Existing matching plain encode report")
    parser.add_argument("--upper-support", action="store_true", help="Run the registered upper-rate control at CRF 28")
    a = parser.parse_args()
    if bool(a.ball_track) != bool(a.ball_sha256) or (a.reuse_plain and not a.ball_track):
        parser.error("A ball track and its hash must be supplied together; plain reuse requires a track")
    if a.upper_support and (not a.ball_track or a.reuse_plain):
        parser.error("The new upper anchor requires the matched-placement map and a fresh plain encode")
    n = a.frames
    a.output.mkdir(exist_ok=False, parents=True)
    begin = time.monotonic()
    cpu_start = cpu()
    cv2.setNumThreads(1)
    confirm(a.config, "eb67e4a972cde36895addef8db306c963423f814a31be7273a31c696ce5e98c6")
    cfg = json.loads(a.config.read_text())
    confirm(a.source, cfg["source"]["source_video_sha256"])
    tools = {k: svtav1.tool(k) for k in ("ffmpeg", "ffprobe", "SvtAv1EncApp", "dav1d")}
    probe = subprocess.check_output([tools["ffprobe"]["path"], "-v", "error", "-show_streams",
                                     "-of", "json", str(a.source)], timeout=10)
    stream, = json.loads(probe)["streams"]
    if [stream[k] for k in ("width", "height", "avg_frame_rate", "nb_frames")] != [1920, 1080, "50/1", "240"]:
        raise ValueError("Source geometry/cadence mismatch")
    boxes = []
    for entry in cfg["units"]:
        path = a.conditioning / (entry["unit"] + ".npz")
        confirm(path, entry["sha256"])
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["t"], np.arange(240)):
                raise ValueError("Unexpected conditioning timestamps")
            boxes.append(data["box"][:n])
    regions = np.zeros((n, HEIGHT, WIDTH), bool)
    for tick, region in enumerate(regions):
        for unit in boxes:
            x, y, w, h = unit[tick]
            x0, y0 = max(0, math.floor(x / 2)), max(0, math.floor(y * HEIGHT / 1080))
            x1, y1 = min(WIDTH, math.ceil((x + w) / 2)), min(HEIGHT, math.ceil((y + h) * HEIGHT / 1080))
            region[y0:y1, x0:x1] = True
        if not (0 < region.sum() < HEIGHT * WIDTH):
            raise ValueError("Unusable player boxes")
    if a.ball_track:
        confirm(a.ball_track, a.ball_sha256)
        track = json.loads(a.ball_track.read_text())
        if (track["source_sha256"] != sha(a.source) or track["fps"] != [50, 1]
                or track["canvas_size"] != [1920, 1080] or len(track["centers"]) != 240):
            raise ValueError("Ball track incompatible with this source")
        for region, center in zip(regions, track["centers"][:n], strict=True):
            if center is not None:
                x, y = origin(center, 1920, 1080)
                region[math.floor(y*HEIGHT/1080):math.ceil((y+32)*HEIGHT/1080), x//2:(x+32)//2] = True
    roi = a.output / "roi.txt"
    roi.write_text(roi_text(regions, WIDTH, HEIGHT, -20, outside=True))
    raw = a.output / "source.yuv"
    conversion = [tools["ffmpeg"]["path"], "-v", "error", "-nostdin", "-threads", "2", "-i", str(a.source),
                  "-frames:v", str(n), "-filter_threads", "1", "-vf", "scale=960:544:flags=area:out_range=tv,format=yuv420p",
                  "-fps_mode", "passthrough", "-f", "rawvideo", str(raw)]
    subprocess.run(conversion, check=True, timeout=60)
    shape = (n, HEIGHT * 3 // 2, WIDTH)
    if raw.stat().st_size != math.prod(shape):
        raise ValueError("Source decode incomplete")
    source = np.memmap(raw, dtype=np.uint8, mode="r", shape=shape)
    rows = []
    crfs = [28] if a.upper_support else ([48] if n == 24 else [36, 48, 60])
    for crf in crfs:
        for mode in ("plain", "roi"):
            work = a.output / f"{mode}-{crf}"
            work.mkdir()
            if a.reuse_plain and mode == "plain":
                saved = json.loads(a.reuse_plain.read_text())
                if (saved["source_sha256"] != sha(a.source) or saved["scaled_yuv_sha256"] != sha(raw)
                        or saved["tools"] != tools or saved["frames"] != n):
                    raise ValueError("Different source, timing, or tools in plain control")
                row, = [r.copy() for r in saved["rows"] if r["crf"] == crf and r["mode"] == "plain"]
                confirm(Path(row["stream"]), row["stream_sha256"])
                confirm(Path(row["decoded"]), row["decoded_sha256"])
            else:
                row = svtav1.code(raw, work, width=WIDTH, height=HEIGHT, fps=Fraction(50), frames=n, crf=crf,
                                  preset=4, threads=2, full_range=False, encoder=tools["SvtAv1EncApp"]["path"],
                                  decoder=tools["dav1d"]["path"], timeout=150,
                                  extra=["--roi-map-file", str(roi)] if mode == "roi" else [])
            stream_data = Path(row["stream"]).read_bytes()
            sizes = svtav1.ivf_frames(stream_data)
            offset = 32
            if struct.unpack_from("<HHIIII", stream_data, 12) != (WIDTH, HEIGHT, 50, 1, n, 0):
                raise ValueError("IVF header changed")
            for tick, size in enumerate(sizes):
                if struct.unpack_from("<IQ", stream_data, offset) != (size, tick):
                    raise ValueError("IVF timing mismatch")
                offset += 12 + size
            if row["decoded_frames"] != n or len(sizes) != n or row["file_bytes"] != sum(sizes) + 32 + n * 12:
                raise ValueError("Incomplete native reconstruction or byte account")
            decoded = np.memmap(row["decoded"], dtype=np.uint8, mode="r", shape=shape)
            inside, outside = 0., 0.
            for tick, region in enumerate(regions):
                squared = np.square(decoded[tick, :HEIGHT].astype(float) - source[tick, :HEIGHT])
                inside += squared[region].sum()
                outside += squared[~region].sum()
            box_count = int(regions.sum())
            row.update(mode=mode, crf=crf, decoded_sha256=sha(Path(row["decoded"])),
                       complete_kbps=row["file_bytes"] * 8 * 50 / n / 1000,
                       box_y_psnr=10 * math.log10(255**2 * box_count / inside),
                       complement_y_psnr=10 * math.log10(255**2 * (regions.size - box_count) / outside))
            rows.append(row)
        plain, allocated = rows[-2:]
        if allocated["decoded_sha256"] == plain["decoded_sha256"] or (
                n == 24 and allocated["file_bytes"] >= plain["file_bytes"]):
            raise ValueError(f"Allocation smoke or decoded difference failed, CRF {crf}")
    report = dict(frames=n, source=a.source.name, source_sha256=sha(a.source), tools=tools,
                  conversion=conversion, scaled_yuv_sha256=sha(raw), map_sha256=sha(roi),
                  opencv=cv2.__version__, numpy=np.__version__, host=platform.platform(), gpu=None,
                  script_sha256=sha(Path(__file__)), root_revision=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], text=True).strip(), config_sha256=sha(a.config),
                  elapsed_seconds=time.monotonic()-begin, cpu_seconds=cpu()-cpu_start, rows=rows)
    if a.ball_track:
        report.update(ball_track_sha256=a.ball_sha256, plain_report_sha256=sha(a.reuse_plain) if a.reuse_plain else None,
                      luma_region="union of both player boxes and automatic ball patch footprints")
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{n} frames, {len(rows)} streams, {report['elapsed_seconds']:.2f}s wall, {report['cpu_seconds']:.2f}s CPU")


if __name__ == "__main__":
    main()
