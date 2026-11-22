"""Encoder-only joint player/ball supports and same-QP ROI effectiveness check."""

import json
import subprocess
import tarfile

import cv2
import numpy as np

from experiments.tennis.k4b_pose import player_masks
from experiments.tennis.w1_roi import roi_text
from experiments.tennis.shanghai_ball_rate_crf import sha
from src.codecs import svtav1
from src.pointstream import ball
from src.segmentation import masks as mask_format
from src.segmentation.masks import ClipMasks

FEATURE_SHA = "994101c8d3466080777c262d92b9f15953c5636cc4addae06f75ad478d3594ee"
BALL_SHA = "3ad726835c0e51c11daa80d9b54630a1e52494227680b434254e583950426918"
IDS = ["person:000000000000", "person:000000000001"]


def region_map(features, payload, n, width, height, work):
    assert sha(features) == FEATURE_SHA and sha(payload) == BALL_SHA
    with tarfile.open(features) as tar:
        prefix = "publish/test__match102@120/"
        f = json.load(tar.extractfile(prefix + "features.json"))
        rle = tar.extractfile(prefix + "sam-masks.rle").read()
    if mask_format.zstd is None and rle.startswith(b"\x28\xb5\x2f\xfd"):
        raw = subprocess.run(
            ["zstd", "-q", "-d", "-c"], input=rle, check=True, capture_output=True
        ).stdout
    else:
        raw = mask_format.decompress(rle)
    clip = ClipMasks.from_doc(json.loads(raw))
    assert len(clip.frames) == 240
    pos, _ = ball.decode_frames(payload.read_bytes(), 240, 960, 544)
    regions = np.zeros((n, 540, 960), bool)
    for t in range(n):
        masks = player_masks(clip.frames[t], f["aliases"], IDS)
        assert set(masks) == set(IDS)
        for actor in IDS:
            m = masks[actor]
            assert m.shape == (540, 960) and m.any()
            regions[t] |= m
        if pos[t] is not None:
            x = round(pos[t][0])
            y = round(pos[t][1] * 540 / 544)
            regions[t, max(0, y - 8) : min(540, y + 9), max(0, x - 8) : min(960, x + 9)] = True
    text = roi_text(regions, width, height, -40, outside=True)
    assert len(text.splitlines()) == n and all(
        "0" in row.split()[1:] and "40" in row.split()[1:] for row in text.splitlines()
    )
    path = work / "roi.txt"
    path.write_text(text)
    return path, regions


def compare_same_q(
    native, selected_yuv, row, source_regions, width, height, frames, fps, tools, work, pub
):
    q = row["quarter_crf"]
    twin = pub / f"plain-twin-q{q}.ivf"
    cmd = svtav1.encode_command(
        tools["SvtAv1EncApp"]["path"],
        native,
        twin,
        width=width,
        height=height,
        fps=fps,
        frames=frames,
        crf=q / 4,
        preset=4,
        full_range=False,
    )
    subprocess.run(cmd, check=True, capture_output=True, timeout=500)
    assert len(svtav1.ivf_frames(twin.read_bytes())) == frames
    other = work / "plain-twin.yuv"
    subprocess.run(
        svtav1.decode_command(tools["dav1d"]["path"], twin, other, threads=1),
        check=True,
        capture_output=True,
        timeout=250,
    )
    shape = (frames, height * 3 // 2, width)
    arrays = [np.memmap(p, np.uint8, "r", shape=shape) for p in (native, other, selected_yuv)]
    sse = np.zeros((2, 2), np.int64)
    sizes = np.zeros(2, np.int64)
    changed = 0
    for tick in range(frames):
        m = cv2.resize(
            source_regions[tick].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        truth, ordinary, encoded = [x[tick, :height].astype(np.int16) for x in arrays]
        changed += int(np.count_nonzero(ordinary != encoded))
        for j, region in enumerate((m, ~m)):
            sizes[j] += region.sum()
            for i, reconstruction in enumerate((ordinary, encoded)):
                delta = (truth[region] - reconstruction[region]).astype(np.int32)
                sse[i, j] += np.square(delta).sum(dtype=np.int64)
    assert sizes.min() > 0

    def psnr(i, j):
        return float(10 * np.log10(255.0**2 * sizes[j] / sse[i, j])) if sse[i, j] else None

    return dict(
        twin_file=twin.name,
        twin_bytes=twin.stat().st_size,
        twin_sha256=sha(twin),
        twin_decoded_sha256=sha(other),
        selected_decoded_sha256=sha(selected_yuv),
        changed_luma_pixels=changed,
        non_inert=sha(other) != sha(selected_yuv) and changed > 0,
        support_pixels=int(sizes[0]),
        outside_pixels=int(sizes[1]),
        plain_luma_psnr_in=psnr(0, 0),
        roi_luma_psnr_in=psnr(1, 0),
        plain_luma_psnr_out=psnr(0, 1),
        roi_luma_psnr_out=psnr(1, 1),
    )
