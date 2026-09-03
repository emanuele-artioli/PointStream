"""Code the next plate as a P-frame against the previous one, not as a subtraction.

`plans/done/BP24-findings.md` §17 tested the wrong mechanism. It subtracted two plates
pixel by pixel and coded the difference, which is the worst available way to do
this: subtraction destroys the spatial correlation a transform coder depends on,
turning every misaligned edge into a double edge. Unsurprisingly the delta cost
*more* than coding the plate fresh.

**A video codec already solves exactly this problem.** Inter prediction does
block-wise motion search, so a camera that panned between two plates is handled
by motion vectors rather than by a difference full of edges. So the right
question is not "how big is `B - A`?" but "how big is **B as a P-frame whose
reference is A**?"

That reframing has a pleasing consequence: **the sequence of per-scene plates is
itself a video**, at roughly one frame per point. Coding it as one is not a new
mechanism, it is the ordinary one.

**What is measured here.** Frames `[A, B]` are encoded as a two-frame video with
one I-frame and one P-frame, and `ffprobe` reports each frame's size. The
P-frame's size is the *marginal* cost of adding B to a stream that already
carries A — which is exactly the cost the proposal claims, since the client
already holds A from the previous scene's payload. It is compared against coding
B alone, all-intra, at the same encoder and setting.

**The control matters more than the arms.** Two consecutive frames *within* one
scene are nearly identical, so their P-frame must come out tiny. If it does not,
the harness is not measuring inter prediction and neither arm means anything.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.headroom.real import load_rgb_stack
from experiments.tier.clip import BP21_CLIPS
from src.components.codec import tools
from src.contracts import paths as ps_paths

OUT = ps_paths.outputs() / "bp24-ladder" / "plate-interframe.json"

#: One modern and one widely-available encoder, both driven through ffmpeg so
#: per-frame sizes come back the same way. `SvtAv1EncApp` is the project's av1
#: path but it is a standalone binary that does not report per-frame sizes, and
#: the question here is about the technique rather than about a specific build.
ENCODERS = (
    ("libx265", ("-preset", "veryfast", "-x265-params", "log-level=none")),
    ("libaom-av1", ("-cpu-used", "8", "-usage", "realtime")),
)
CRFS = (28, 38)


def _frames(video: str, scene: str, count: int = 1, offset: int = 0) -> np.ndarray:
    pngs = sorted((BP21_CLIPS / video / scene / "window").glob("frame_*.png"))
    chosen = pngs[offset : offset + count]
    if len(chosen) < count:
        raise FileNotFoundError(f"{video}/{scene} has too few frames")
    return load_rgb_stack(chosen)


def _write_y4m_like(frames: np.ndarray, path: Path, ffmpeg: str) -> None:
    """Frames to a lossless intermediate the encoders can all read."""
    count, height, width, _ = frames.shape
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-framerate", "1",
            "-i", "-", "-c:v", "ffv1", str(path),
        ],
        input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes(),
        check=True,
        capture_output=True,
    )


def _frame_sizes(path: Path, ffprobe: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            ffprobe, "-hide_banner", "-loglevel", "error",
            "-select_streams", "v:0", "-show_entries", "frame=pkt_size,pict_type",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout.decode("utf-8", "replace"))
    return [
        {"size": int(f.get("pkt_size", 0)), "type": f.get("pict_type", "?")}
        for f in payload.get("frames", [])
    ]


def measure(
    pair_frames: np.ndarray, encoder: str, extra: tuple[str, ...], crf: int
) -> dict[str, Any]:
    """Two-frame inter encode, and the same second frame coded all-intra."""
    ffmpeg = tools.resolve_ffmpeg().path
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))

    with tempfile.TemporaryDirectory(prefix="ps_inter_") as tmp_dir:
        tmp = Path(tmp_dir)
        source = tmp / "pair.mkv"
        _write_y4m_like(pair_frames, source, ffmpeg)

        # One I-frame then one P-frame: the default GOP is long enough that the
        # second frame will not be forced to intra.
        inter = tmp / "inter.mkv"
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
             "-c:v", encoder, "-crf", str(crf), *extra, "-g", "240", str(inter)],
            check=True, capture_output=True,
        )
        inter_sizes = _frame_sizes(inter, ffprobe)

        # The same second frame, alone, all-intra: what sending it fresh costs.
        second = tmp / "second.mkv"
        _write_y4m_like(pair_frames[1:2], second, ffmpeg)
        intra = tmp / "intra.mkv"
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(second),
             "-c:v", encoder, "-crf", str(crf), *extra, str(intra)],
            check=True, capture_output=True,
        )
        intra_sizes = _frame_sizes(intra, ffprobe)

    p_frames = [f for f in inter_sizes[1:]]
    marginal = sum(f["size"] for f in p_frames)
    fresh = sum(f["size"] for f in intra_sizes)
    return {
        "encoder": encoder,
        "crf": crf,
        "inter_frame_types": [f["type"] for f in inter_sizes],
        "keyframe_bytes": inter_sizes[0]["size"] if inter_sizes else None,
        "marginal_pframe_bytes": marginal,
        "fresh_intra_bytes": fresh,
        "marginal_over_fresh": round(marginal / fresh, 3) if fresh else None,
    }


def main() -> int:
    blocks: list[dict[str, Any]] = []

    # The control, first and on purpose: two consecutive frames of one scene are
    # nearly identical, so the P-frame must be tiny. If it is not, nothing below
    # is measuring inter prediction.
    control = _frames("alcaraz_highlights", "scene_000", count=2)
    for encoder, extra in ENCODERS:
        for crf in CRFS:
            try:
                row = measure(control, encoder, extra, crf)
            except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
                row = {"encoder": encoder, "crf": crf, "error": repr(exc)}
            row["arm"] = "CONTROL consecutive frames, same scene"
            blocks.append(row)

    for video, scene_a, scene_b in (
        ("alcaraz_highlights", "scene_000", "scene_010"),
        ("federer_djokovic", "scene_001", "scene_003"),
    ):
        pair = np.concatenate(
            [_frames(video, scene_a, 1), _frames(video, scene_b, 1)], axis=0
        )
        for encoder, extra in ENCODERS:
            for crf in CRFS:
                try:
                    row = measure(pair, encoder, extra, crf)
                except Exception as exc:  # noqa: BLE001
                    row = {"encoder": encoder, "crf": crf, "error": repr(exc)}
                row["arm"] = f"{video} {scene_a} -> {scene_b}"
                blocks.append(row)

    payload = {
        "question": (
            "how big is plate B as a P-frame referencing plate A, against "
            "coding B fresh as intra?"
        ),
        "why_this_supersedes_the_subtraction_test": (
            "Pixel subtraction destroys the spatial correlation a transform "
            "coder depends on. Inter prediction does block-wise motion search, "
            "which is the mechanism a panned camera actually needs. The earlier "
            "delta test measured the wrong thing."
        ),
        "reading": (
            "marginal_over_fresh below 1.0 means sending B as a P-frame against "
            "A is cheaper than sending it fresh, i.e. the proposal pays."
        ),
        "rows": blocks,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in blocks:
        if "error" in row:
            print(f"  {row['arm'][:38]:>38}  {row['encoder']:>11} crf{row['crf']}  FAILED {row['error'][:60]}", flush=True)
            continue
        print(
            f"  {row['arm'][:38]:>38}  {row['encoder']:>11} crf{row['crf']}  "
            f"types={''.join(row['inter_frame_types'])}  "
            f"P={row['marginal_pframe_bytes']:>9,} B  "
            f"fresh={row['fresh_intra_bytes']:>9,} B  "
            f"ratio={row['marginal_over_fresh']}",
            flush=True,
        )
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
