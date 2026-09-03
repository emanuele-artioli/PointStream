"""Does the 31-53% inter-frame saving survive a low-delay encoder?

`plans/done/BP24-findings.md` §18 measured that coding the next plate as a P-frame
against the previous saves 31-53% with av1. `plans/done/BP30-background-stream.md` §1
records why that number cannot be quoted yet: it was measured with the encoders'
default configuration, which allows **B-frames and lookahead**, and both let
frame *n*'s decisions depend on frame *n+1*.

A scheme that depends on the future is not one where each scene's payload is
independent of every future scene — which is the whole claim. At two frames it
made no difference, because there was no future frame to look at. Over a real
sequence it will.

So this re-measures the same pairs under **low-delay P**: no B-frames, no
lookahead, no multi-pass. If the saving survives, it is achievable live. If it
collapses, §18's number belongs to an offline archiver and BP30's premise is
much weaker.

**The control comes first and decides whether anything else is readable.** Two
consecutive frames of one scene must still come back as a few percent. If
low-delay flags have accidentally disabled inter prediction altogether, the
control is where that shows, and every other row would otherwise read as an
honest-looking 1.0.
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

OUT = ps_paths.outputs() / "bp24-ladder" / "plate-lowdelay.json"

#: Each encoder twice: as §18 ran it, and forbidden from looking ahead.
#:
#: The low-delay flags are per-encoder because there is no portable spelling.
#: `-bf 0` covers ffmpeg's own B-frame insertion; the encoder-specific parameter
#: covers its internal lookahead, which `-bf 0` does not touch.
CONFIGS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("libx265", "default (as findings §18)", ("-preset", "veryfast", "-x265-params", "log-level=none")),
    (
        # `rc-lookahead=0` is not a valid low-delay setting for x265: it is the
        # rate-control lookahead, not the prediction lookahead, and zeroing it
        # made a P-frame between two nearly identical frames come back *larger*
        # than a fresh intra (control ratio 1.075). That is the control catching
        # a broken configuration, which is what it is for. Forbid B-frames and
        # leave rate control alone.
        "libx265",
        "no B-frames",
        ("-preset", "veryfast", "-bf", "0", "-x265-params", "log-level=none:bframes=0"),
    ),
    (
        # Strict causality also needs zero *rate-control* lookahead. x265's
        # minimum usable value is not zero, so this arm reports what a bounded
        # lookahead costs; for one plate per scene, a lookahead of N frames is a
        # delay of N scenes, which is why av1 mattering here is not academic.
        "libx265",
        "no B-frames, lookahead 1",
        ("-preset", "veryfast", "-bf", "0",
         "-x265-params", "log-level=none:bframes=0:rc-lookahead=1"),
    ),
    ("libaom-av1", "default (as findings §18)", ("-cpu-used", "8", "-usage", "realtime")),
    (
        "libaom-av1",
        "low-delay P",
        ("-cpu-used", "8", "-usage", "realtime", "-lag-in-frames", "0", "-bf", "0"),
    ),
)
CRF = 38


def _frames(video: str, scene: str, count: int = 1) -> np.ndarray:
    pngs = sorted((BP21_CLIPS / video / scene / "window").glob("frame_*.png"))
    if len(pngs) < count:
        raise FileNotFoundError(f"{video}/{scene} has too few frames")
    return load_rgb_stack(pngs[:count])


def _lossless(frames: np.ndarray, path: Path, ffmpeg: str) -> None:
    _, height, width, _ = frames.shape
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-framerate", "1", "-i", "-", "-c:v", "ffv1", str(path)],
        input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes(),
        check=True, capture_output=True,
    )


def _sizes(path: Path, ffprobe: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [ffprobe, "-hide_banner", "-loglevel", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pkt_size,pict_type", "-of", "json", str(path)],
        check=True, capture_output=True,
    )
    return [
        {"size": int(f.get("pkt_size", 0)), "type": f.get("pict_type", "?")}
        for f in json.loads(result.stdout.decode("utf-8", "replace")).get("frames", [])
    ]


def measure(pair: np.ndarray, encoder: str, extra: tuple[str, ...]) -> dict[str, Any]:
    ffmpeg = tools.resolve_ffmpeg().path
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    with tempfile.TemporaryDirectory(prefix="ps_ld_") as tmp_dir:
        tmp = Path(tmp_dir)
        src = tmp / "pair.mkv"
        _lossless(pair, src, ffmpeg)
        inter = tmp / "inter.mkv"
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-c:v", encoder, "-crf", str(CRF), *extra, "-g", "240", str(inter)],
            check=True, capture_output=True,
        )
        inter_sizes = _sizes(inter, ffprobe)

        second = tmp / "second.mkv"
        _lossless(pair[1:2], second, ffmpeg)
        intra = tmp / "intra.mkv"
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(second),
             "-c:v", encoder, "-crf", str(CRF), *extra, str(intra)],
            check=True, capture_output=True,
        )
        intra_sizes = _sizes(intra, ffprobe)

    marginal = sum(f["size"] for f in inter_sizes[1:])
    fresh = sum(f["size"] for f in intra_sizes)
    return {
        "encoder": encoder,
        "frame_types": "".join(f["type"] for f in inter_sizes),
        "marginal_bytes": marginal,
        "fresh_intra_bytes": fresh,
        "marginal_over_fresh": round(marginal / fresh, 3) if fresh else None,
        # A B-frame anywhere means the encode was not causal, whatever the flags
        # claimed. Reported rather than assumed, because a flag existing is not
        # a feature working.
        "contains_b_frames": "B" in "".join(f["type"] for f in inter_sizes),
    }


def main() -> int:
    arms: list[tuple[str, np.ndarray]] = [
        ("CONTROL consecutive frames, one scene",
         _frames("alcaraz_highlights", "scene_000", count=2)),
    ]
    for video, a, b in (
        ("alcaraz_highlights", "scene_000", "scene_010"),
        ("federer_djokovic", "scene_001", "scene_003"),
    ):
        arms.append((
            f"{video} {a} -> {b}",
            np.concatenate([_frames(video, a), _frames(video, b)], axis=0),
        ))

    rows: list[dict[str, Any]] = []
    for arm, pair in arms:
        for encoder, label, extra in CONFIGS:
            try:
                row = measure(pair, encoder, extra)
            except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
                row = {"encoder": encoder, "error": repr(exc)}
            row["arm"] = arm
            row["config"] = label
            rows.append(row)
            if "error" in row:
                print(f"  {arm[:36]:>36} {encoder:>11} {label:<24} FAILED {row['error'][:50]}", flush=True)
                continue
            print(
                f"  {arm[:36]:>36} {encoder:>11} {label:<24} "
                f"types={row['frame_types']:<4} "
                f"marginal={row['marginal_bytes']:>9,} B  "
                f"fresh={row['fresh_intra_bytes']:>9,} B  "
                f"ratio={row['marginal_over_fresh']}",
                flush=True,
            )

    payload = {
        "question": (
            "does the inter-frame plate saving survive a low-delay encoder — no "
            "B-frames, no lookahead — which is what makes each scene's payload "
            "independent of every future scene?"
        ),
        "why": (
            "findings §18 measured with default encoder settings, which allow "
            "B-frames and lookahead. Both let frame n's decisions depend on "
            "frame n+1, so a saving that needs them is not achievable live."
        ),
        "reading": (
            "compare each pair's 'default' and 'low-delay P' rows. If the "
            "low-delay ratio is close to the default one, the saving is causal "
            "and quotable. If it collapses toward 1.0, §18's number belongs to "
            "an offline encoder."
        ),
        "crf": CRF,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
