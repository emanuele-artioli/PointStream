"""One background frame per scene, for a sequence long enough to amortise over.

BP24's ladder measured two clips. `https://github.com/emanuele-artioli/PointStream/blob/ec581e9/plans/prompts/claude-bp30.md` is explicit
that this does not meet the bar `presley` uses (n>=6 videos before a
significance claim), and BP30 needs something worse than that: a *multi-scene*
sequence, where the whole question is what scene *n* costs given scenes 1..n-1.
The cached BP21 windows hold two scenes per video, which cannot answer it.

The dataset already carries the scene decomposition. `assets/dataset/<video>/
scenes/` holds one thumbnail per scene, named with the scene's **end**
timestamp and its duration -- verified rather than assumed: alcaraz_highlights'
last thumbnail ends at 494.277 against the video's 494.353 s duration, and each
name's `end - dur` lands on the previous name's end. So a scene's start is
recoverable and a representative frame can be pulled from the 4K source.

**Which scenes.** The decomposition classifies each scene, and `point` is the
rally view -- the one where the background is the court. Those are the scenes
whose backgrounds are plausibly similar to each other, which is the hypothesis
BP30 exists to price. Interludes and replays are a different camera and would
measure a different question.

**What this is not.** These are scene *frames*, not player-masked plates and not
stitched panoramas. That is deliberate for a first measurement: findings §18 and
§19 measured on exactly this kind of frame, so the numbers here sit on the same
axis as the 31-53% already recorded, and a frame carrying players is the
*conservative* case -- a real background plate has less moving content to
mispredict, not more. Brief §2 recommends this (Option B, background as video)
over normalising a panorama canvas.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.components.codec import tools
from src.contracts import paths as ps_paths

#: `<end>_dur_<seconds>_..._class_cluster_<name>_conf_<confidence>.jpg`
_SCENE_NAME = re.compile(
    r"^(?P<end>\d+\.\d+)_dur_(?P<dur>\d+\.\d+)_.*_class_cluster_(?P<cls>[a-z]+)_conf_(?P<conf>\d\.\d+)"
)

#: The rally view. The other clusters are interludes, replays and crowd shots.
POINT_CLASS = "point"


@dataclass(frozen=True)
class Scene:
    """One scene of a source video, located in time."""

    index: int
    start: float
    end: float
    cls: str
    confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def sample_time(self, fraction: float = 0.5) -> float:
        """A timestamp inside the scene, away from its cut boundaries."""
        return self.start + self.duration * fraction


def dataset_root(video: str) -> Path:
    return ps_paths.assets() / "dataset" / video


def source_video(video: str) -> Path:
    return ps_paths.assets() / "raw_4k" / f"{video}.mp4"


def list_scenes(video: str, *, cls: str | None = POINT_CLASS, min_duration: float = 2.0) -> list[Scene]:
    """Scene boundaries recovered from the decomposition's thumbnail names.

    Scenes shorter than ``min_duration`` are dropped: a sub-second scene is
    usually a cut artefact, and its single frame is as likely to be a
    transition as a background.
    """
    directory = dataset_root(video) / "scenes"
    if not directory.is_dir():
        raise FileNotFoundError(f"no scene decomposition for {video!r} at {directory}")
    scenes: list[Scene] = []
    previous_end = 0.0
    for index, path in enumerate(sorted(directory.glob("*.jpg"))):
        match = _SCENE_NAME.match(path.name)
        if match is None:
            raise ValueError(f"unparsable scene name, refusing to guess: {path.name}")
        end = float(match.group("end"))
        start = max(previous_end, end - float(match.group("dur")))
        previous_end = end
        scene = Scene(
            index=index,
            start=start,
            end=end,
            cls=match.group("cls"),
            confidence=float(match.group("conf")),
        )
        if scene.duration < min_duration:
            continue
        if cls is not None and scene.cls != cls:
            continue
        scenes.append(scene)
    return scenes


def plate_cache(video: str, height: int) -> Path:
    return ps_paths.outputs() / "bp30-background" / "plates" / f"{video}_h{height}"


def extract_plates(
    video: str,
    scenes: list[Scene],
    *,
    height: int = 2160,
    fraction: float = 0.5,
) -> list[Path]:
    """One PNG per scene, cached. Returns the paths in scene order.

    ``height`` scales the 4K source. Native is the honest resolution and is the
    default; a smaller one is for making a sweep affordable and must be reported
    with the result, because a rate measured at 540p is not a rate at 2160p.
    """
    ffmpeg = tools.resolve_ffmpeg()
    source = source_video(video)
    if not source.is_file():
        raise FileNotFoundError(f"no source video for {video!r} at {source}")
    cache = plate_cache(video, height)
    cache.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for scene in scenes:
        timestamp = scene.sample_time(fraction)
        destination = cache / f"scene_{scene.index:03d}_t{timestamp:08.3f}.png"
        if not destination.is_file():
            # `-ss` before `-i` seeks by keyframe and is fast; the frame it
            # lands on is inside the scene either way because the sample point
            # is mid-scene rather than at a cut.
            result = subprocess.run(
                [
                    ffmpeg.path, "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{timestamp:.3f}", "-i", str(source),
                    "-frames:v", "1", "-vf", f"scale=-2:{height}",
                    str(destination),
                ],
                capture_output=True,
            )
            if result.returncode != 0 or not destination.is_file():
                detail = (result.stderr or b"").decode("utf-8", "replace").strip()
                raise RuntimeError(f"could not extract {video} scene {scene.index}: {detail[:300]}")
        paths.append(destination)
    return paths


def load_plates(paths: list[Path]) -> list[np.ndarray]:
    """Read cached plates as RGB uint8, refusing a ragged set.

    Inter prediction needs a fixed frame size, so a mismatch is a hard error
    here rather than a surprise several encodes later (brief §2).
    """
    import cv2

    plates: list[np.ndarray] = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read plate {path}")
        plates.append(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    shapes = {p.shape for p in plates}
    if len(shapes) > 1:
        raise ValueError(f"plates have mixed shapes {shapes}; the stream needs one size")
    return plates


def extract_consecutive(
    video: str,
    scene: Scene,
    *,
    count: int = 2,
    height: int = 2160,
    fraction: float = 0.4,
) -> list[Path]:
    """``count`` *adjacent* frames from inside one scene, cached.

    This is the control's material and it has to be genuinely consecutive.
    Sampling two frames a fraction of a scene apart is a different and much
    weaker test: findings §18/§19 put truly adjacent frames at 1.2-3.3%, and
    frames a second apart in a rally come back several times that -- which
    would widen the control until it could no longer catch the broken encoder
    configuration §19 records it catching.
    """
    ffmpeg = tools.resolve_ffmpeg()
    source = source_video(video)
    if not source.is_file():
        raise FileNotFoundError(f"no source video for {video!r} at {source}")
    cache = plate_cache(video, height) / f"consecutive_{scene.index:03d}"
    cache.mkdir(parents=True, exist_ok=True)
    paths = [cache / f"frame_{i:02d}.png" for i in range(count)]
    if not all(p.is_file() for p in paths):
        timestamp = scene.sample_time(fraction)
        result = subprocess.run(
            [
                ffmpeg.path, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{timestamp:.3f}", "-i", str(source),
                "-frames:v", str(count), "-vf", f"scale=-2:{height}",
                str(cache / "frame_%02d.png"),
            ],
            capture_output=True,
        )
        # ffmpeg's image2 muxer numbers from 1, so rename onto the 0-based names
        # this function promises rather than leaving two conventions around.
        for index in range(count):
            produced = cache / f"frame_{index + 1:02d}.png"
            if produced.is_file() and produced != paths[index]:
                produced.replace(paths[index])
        if not all(p.is_file() for p in paths):
            detail = (result.stderr or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(
                f"could not extract {count} consecutive frames from {video} "
                f"scene {scene.index}: {detail[:300]}"
            )
    return paths
