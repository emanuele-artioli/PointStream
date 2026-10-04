"""Source seconds that must not enter a training loader.

Clip 3 filenames are source frame indices at 30 fps, so second 210 is
006300.jpg through 006329.jpg. 0 s stays: the hand is on the press and the
parts bin is the task.
"""

from __future__ import annotations

from pathlib import Path

CLIP3_STEM = "clip_03_factory001_worker001_00000"
# 210 s is the aisle. 240 s leaves the press for the parts bin.
# 420 s is blown out, with a second person crossing the frame.
CLIP3_DROP_SECONDS = frozenset({210, 240, 420})


def sampled_frame_kept(folder: Path, frame_name: str) -> bool:
    if CLIP3_STEM not in Path(folder).as_posix():
        return True
    stem = Path(frame_name).stem
    if not stem.isdigit():
        return True
    return (int(stem) // 30) not in CLIP3_DROP_SECONDS
