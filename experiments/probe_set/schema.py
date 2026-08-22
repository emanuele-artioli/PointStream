"""Probe-set schema constants and the locked train / held-out split.

``pointstream.probe_set.v1`` wrote a clip list independently of the view it
materialised, and never named its coordinate system. That schema is not
trustworthy and is never emitted here. Everything this package writes is
``pointstream.probe_set.v2``.
"""

from __future__ import annotations

SCHEMA_ID = "pointstream.probe_set.v2"
LEGACY_SCHEMA_ID = "pointstream.probe_set.v1"

COORDINATE_SYSTEM = "track_local"

DEFAULT_SEED = 20260712
DEFAULT_NUM_CLIPS = 12
DEFAULT_CLIP_LEN_FRAMES = 48
DEFAULT_MIN_FRAMES = 8

SELECTION_RULE = (
    "round-robin across training-split videos; one contiguous window per track, "
    "taken from the track's sorted source-filename indices, starting at a seeded "
    "random offset. Eligibility: primary track directory with a sibling "
    "_skeleton directory and at least min_frames colour frames."
)

# Locked 2026-07-11. The held-out pair is what keeps eval-general honest;
# sampling probe clips from them would leak the test videos into triage.
TRAINING_SPLIT_VIDEOS: tuple[str, ...] = (
    "alcaraz_perricard",
    "alcaraz_ruud",
    "djokovic_federer",
    "federer_djokovic",
    "sinner_alcaraz",
)
HELD_OUT_VIDEOS: tuple[str, ...] = (
    "alcaraz_highlights",
    "djokovic_zverev",
)

CLIPS_VIEW_NAME = "clips"
TRAINING_VIEW_NAME = "training_view"

CONDITION_DIR_SUFFIXES: tuple[str, ...] = (
    "_canny",
    "_skeleton",
    "_pose_body",
    "_pose_racket",
)
SIDECAR_SUFFIXES: tuple[str, ...] = (
    "_caption.json",
    "_keypoints.json",
    "_metadata.json",
)


class ProbeSetError(Exception):
    """One or more probe-set invariants failed."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        listing = "\n".join(f"  - {item}" for item in self.violations)
        super().__init__(f"{len(self.violations)} probe-set violation(s):\n{listing}")
