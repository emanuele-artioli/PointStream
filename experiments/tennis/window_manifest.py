"""One generic window manifest for the hybrid pipeline (W7; docs/experiments/w7-untouched.md).

Until W7 every stage pinned the two development windows Al 031@129 and Med 014@194 by constants (k6b_native_base.SOURCES,
k6b_rgb_diagnostic.SOURCES, w1_roi.AL/MED, w3_neural_baseline.WINDOWS, k6b_foreground_video.PILOTS). A `WindowSpec` carries
what a stage needs to know about a window; `FROZEN_PAIR` reproduces the old constants exactly, so a stage called with its old
arguments behaves as before (tests/test_window_manifest.py). Importing this module has no side effect.

Window id convention (K4/K6): ``<match>/<clip>@<start + 120>``, i.e. the centre frame of the 240-frame window.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

FRAMES = 240
SHA = re.compile(r"^[0-9a-f]{64}$")

#: Per-match adapters (rank 4, 8-bit fit job; the 4-bit step-5 coding is applied at generation) on the one tennis base.
#: Hashes are of the jobs' published.tar (docs/experiments/w7-untouched.md, 2026-10-30 lineage entry).
BASE = {"job": "20261023T194934Z-b0dcbb19", "published_sha256": "075b001a3ae9dcc9fdda9fb81e4240aaf3a9da228d707ac58733db094d983285",
        "weights": "AnimateAnyone original, tennis-base@fit-100.pt"}
ADAPTERS: dict[str, dict[str, str]] = {
    "alcaraz_hurkacz": {"key": "al", "job": "20261023T221650Z-3d7cbe2f", "name": "ah-C",
                        "published_sha256": "f1a488f2a3ddecf57cff451801f64bc07ae652c2b8b998d4ac844ba8685f5fa2"},
    "medvedev_struff": {"key": "med", "job": "20261023T204452Z-d5c90446", "name": "med-C",
                        "published_sha256": "5fc200fc50f5af68e561fa10baba54520043e5770555712caf6e89092b175427"},
}
#: How each window gets its conditioning crops (docs/experiments/w7-untouched.md, 2026-10-30 bank entry): every Al/Med window so
#: far took one crop per player from its own frame 0 (reference_frame == 0 in the window's packets), and the "12-image bank" is
#: the set of those crops over the match's six windows. `window_frame0` is that recipe applied to the new window.
REFERENCE_MODES = ("window_frame0",)


@dataclass(frozen=True)
class WindowSpec:
    name: str  # short key: file names and `--window`
    match: str
    clip: str
    start: int  # first native frame of the 240-frame window inside the clip
    frames: int
    fps: int
    source_sha256: str  # of the clip file
    source_yuv_sha256: str | None  # of K4's conversion of the window (k4b_features.convert_command), None until computed
    reference: str = "window_frame0"
    tier: str = "dev"

    @property
    def id(self) -> str:
        return f"{self.match}/{self.clip}@{self.start + self.frames // 2}"

    @property
    def clip_id(self) -> str:
        return f"{self.match}/{self.clip}"

    @property
    def safe_id(self) -> str:
        return self.id.replace("/", "__")

    @property
    def adapter(self) -> dict[str, str]:
        return ADAPTERS[self.match]

    @property
    def source_file(self) -> str:
        return f"{self.match}__{self.clip}.mp4"


#: The two development windows exactly as k6b_native_base.SOURCES / w1_roi spelled them.
FROZEN_PAIR = (
    WindowSpec("al", "alcaraz_hurkacz", "031", 9, FRAMES, 25, "054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76",
               "e7b960cd2456b439c5439b337a45e47d2a0a44a2a32025a2de595af3feed4fa2", tier="frozen"),
    WindowSpec("med", "medvedev_struff", "014", 74, FRAMES, 50, "a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be",
               "53633ce617b12d779bc8c5746d8d44cbe090ced6c49ba79dbe246cc6abacfb08", tier="frozen"),
)


def legacy_sources(windows: tuple[WindowSpec, ...] | list[WindowSpec] = FROZEN_PAIR) -> list[tuple[str, int, int, str, str]]:
    """The `SOURCES` tuples of k6b_native_base / k6b_rgb_diagnostic: (name, start, fps, source sha, yuv sha)."""
    out = []
    for w in windows:
        if w.source_yuv_sha256 is None:
            raise ValueError(f"{w.name}: conversion hash not computed")
        out.append((w.name, w.start, w.fps, w.source_sha256, w.source_yuv_sha256))
    return out


def check(windows: list[WindowSpec]) -> None:
    """Reject a malformed or self-overlapping manifest before any stage runs."""
    if not windows:
        raise ValueError("empty manifest")
    names = [w.name for w in windows]
    ids = [w.id for w in windows]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("duplicate window")
    for w in windows:
        if w.frames != FRAMES or w.start < 0 or w.match not in ADAPTERS or w.reference not in REFERENCE_MODES:
            raise ValueError(f"{w.name}: bad frames, start, match or reference mode")
        if not SHA.match(w.source_sha256) or (w.source_yuv_sha256 is not None and not SHA.match(w.source_yuv_sha256)):
            raise ValueError(f"{w.name}: bad hash")
        if not re.match(r"^[a-z0-9]+$", w.name):
            raise ValueError(f"{w.name}: name must be [a-z0-9]+ (it is used in file names)")
    by_clip: dict[str, list[WindowSpec]] = {}
    for w in windows:
        by_clip.setdefault(w.clip_id, []).append(w)
    for clip, group in by_clip.items():
        group.sort(key=lambda w: w.start)
        for a, b in zip(group, group[1:]):
            if b.start < a.start + a.frames:
                raise ValueError(f"windows overlap in {clip}")


def dump(windows: list[WindowSpec], path: Path, **extra: Any) -> None:
    check(windows)
    path.write_text(json.dumps({"schema": 1, "base": BASE, "adapters": ADAPTERS, **extra, "windows": [asdict(w) for w in windows]},
                               indent=1))


def load(path: Path) -> list[WindowSpec]:
    data = json.loads(path.read_text())
    if data.get("schema") != 1:
        raise ValueError("unknown window manifest schema")
    windows = [WindowSpec(**w) for w in data["windows"]]
    check(windows)
    return windows


def from_selection(selection: dict[str, Any], yuv: dict[str, str] | None = None) -> list[WindowSpec]:
    """The Tier 1 windows of `w7_select select` (docs/experiments/w7-specs/selection.json), Al then Med in its order.

    Names are `al028`, `med000`...; `yuv` maps a clip id to the conversion hash once it has been computed."""
    out: list[WindowSpec] = []
    for match, m in selection["matches"].items():
        short = ADAPTERS[match]["key"]
        for s in m["selected"]:
            clip = s["clip"].split("/")[1]
            out.append(WindowSpec(f"{short}{clip}", match, clip, s["start"], FRAMES, int(s["fps"]), s["sha256"],
                                  (yuv or {}).get(s["clip"]), tier="w7-tier1"))
    check(out)
    return out


def select(windows: list[WindowSpec], names: list[str] | None) -> list[WindowSpec]:
    """`names` in manifest order; None keeps all."""
    if names is None:
        return list(windows)
    known = {w.name: w for w in windows}
    missing = [n for n in names if n not in known]
    if missing:
        raise ValueError(f"unknown windows: {missing}")
    return [known[n] for n in names]


def cohort_tuples(windows: list[WindowSpec]) -> list[list[Any]]:
    """The `cohort` field of a confirmation-style manifest (k6b_confirm.cohort_of): [clip id, start, fps, source sha256]."""
    return [[w.clip_id, w.start, w.fps, w.source_sha256] for w in windows]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(args: Any, frozen_paths: list[Path], frozen_sources: list[tuple[str, int, int, str, str]],
         ) -> tuple[list[Path], list[tuple[str, int, int, str, str]], list[str]]:
    """Source files, `SOURCES` tuples and window ids of a stage: the frozen Al/Med pair when `args.windows_file` is None
    (the stage's own `--al --med` and `SOURCES`), else the selected manifest windows with one `--source` each, in order."""
    if args.windows_file is None:
        return frozen_paths, frozen_sources, [w.id for w in FROZEN_PAIR]
    windows = select(load(args.windows_file), args.window or None)
    if len(args.source or []) != len(windows):
        raise ValueError("one --source per selected window, in order")
    return list(args.source), legacy_sources(windows), [w.id for w in windows]
