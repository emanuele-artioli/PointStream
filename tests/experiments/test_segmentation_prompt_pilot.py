from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.segmentation.__main__ as cli
from experiments.segmentation import prompt_pilot
from src.segmentation import ClipMasks


class _FakeBackend:
    def __init__(self, name: str) -> None:
        self.name = name

    def segment(self, source: Path, domain, *, max_frames: int | None = None) -> ClipMasks:
        masks = ClipMasks(domain.classes, 16, 24, 10.0)
        hand = np.zeros((16, 24), dtype=bool)
        hand[4:10, 4 + len(self.name) % 3 : 12] = True
        for index in range(max_frames or 1):
            masks.add(index, domain.classes[-1], 1, hand)
        masks.meta.update({"backend": self.name, "timing": {"ms_per_frame": 1.0, "fps": 1000.0}})
        return masks


def _video(path: Path, frames: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (24, 16))
    for index in range(frames):
        writer.write(np.full((16, 24, 3), 10 * index, np.uint8))
    writer.release()


def test_pilot_runs_every_variant_and_passes_its_own_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "datasets"
    for clip in cli.load_domain("egocentric").clips[:1]:
        _video(root / clip, 6)
    for clip in cli.load_domain("egocentric").clips[1:]:
        (root / clip).parent.mkdir(parents=True, exist_ok=True)
        (root / clip).write_bytes(b"unused")
    monkeypatch.setenv("PS_DATASETS_ROOT", str(root))
    monkeypatch.setattr(cli, "build", _FakeBackend)
    out = tmp_path / "stage"
    monkeypatch.setenv("PS_STAGE_DIR", str(out))
    monkeypatch.setenv("PS_VALIDATION_PATH", str(tmp_path / "validation.json"))

    assert prompt_pilot.main(["--domain", "egocentric", "--frames", "4", "--clips", "1"]) == 0

    pilot = json.loads((out / "pilot.json").read_text())
    assert set(pilot["variants"]) == set(prompt_pilot.VARIANTS["egocentric"])
    assert pilot["variants"]["yoloe-26x_person_hand"]["prompts"] == {"arm": "person", "hand": "hand"}
    assert len(pilot["agreement"]["sam_arm_hand"]) == 5
    assert (out / "sheets" / "clip_01_factory001_worker001_00001.jpg").is_file()
    assert prompt_pilot.validate(["--validate", "--domain", "egocentric"]) == 0
