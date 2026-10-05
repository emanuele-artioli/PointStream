from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.segmentation.__main__ as cli
from src.segmentation import ClipMasks
from src.segmentation.sources import sha256_file


class _FakeBackend:
    """Reference masks a square; the candidate shifts it, so J is below 1."""

    def __init__(self, name: str) -> None:
        self.name = name

    def segment(self, source: Path, domain, *, max_frames: int | None = None) -> ClipMasks:
        masks = ClipMasks(domain.classes, 20, 20, 25.0)
        shift = 0 if self.name == "sam31" else 2
        for index in range(max_frames or 3):
            mask = np.zeros((20, 20), dtype=bool)
            mask[5:12, 5 + shift : 12 + shift] = True
            masks.add(index, domain.classes[0], 1, mask, 0.9)
        masks.meta.update({"backend": self.name, "timing": {"ms_per_frame": 4.0, "fps": 250.0}})
        return masks


def test_suite_runs_every_backend_then_benchmarks_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "build", _FakeBackend)
    monkeypatch.setattr(
        "src.segmentation.sources.gpu_identity", lambda: {"uuid": "GPU-test", "name": "Fake"}
    )
    clip = tmp_path / "match" / "000.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"not really a video")
    out = tmp_path / "suite"

    code = cli.main(
        [
            "suite",
            "--domain",
            "tennis",
            "--backends",
            "sam31,yoloe-26n",
            "--source",
            str(clip),
            "--out",
            str(out),
            "--max-frames",
            "3",
        ]
    )

    assert code == 0
    run = out / "tennis" / "yoloe-26n" / "match_000"
    provenance = json.loads((run / "provenance.json").read_text())
    assert provenance["source"]["sha256"] == sha256_file(clip)
    assert provenance["frames"] == 3 and provenance["domain"] == "tennis"
    report = json.loads((out / "tennis" / "report.json").read_text())
    [row] = report["summary"]
    assert row["backend"] == "yoloe-26n" and 0 < row["J"] < 1 and row["ms_per_frame"] == 4.0

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": [{"path": str(clip), "sha256": sha256_file(clip)}]}))
    monkeypatch.setenv("PS_VALIDATION_PATH", str(tmp_path / "validation.json"))
    assert cli.main(["validate", "--root", str(out), "--manifest", str(manifest)]) == 0
    validation = json.loads((tmp_path / "validation.json").read_text())
    assert validation["passed"] and len(validation["checks"]) >= 8

    clip.write_bytes(b"changed")  # a changed input must fail the gate
    manifest.write_text(json.dumps({"clips": [{"path": str(clip), "sha256": sha256_file(clip)}]}))
    assert cli.main(["validate", "--root", str(out), "--manifest", str(manifest)]) == 1


def test_scene_numbered_clips_keep_their_match_name() -> None:
    assert cli.clip_id(Path("tennis_games/alcaraz_hurkacz/000.mp4")) == "alcaraz_hurkacz_000"
    assert cli.clip_id(Path("curated/clip_01_factory001.mp4")) == "clip_01_factory001"


def test_sheet_tiles_runs_side_by_side(tmp_path: Path) -> None:
    import cv2

    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(3):
        cv2.imwrite(str(frames / f"{index:03d}.png"), np.full((20, 40, 3), 30 * index, np.uint8))
    runs = []
    for name in ("sam31", "yoloe-26n"):
        masks = _FakeBackend(name).segment(frames, cli.load_domain("tennis"), max_frames=3)
        masks.width = 40  # the fake backend draws 20x20; the frames are 20x40
        masks.frames = [[] for _ in range(3)]
        runs.append(masks.save(tmp_path / name).parent)
    out = tmp_path / "sheet.png"
    assert (
        cli.main(
            [
                "sheet",
                "--source",
                str(frames),
                "--runs",
                *map(str, runs),
                "--frames",
                "0,2",
                "--width",
                "80",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert cv2.imread(str(out)).shape == (2 * 40, 2 * 80, 3)
