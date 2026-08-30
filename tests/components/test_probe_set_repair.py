"""Required behaviour for the probe-set link repair.

The repair rewrites thousands of symlinks in place, so the properties that
matter are about what it *refuses* to do as much as what it fixes. A repair
that invents a plausible-looking target is worse than the dangling link it
replaced: a dangling link fails loudly, a wrong one silently feeds the wrong
frames into a probe.

**Deliberately not tested here:** the real 3,033-link view under `assets/`,
which is data rather than code and is covered by
`test_probe_set.py::TestAgainstTheRealTrees`; and NFS behaviour, which no unit
test can reproduce.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from experiments.probe_set.repair_links import repair


def _view(tmp_path: Path) -> tuple[Path, Path]:
    """A miniature data root: a dataset, and a view linking into it absolutely.

    The links are written the way the pre-2026-08-29 builder wrote them —
    absolute, pointing at a location that no longer exists — because that is
    the state the repair exists to fix.
    """
    data_root = tmp_path / "data"
    dataset = data_root / "assets" / "dataset" / "vid" / "segmentations" / "scene_000" / "track_0001"
    dataset.mkdir(parents=True)
    for index in range(3):
        (dataset / f"frame_{index:06d}.png").write_bytes(b"x")

    view = data_root / "assets" / "probe_set" / "clips" / "vid" / "scene_000" / "track_0001"
    view.mkdir(parents=True)
    stale = Path("/nowhere/checkout/assets/dataset/vid/segmentations/scene_000/track_0001")
    for index in range(3):
        (view / f"frame_{index:06d}.png").symlink_to(stale / f"frame_{index:06d}.png")
    return data_root, data_root / "assets" / "probe_set"


class TestRepair:
    def test_it_repairs_every_dangling_link(self, tmp_path: Path) -> None:
        data_root, view = _view(tmp_path)
        report = repair(root=view, data_root=data_root)
        assert report.scanned == 3
        assert report.repaired == 3
        assert report.unresolved == []
        assert report.ok
        for link in sorted(view.rglob("*")):
            assert link.exists(), f"{link} still dangles"

    def test_repaired_links_are_relative_so_a_later_move_cannot_break_them(
        self, tmp_path: Path
    ) -> None:
        """The point of the fix, not a stylistic preference.

        An absolute link records where the data sat; a relative one records how
        the view and the dataset sit relative to each other, which is what
        survives the pair being moved together.
        """
        data_root, view = _view(tmp_path)
        repair(root=view, data_root=data_root)
        for link in sorted(p for p in view.rglob("*") if p.is_symlink()):
            assert not os.readlink(link).startswith("/"), f"{link} was repaired to an absolute path"

        moved = tmp_path / "moved"
        (tmp_path / "data").rename(moved)
        for link in sorted(p for p in (moved / "assets" / "probe_set").rglob("*") if p.is_symlink()):
            assert link.exists(), "a repaired link did not survive moving the data root"

    def test_a_dry_run_changes_nothing(self, tmp_path: Path) -> None:
        data_root, view = _view(tmp_path)
        before = {p: os.readlink(p) for p in view.rglob("*") if p.is_symlink()}
        report = repair(root=view, data_root=data_root, apply=False)
        assert report.repaired == 3
        assert {p: os.readlink(p) for p in view.rglob("*") if p.is_symlink()} == before

    def test_an_already_valid_link_is_left_alone(self, tmp_path: Path) -> None:
        data_root, view = _view(tmp_path)
        repair(root=view, data_root=data_root)
        second = repair(root=view, data_root=data_root)
        assert second.already_valid == 3
        assert second.repaired == 0

    def test_a_target_that_does_not_exist_is_reported_not_invented(self, tmp_path: Path) -> None:
        """The refusal that matters. A confidently wrong link beats no link only
        for a tool that is measuring nothing."""
        data_root, view = _view(tmp_path)
        orphan = view / "clips" / "vid" / "scene_000" / "track_0001" / "frame_000099.png"
        orphan.symlink_to("/nowhere/checkout/assets/dataset/vid/segmentations/scene_000/track_0001/frame_000099.png")
        report = repair(root=view, data_root=data_root)
        assert report.repaired == 3
        assert len(report.unresolved) == 1
        assert "frame_000099" in report.unresolved[0]
        assert not report.ok
        assert orphan.is_symlink() and not orphan.exists(), "an unresolvable link was rewritten"

    def test_a_link_with_no_assets_component_is_not_guessed_at(self, tmp_path: Path) -> None:
        data_root, view = _view(tmp_path)
        odd = view / "clips" / "vid" / "scene_000" / "track_0001" / "elsewhere.png"
        odd.symlink_to("/somewhere/entirely/other/elsewhere.png")
        report = repair(root=view, data_root=data_root)
        assert len(report.unresolved) == 1
        assert "elsewhere" in report.unresolved[0]

    def test_a_missing_view_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no probe-set view"):
            repair(root=tmp_path / "absent")


class TestManifestPathAnchoring:
    """A relative manifest path must resolve against the data root, not the cwd.

    This is the half of the 2026-08-29 breakage that repairing symlinks does
    *not* fix. The manifest deliberately records `assets/dataset/...` rather
    than a machine path — a probe set that hardcodes one checkout is not
    portable — but `Path.resolve()` anchors a relative path at the current
    working directory. That was invisibly correct while `assets/` sat in the
    checkout and everything ran from the repo root, and became wrong the moment
    the data moved, reporting every frame as "not under the named track" while
    naming the track correctly.
    """

    def test_a_relative_manifest_path_follows_the_data_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.contracts import paths as ps_paths
        from experiments.probe_set.verify import _anchor

        monkeypatch.setenv(ps_paths.ENV_DATA_ROOT, str(tmp_path / "elsewhere"))
        anchored = _anchor(Path("assets/dataset/vid/segmentations/scene_000/track_0001"))
        assert anchored == tmp_path / "elsewhere" / "assets/dataset/vid/segmentations/scene_000/track_0001"

    def test_it_does_not_depend_on_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The actual defect: the answer moved when the process did."""
        from src.contracts import paths as ps_paths
        from experiments.probe_set.verify import _anchor

        monkeypatch.setenv(ps_paths.ENV_DATA_ROOT, str(tmp_path / "data"))
        relative = Path("assets/dataset/vid")
        from_here = _anchor(relative)
        monkeypatch.chdir(tmp_path)
        assert _anchor(relative) == from_here

    def test_an_absolute_manifest_path_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.contracts import paths as ps_paths
        from experiments.probe_set.verify import _anchor

        monkeypatch.setenv(ps_paths.ENV_DATA_ROOT, str(tmp_path / "data"))
        absolute = tmp_path / "somewhere" / "assets" / "dataset"
        assert _anchor(absolute) == absolute
