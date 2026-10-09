from __future__ import annotations

import json
from pathlib import Path

from experiments.background.g1 import restore_clips, save_clip


def test_finished_clips_survive_a_stop_and_come_back_on_resume(tmp_path: Path) -> None:
    publish = tmp_path / "publish"
    clip_dir = publish / "clips" / "ott__test_2"
    clip_dir.mkdir(parents=True)
    (clip_dir / "result.json").write_text("{}")
    (clip_dir / "masks.rle").write_bytes(b"masks")
    checkpoints = tmp_path / "checkpoint"
    save_clip(checkpoints, clip_dir, "ott/test_2", {"id": "ott/test_2", "lens": {"f": 1.0}}, {"frames": 3})
    (checkpoints / "clips" / ".unfinished.tmp").mkdir()  # an interrupted copy is ignored

    resumed = tmp_path / "attempt2" / "publish"
    restored = restore_clips(checkpoints, resumed)

    assert list(restored) == ["ott/test_2"]
    assert restored["ott/test_2"]["result"]["lens"] == {"f": 1.0}
    assert restored["ott/test_2"]["sam"] == {"frames": 3}
    copied = resumed / "clips" / "ott__test_2"
    assert (copied / "masks.rle").read_bytes() == b"masks"
    assert not (copied / "checkpoint.json").exists()
    assert json.loads((copied / "result.json").read_text()) == {}
