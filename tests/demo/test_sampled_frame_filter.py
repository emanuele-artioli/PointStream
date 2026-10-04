from pathlib import Path

from demo.models.sampled_seconds import CLIP3_DROP_SECONDS, sampled_frame_kept

CLIP3 = Path("clip_03_factory001_worker001_00000/f000000-f012629")


def test_clip3_drops_aisle_bin_and_blowout():
    assert CLIP3_DROP_SECONDS == frozenset({210, 240, 420})
    for name in ("006300.jpg", "006329.jpg", "007200.jpg", "007229.jpg", "012600.jpg", "012629.jpg"):
        assert sampled_frame_kept(CLIP3, name) is False


def test_clip3_keeps_the_press_including_the_opening_second():
    for name in ("000000.jpg", "000029.jpg", "000900.jpg", "011700.jpg"):
        assert sampled_frame_kept(CLIP3, name) is True


def test_other_recordings_are_unchanged():
    folder = Path("clip_01_factory001_worker001_00001/f000000-f035129")
    assert sampled_frame_kept(folder, "006300.jpg") is True
