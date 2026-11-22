"""The boundary smoke must preserve the actual final-then-initial native frame order."""

import gzip
from hashlib import sha256

import pytest

from experiments.tennis import return_svt_continuous as control


@pytest.mark.parametrize("half", [48, 240])
def test_pair_order_and_integrity(tmp_path, monkeypatch, half):
    monkeypatch.setattr(control, "FRAME", 1)
    originals = [bytes(range(240)), bytes(reversed(range(240)))]
    inputs = [tmp_path / "first.gz", tmp_path / "second.gz"]
    hashes = [sha256(data).hexdigest() for data in originals]
    for path, data in zip(inputs, originals, strict=True):
        path.write_bytes(gzip.compress(data))
    output = tmp_path / "pair"
    control.write_input(output, inputs, hashes, half)
    assert output.read_bytes() == originals[0][-half:] + originals[1][:half]
    inputs[1].write_bytes(gzip.compress(originals[1] + b"x"))
    with pytest.raises(AssertionError):
        control.write_input(tmp_path / "bad", inputs, hashes, half)
