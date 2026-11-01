import pytest

from experiments.tennis.w7_tier2_billing import bill


def test_billing_refuses_changed_inputs_before_loading_checkpoint(tmp_path):
    archive, guider = tmp_path / 'fit.tar', tmp_path / 'guider.pth'
    archive.write_bytes(b'fit')
    guider.write_bytes(b'guider')
    with pytest.raises(ValueError, match='sha256 mismatch'):
        bill(archive, 'fit.pt', guider, '0' * 64, '0' * 64)
