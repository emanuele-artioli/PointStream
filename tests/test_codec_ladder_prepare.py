import json

import numpy as np
import pytest

from experiments.codec_ladder_prepare import selected_masks
from experiments.tennis.k6b_rate_streams import encode_masks
from src.segmentation.masks import ClipMasks, Instance, compress


def visor_blob():
    mask = np.zeros((4, 8), bool)
    mask[1:3, 2:5] = True
    clip = ClipMasks(('hand',), 4, 8, 50, labelled=[True] * 3)
    clip.ensure_frames(3)
    instance = Instance.from_mask('hand', 0, mask, provenance='interpolated')
    assert instance is not None
    clip.frames[1] = [instance]
    clip.meta.update(video_indices=[10, 11, 12], drift=[0, 0, 0], keyframes=[1])
    return clip


def test_visor_alignment_and_all_empty_frames_retained():
    clip = visor_blob()
    masks, record = selected_masks(compress(json.dumps(clip.to_doc()).encode()), kind='visor',
                                   first=0, frames=3, source_first=10, width=8, height=4)
    assert masks.shape == (3, 4, 8)
    assert masks.dtype == bool
    assert not masks[0].any() and masks[1].sum() == 6 and not masks[2].any()
    assert record['keyframes_selected'] == [1]


@pytest.mark.parametrize('change', ['indices', 'drift', 'labels'])
def test_visor_reject_unaligned_or_unlabelled(change):
    clip = visor_blob()
    if change == 'indices':
        clip.meta['video_indices'][1] = 99
    elif change == 'drift':
        clip.meta['drift'][1] = 1
    else:
        assert clip.labelled is not None
        clip.labelled[1] = False
    with pytest.raises(ValueError):
        selected_masks(compress(json.dumps(clip.to_doc()).encode()), kind='visor',
                       first=0, frames=3, source_first=10, width=8, height=4)


def test_tennis_union_and_nearest_resize():
    masks = np.zeros((3, 2, 2, 4), bool)
    masks[1, 0, 0, 1] = True
    masks[1, 1, 1, 2] = True
    output, record = selected_masks(encode_masks(masks, 'rle'), kind='tennis', first=1, frames=2,
                                   source_first=10, width=8, height=4)
    assert output.shape == (2, 4, 8)
    assert output[0].sum() == 8 and not output[1].any()
    assert 'not_independent_ground_truth' in record['provenance']


def test_validator_rejects_cost_forecast_tampering(tmp_path, monkeypatch):
    import hashlib
    import io
    import tarfile
    from experiments.codec_ladder_prepare import validate

    source = bytes(1920 * 1080 * 3 // 2)
    mask_file = io.BytesIO()
    np.savez_compressed(mask_file, foreground=np.zeros((1, 1080, 1920), bool))
    payloads = {'source.yuv': source, 'masks.npz': mask_file.getvalue()}
    record = {'frames': 1, 'no_inference_or_quality_selection': True,
              'outputs': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in payloads.items()}}
    payloads['preparation.json'] = json.dumps(record).encode()
    with tarfile.open(tmp_path / 'published.tar', 'w') as archive:
        for name, data in payloads.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    gate = {'measured_prepare_seconds': 1, 'measured_publication_seconds': 1,
            'measured_total_seconds': 2, 'actual_frames': 1, 'target_frames': 64,
            'forecast_full_seconds': 222, 'full_cap_seconds': 600, 'passed': True}
    path = tmp_path / 'preparation-result.json'
    path.write_text(json.dumps({**record, 'cost_gate': gate}))
    monkeypatch.setenv('PS_STAGE_DIR', str(tmp_path))
    monkeypatch.setenv('PS_VALIDATION_PATH', str(tmp_path / 'validation.json'))
    assert validate()
    gate['forecast_full_seconds'] = 0
    path.write_text(json.dumps({**record, 'cost_gate': gate}))
    assert not validate()
