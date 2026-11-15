import pytest

from tools.datasets.annotation_pilot import select


def rows():
    return [dict(id=f'{group}/{i}', match=group, frame=i) for group in 'abcde' for i in range(8)]


def test_hash_selection_is_order_independent_and_diverse():
    chosen = select(rows())
    assert [r['id'] for r in chosen] == [r['id'] for r in select(list(reversed(rows()))) ]
    assert len(chosen) == 12
    assert len({r['id'] for r in chosen}) == 12
    assert {r['match'] for r in chosen} == set('abcde')


def test_selection_refuses_undercoverage_and_duplicates():
    with pytest.raises(ValueError):
        select(rows()[:8])
    with pytest.raises(ValueError):
        select(rows() + [rows()[0]])
    with pytest.raises(ValueError):
        select(rows(), count=50)


def test_export_binds_archive_marks_and_preserves_blinding(tmp_path, monkeypatch):
    import base64
    import hashlib
    import io
    import json
    import tarfile

    import numpy as np
    from PIL import Image

    from src.segmentation.masks import ClipMasks, compress
    from tools.datasets import annotation_pilot as pilot

    def digest(data):
        return hashlib.sha256(data).hexdigest()
    def encoded(value):
        return json.dumps(value).encode()
    selection = {'sources': [dict(match=g, clip='clip', sha256=digest(g.encode()),
                                  converted_sha256=digest((g+'converted').encode())) for g in 'abc']}
    selection_bytes = encoded(selection)
    selection_sha = digest(selection_bytes)
    image = io.BytesIO()
    Image.new('RGB', (1920, 1080), (80, 100, 120)).save(image, format='JPEG')
    original = 'data:image/jpeg;base64,' + base64.b64encode(image.getvalue()).decode()
    clip = ClipMasks(['person', 'player'], 540, 960, 25)
    clip.add(0, 'person', 1, np.ones((540, 960), np.uint8))
    masks = compress(encoded(clip.to_doc()))
    frames, full, files = [], [], {}
    for i, (g, t) in enumerate((g, t) for g in 'abc' for t in range(4)):
        row = dict(id=f'{g}/clip/{t}', match=g, clip='clip', frame=t, fps='25',
                   masks=f'masks-{i:03d}.json.zst', masks_sha256=digest(masks), instances=[{'id': 1}])
        frames.append(row)
        full.append({**row, 'original': original})
        files[f'publish/{row["masks"]}'] = masks
        files[f'publish/frame-{i:03d}.json'] = encoded(full[-1])
    review = dict(selection_sha256=selection_sha, frames=frames)
    review_bytes = encoded(review)
    files['publish/instances.json'] = review_bytes
    marks = dict(schema=1, selection_sha256=selection_sha, result_sha256=digest(review_bytes),
                 marks={r['id']: dict(selected=[1], visible=1, missing_instance=False, confirmed=True) for r in frames})
    marks_bytes = encoded(marks)
    archive_path = tmp_path / 'frozen.tar'
    with tarfile.open(archive_path, 'w') as archive:
        for name, data in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    marks_path, selection_path = tmp_path / 'marks.json', tmp_path / 'selection.json'
    marks_path.write_bytes(marks_bytes)
    selection_path.write_bytes(selection_bytes)
    for constant, value in [('ARCHIVE_SHA', pilot.digest(archive_path)), ('MARKS_SHA', digest(marks_bytes)),
                            ('SELECTION_SHA', selection_sha), ('INSTANCES_SHA', digest(review_bytes))]:
        monkeypatch.setattr(pilot, constant, value)
    output = tmp_path / 'pilot'
    result = pilot.export(archive_path, marks_path, selection_path, output)
    assert result['count'] == 12 and result['blind_count'] == 3
    assert result['status'] == 'awaiting_human_review_no_new_reference_labels'
    prepared = json.loads((output / 'review/review.json').read_text())
    assert sum(r['blind'] for r in prepared['frames']) == 3
    assert all(r['overlay'] is None for r in prepared['frames'] if r['blind'])
    assert len(list((output / 'review').glob('source-*'))) == 12
    assert len(list((output / 'review').glob('overlay-*'))) == 9
    with pytest.raises(FileExistsError):
        pilot.export(archive_path, marks_path, selection_path, output)
    marks_path.write_bytes(marks_bytes + b' ')
    with pytest.raises(ValueError, match='SHA256'):
        pilot.export(archive_path, marks_path, selection_path, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
