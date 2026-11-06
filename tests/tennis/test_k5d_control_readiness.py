from typing import Any

import pytest

from experiments.tennis.k5d_control_readiness import jpeg_size


def test_jpeg_dimensions_are_checked_without_pixel_decode() -> None:
    # SOI, APP0, SOF0: 720 x 1280; enough header for this parser.
    data = b'\xff\xd8\xff\xe0\x00\x04ab\xff\xc0\x00\x07\x08\x02\xd0\x05\x00'
    assert jpeg_size(data) == (1280, 720)
    with pytest.raises(ValueError, match='dimensions'):
        jpeg_size(b'\xff\xd8\xff\xd9')
    with pytest.raises(ValueError, match='start marker'):
        jpeg_size(b'not jpeg')


def test_readiness_checks_complete_registered_zip_and_headers(tmp_path, monkeypatch) -> None:
    import hashlib
    import json
    import zipfile
    from experiments.tennis import k5d_control_readiness as ready

    labels = b'Frame,Visibility,X,Y\na.jpg,1,100,120\nb.jpg,0,0,0\n'
    jpeg = b'\xff\xd8\xff\xc0\x00\x07\x08\x02\xd0\x05\x00'
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('Dataset/game9/Clip5/Label.csv', labels)
        z.writestr('Dataset/game9/Clip5/a.jpg', jpeg)
        z.writestr('Dataset/game9/Clip5/b.jpg', jpeg)
    clip: dict[str, Any] = {'clip_id': 'tracknet/game9/Clip5', 'label': 'Dataset/game9/Clip5/Label.csv', 'label_rows': 2,
            'jpg_entries': 2, 'visibility_counts': {'0': 1, '1': 1, '2': 0, '3': 0},
            'label_sha256': hashlib.sha256(labels).hexdigest(), 'roles': ['full_retrospective_benchmark']}
    manifest = tmp_path / 'selection.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'tracknet': {'clips': [clip]}}))
    monkeypatch.setattr(ready.fleet, 'SOURCE', manifest)
    monkeypatch.setattr(ready.fleet, 'SOURCE_SHA', ready.oracle.sha(manifest))
    monkeypatch.setattr(ready.fleet, 'EXPECTED', {**ready.fleet.EXPECTED, 'archive': ready.oracle.sha(archive)})
    assert ready.source(archive, clip['clip_id'], 2)['labels'] == 2
    with pytest.raises(ValueError, match='entire registered'):
        ready.source(archive, clip['clip_id'], 1)
    archive.write_bytes(archive.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='archive differs'):
        ready.source(archive, clip['clip_id'], 2)
