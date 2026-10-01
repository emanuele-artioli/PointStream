import hashlib
import json
import numpy as np
import pytest
from PIL import Image
from experiments.generative import gvcrt_prepare_rgb as worker

def test_complete_pixel_and_manifest_parity(tmp_path,monkeypatch):
    rgb=np.arange(48*360*640*3,dtype=np.uint8).reshape(48,360,640,3)
    src=tmp_path/'source.npy';np.save(src,rgb)
    monkeypatch.setattr(worker,'FILE_SHA',worker.digest(src));monkeypatch.setattr(worker,'RGB_SHA',hashlib.sha256(rgb.tobytes()).hexdigest())
    receipt=worker.prepare(src,tmp_path/'result');frames=json.loads((tmp_path/'result/frames.json').read_text())
    assert len(frames)==48 and receipt['png_pixel_parity']
    assert [f['source_frame_index'] for f in frames]==list(range(48))
    with Image.open(frames[47]['path']) as image:assert image.tobytes()==rgb[47].tobytes()

def test_identity_failure_creates_no_output(tmp_path):
    src=tmp_path/'wrong.npy';np.save(src,np.zeros((1,1,1,3),dtype=np.uint8))
    with pytest.raises(ValueError,match='file identity'):worker.prepare(src,tmp_path/'result')
    assert not (tmp_path/'result').exists()
