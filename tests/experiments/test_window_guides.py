from pathlib import Path
import numpy as np
import pytest
from experiments.gate_a_confirmation.window_guides import collect, rgb_to_bgr, raster_box, validate_source
from src.components.detection.geometry import Box
from src.components.detection.types import Detection

def test_color_and_source_shape():
    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    assert rgb_to_bgr(rgb).tolist() == [[[30, 20, 10]]]
    assert rgb.tolist() == [[[10, 20, 30]]]
    source = np.zeros((96, 2, 3, 3), dtype=np.uint8)
    assert validate_source(source, 12, full_shape=source.shape).shape[0] == 12
    with pytest.raises(ValueError):
        validate_source(source[:12], 12, full_shape=source.shape)
    with pytest.raises(ValueError):
        validate_source(source, 3, full_shape=source.shape)

def test_half_open_clipped_boxes():
    assert raster_box(Box(-.5, 1.2, 3.1, 5.7), 5, 4) == (0, 1, 4, 5)
    with pytest.raises(ValueError):
        raster_box(Box(8, 1, 9, 3), 5, 4)

def test_negative_frames_missing_masks_and_rgb_appearance(tmp_path):
    source = np.zeros((3, 3, 4, 3), dtype=np.uint8)
    source[..., 0] = 11
    source[..., 2] = 33
    class Detector:
        def __init__(self): self.index = 0
        def detect(self, frame):
            assert frame[0, 0].tolist() == [33, 0, 11]
            i = self.index; self.index += 1
            return [] if i == 1 else [Detection('person', Box(1, 1, 3, 3))]
    class Tracker:
        def reset(self): self.reset_called = True
        def update(self, frame, found, predictor):
            assert self.reset_called
            return list(found)
    class Segmenter:
        def __init__(self): self.index = 0
        def segment(self, frame, item):
            self.index += 1
            assert frame[0, 0].tolist() == [33, 0, 11]
            return None if self.index <= 2 else np.ones((2, 2), dtype=bool)
    objects, records = collect(source, Detector(), Tracker(), Segmenter(), tmp_path,
                               max_tracks=2, max_seconds=10)
    assert len(records) == 3 and records[1]['detected'] == []
    assert records[1]['tracked'][0]['history_held'] and records[1]['tracked'][0]['stale']
    assert records[1]['tracked'][0]['selected_raw_indices'] == []
    assert records[0]['tracked'][0]['mask_missing']
    mask = np.load(tmp_path / objects[0]['mask_file'])
    assert not mask[:2].any() and mask[2].sum() == 4
    appearance = np.load(tmp_path / objects[0]['appearance_file'])
    assert appearance[0, 0].tolist() == [11, 0, 33]

def test_bad_mask_shape_preserves_partial_frames(tmp_path):
    class Detector:
        def detect(self, frame): return [Detection('person', Box(0, 0, 2, 2))]
    class Tracker:
        def reset(self): pass
        def update(self, frame, found, predictor): return list(found)
    class Segmenter:
        def segment(self, frame, item): return np.ones((1, 1), dtype=bool)
    with pytest.raises(ValueError, match='shape'):
        collect(np.zeros((2, 3, 3, 3), dtype=np.uint8), Detector(), Tracker(), Segmenter(),
                tmp_path, max_tracks=1, max_seconds=10)
    assert (tmp_path / 'track_0000_mask.npy').exists()
    import json
    records=json.loads((tmp_path / 'frames.json').read_text())
    assert records[0]['status']=='processing' and records[1]['status']=='unprocessed'

def test_rss_guard_records_exceedance_and_model_calls():
    from experiments.gate_a_confirmation.window_guides import RssGuard, DeviceModel
    values = iter([12, 21])
    guard = RssGuard(20, reader=lambda: next(values))
    assert guard.check('before') == 12
    with pytest.raises(ValueError, match='RSS'):
        guard.check('after')
    assert guard.samples == 2 and guard.peak_sampled_bytes == 21
    class Model:
        def predict(self, **kwargs):
            assert kwargs['device'] == 'cpu'
            return 'ok'
    model = DeviceModel(Model(), 'cpu', RssGuard(100, reader=lambda: 10))
    assert model.predict(source=np.zeros((1, 1, 3), dtype=np.uint8), verbose=False) == 'ok'
    assert model.guard.samples == 2

def test_production_player_roles_exclusions_and_held_history(tmp_path):
    from src.components.tracking.tracker import IdentityTracker
    source=np.zeros((2,8,8,3),dtype=np.uint8)
    class Detector:
        def __init__(self):self.calls=0
        def detect(self,frame):
            self.calls+=1
            return [] if self.calls==2 else [Detection('person',Box(3,0,5,3)),
                Detection('person',Box(3,5,5,8)),Detection('person',Box(0,0,1,2)),
                Detection('tennis racket',Box(3,0,4,1))]
        def predict_roi(self,*args):raise AssertionError('two held selector slots need no tracker recovery')
    class Segmenter:
        def segment(self,frame,item):
            assert item.class_name=='player' and item.track_id in ('player_far','player_near')
            return np.ones((int(item.bbox.height),int(item.bbox.width)),dtype=bool)
    objects,records=collect(source,Detector(),IdentityTracker(),Segmenter(),tmp_path,max_tracks=8,max_seconds=10)
    assert len(records[0]['detected'])==4
    assert {d['raw_index'] for d in records[0]['excluded']}=={2,3}
    assert {o['object_id'] for o in objects}=={'player_far','player_near'}
    assert all(o['object_class']=='player' for o in objects)
    assert all(not row['history_held'] for row in records[0]['tracked'])
    assert records[1]['detected']==[] and len(records[1]['tracked'])==2
    assert all(row['history_held'] and row['stale'] and not row['detector_matched']
               and row['consecutive_held_frames']==1 for row in records[1]['tracked'])
    assert all(row['status']=='complete' for row in records)
