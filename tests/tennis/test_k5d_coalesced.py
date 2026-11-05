import argparse
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from experiments.tennis import k5b_trackers as trk, k5d_coalesced as co, windows as tw
from src.codecs import quality


def test_scores_all_points_with_one_shared_context_and_input_checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / 'checkpoint'
    root.mkdir()
    points = [{'codec': codec, 'point': point, 'encoder_decoded_sha256': 'yuv',
               'encoder_rendered_sha256': 'rgb', 'wire_sha256': 'wire',
               'baseline': {}, 'streak': {'guard_off': False}} for codec, point in co.enc.POINTS]
    report = {'clip': 'tracknet/game9/Clip5', 'frames': 1, 'size': [1, 1],
              'resource_map_sha256': 'resources', 'points': points}
    source = {'device': {'capability': [8, 6]}, 'rows': [{'label_kind': 'visible', 'label_x': 1,
              'label_y': 2, 'member': 'frames/frame.jpg'}]}
    (root / 'pilot.json').write_text(json.dumps(report))
    (root / 'source-qualification.json').write_text(json.dumps(source))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('PS_CPU_ALLOWANCE', '1')
    monkeypatch.setattr(co.os, 'sched_getaffinity', lambda _: {0}, raising=False)
    torch = types.ModuleType('torch')
    torch.cuda = SimpleNamespace(device_count=lambda: 1, get_device_capability=lambda _: (8, 6))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(co.cp, 'verify', lambda *a: None)
    monkeypatch.setattr(co.oracle, 'validate_report', lambda *a: None)
    events: list[str] = []
    monkeypatch.setattr(co.cp, 'verify_score_inputs', lambda *a: events.append('verify'))
    clip = {'id': report['clip'], 'dir': 'frames', 'rows': [('frame.jpg',)]}
    monkeypatch.setattr(trk, 'tracknet_clips', lambda *a: [clip])
    monkeypatch.setattr(trk, 'tracknet_labels', lambda *a: [('visible', 1, 2)])
    images = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    def record(name: str, value: Any) -> Any:
        events.append(name)
        return value

    monkeypatch.setattr(trk, 'tracknet_frames', lambda *a: record('source', images))
    net = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type='cuda'))]))
    monkeypatch.setattr(trk, 'BallTrack', lambda *a: record('balltrack', SimpleNamespace(net=net)))
    monkeypatch.setattr(trk, 'Zoo', lambda *a, **k: record('wasb', SimpleNamespace(settings={'wasb-step1': {'net': net}})))
    monkeypatch.setattr(quality, 'load_lpips', lambda *a: record('lpips', object()))
    monkeypatch.setattr(quality, 'load_dists', lambda *a: record('dists', object()))
    monkeypatch.setattr(tw, 'rgb_frames', lambda *a: images)
    monkeypatch.setattr(co.np, 'memmap', lambda *a, **k: images)

    def replay(cmd: list[str], **kwargs: Any) -> None:
        work = Path(cmd[cmd.index('--work-dir') + 1])
        work.mkdir()
        (work / 'receiver.json').write_text(json.dumps({'decoded_yuv_sha256': 'yuv', 'rendered_rgb_sha256': 'rgb',
                                                        'wire_sha256': 'wire', 'resource_map_sha256': 'resources'}))
        events.append('replay')

    monkeypatch.setattr(co.subprocess, 'run', replay)
    contexts: list[tuple[int, int, int]] = []

    def score(image: Any, decoded: Any, rendered: Any, labels: Any, nets: Any, judges: Any, *rest: Any) -> Any:
        contexts.append((id(image), id(labels), id(nets)))
        events.append('score')
        return [], {}, [], {}, None

    monkeypatch.setattr(co.enc, 'score_arms', score)

    def assemble(checkpoint: Path, sha: str, receipts: list[dict[str, str]], output: Path, identity: dict[str, str]) -> dict[str, Any]:
        assert len(receipts) == 6
        assert all(Path(item['path']).exists() for item in receipts)
        events.append('assemble')
        return {'passed': True}

    monkeypatch.setattr(co.cp, 'assemble', assemble)
    args = argparse.Namespace(checkpoint=str(root), output=str(tmp_path / 'output'), archive='archive',
                              balltrack_code='bt-code', balltrack_weight='bt-weight', wasb_code='wasb-code',
                              wasb_weights='wasb-weights', lpips_backbone='lpips', dists_backbone='dists')
    co.score_all(args, 'checkpoint-sha', {'source_sha256': 'source'})
    assert len(contexts) == 6 and len(set(contexts)) == 1
    assert events.count('replay') == events.count('score') == 6
    assert all(events.count(name) == 1 for name in ('source', 'balltrack', 'wasb', 'lpips', 'dists', 'assemble'))
    assert events[0] == 'verify' and events[-2:] == ['verify', 'assemble']
