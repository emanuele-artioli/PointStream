"""Bounded scorer conformance experiment; synthetic inputs are not task truth."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand


def audit_reports(paths):
    rows, receipts = [], []
    def visit(value, source, path):
        if isinstance(value, dict):
            if {'gt_hands', 'matched_hands', 'pck50_matched', 'pck50_all_gt'} <= value.keys():
                denominator = value['gt_hands']
                expected = value['pck50_matched'] * value['matched_hands'] / denominator if denominator else 0
                rows.append({'source': source, 'json_path': path,
                    'reference_hand_instances': denominator, 'matched_hand_instances': value['matched_hands'],
                    'unmatched_reference_hand_instances': denominator - value['matched_hands'],
                    'pck50_matched': value['pck50_matched'], 'pck50_all_reference': value['pck50_all_gt'],
                    'all_reference_identity_error': abs(expected - value['pck50_all_gt']),
                    'registered_frame_count_present': 'registered_frames' in value})
            for key, item in value.items():
                visit(item, source, path + '/' + key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, source, path + '/' + str(i))
    for path in paths:
        data = path.read_bytes()
        receipts.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
        visit(json.loads(data), path.name, '')
    return rows, receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--legacy-revision', default='3f7c121')
    parser.add_argument('--report', action='append', default=[], type=Path)
    parser.add_argument('--track-root', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source_path = 'demo/evaluation/evaluate_robotics_teleop.py'
    old_source = subprocess.check_output(['git', 'show', f'{args.legacy_revision}:{source_path}'], text=True)
    namespace = {'__file__': str(Path(source_path).resolve()), '__name__': 'legacy_task_scorer'}
    exec(compile(old_source, source_path, 'exec'), namespace)
    old_score = namespace['score_pose_tracks']
    ref = []
    for i in range(3):
        hand = SingleHand('Right', .9, [0, 0, 20, 20], [[0, 0, 0]] * 21, [[10 + i * 500, 10]] * 21)
        ref.append(FrameHandPose(i, [hand]))
    interventions = {'complete': ref, 'missing_suffix': ref[:1], 'empty_prediction': [],
                     'reordered': list(reversed(ref)), 'sparse_last_frame': ref[2:]}
    results = {name: {'legacy': old_score(ref, pred), 'corrected': score_pose_tracks(ref, pred)}
               for name, pred in interventions.items()}
    rows, receipts = audit_reports(args.report)
    track_rows = []
    if args.track_root:
        clips = ['clip_01_factory001_worker001_00001', 'clip_02_factory001_worker001_00002',
                 'clip_03_factory001_worker001_00000']
        if args.smoke:
            clips = clips[:1]
        def load_track(path):
            data = path.read_bytes()
            receipts.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
            records = json.loads(data)
            return [FrameHandPose(row['frame_idx'], [SingleHand(**hand) for hand in row['hands']]) for row in records]
        for clip in clips:
            directory = args.track_root / clip
            reference = load_track(directory / 'ref_gt_rtm_hand.json')
            for name in ['ref_encoder_mp_live.json', 'ps_rec_ps_standard_rtm_hand.json',
                         'av1_960x540_p7_250k_reference_trimmed_rtm_hand.json']:
                pred = load_track(directory / name)
                ordered = sorted(pred, key=lambda pose: pose.frame_idx)
                conditions = {'full': pred, 'reversed': list(reversed(pred))}
                for fraction in [.25, .5, .75]:
                    conditions[f'drop_suffix_{fraction}'] = ordered[:int(len(ordered) * (1 - fraction))]
                for condition, selected in conditions.items():
                    track_rows.append({'clip_cache_identity': clip, 'prediction': name, 'condition': condition,
                        'reference_frame_ids': [pose.frame_idx for pose in reference],
                        'prediction_frame_ids': [pose.frame_idx for pose in selected],
                        'legacy': old_score(reference, selected), 'corrected': score_pose_tracks(reference, selected)})
    report = {'kind': 'synthetic scorer conformance plus saved model-reference arithmetic audit',
        'independent_task_accuracy': False, 'source_frame_alignment_verified': False,
        'training_exposure_verified': False,
        'legacy_revision': subprocess.check_output(['git', 'rev-parse', args.legacy_revision], text=True).strip(),
        'legacy_scorer_sha256': hashlib.sha256(old_source.encode()).hexdigest(),
        'corrected_scorer_sha256': hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        'audit_worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'command': sys.argv, 'python': sys.version, 'platform': platform.platform(),
        'gpu_uuid': None, 'native_encoder_decoder': 'not used', 'inputs': receipts,
        'synthetic_interventions': results, 'saved_model_reference_rows': rows,
        'retained_model_reference_interventions': track_rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.out), 'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      'saved_rows': len(rows)}))

if __name__ == '__main__':
    main()
