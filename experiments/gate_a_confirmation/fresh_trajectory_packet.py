"""Pure schema1 static-appearance temporal placement adaptation.

Caller verifies fresh sender/guide identities before use. No source/cache/model
access or decoding occurs here. This restores placement, not dynamic pose.
"""
import copy
import io
import json
import math
import numpy as np

VERSION = 'pointstream.fresh_static_trajectory.v1'
ROLES = frozenset(('player_far', 'player_near'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def box(value, width, height):
    require(isinstance(value, (list, tuple)) and len(value) == 4 and
            all(type(v) in (int, float) and math.isfinite(v) and int(v) == v for v in value),
            'tracked bbox must contain finite integral coordinates; no rounding')
    # Detection Box serializes raster coordinates as integral JSON floats.
    x1, y1, x2, y2 = map(int, value)
    require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
            'tracked bbox must be inside raster')
    return [x1, y1, x2, y2]


def adapt_packet(arrays, metadata, records, receipt, *, policy,
                 missing_policy='skip_recorded', held_policy='preserve_recorded'):
    """Return (new array mapping, metadata, coverage), leaving inputs unchanged.

    Original arrays are shared read-only by convention and never altered or
    removed. Alpha adds a crop-local initial mask template. Tracked boxes alone
    determine placement; mask bounds never infer geometry. Receipt/records must
    already be independently hash-verified by the caller.
    """
    require(policy in ('null', 'opaque', 'alpha'), 'explicit compositing policy required')
    require(missing_policy == 'skip_recorded', 'unsupported missing policy')
    require(held_policy in ('preserve_recorded', 'skip_recorded', 'reject'), 'unsupported held policy')
    require(metadata.get('schema') == 1 and 'temporal_adapter' not in metadata,
            'unadapted schema1 sender packet required')
    n, width, height = (metadata.get(k) for k in ('frame_count', 'width', 'height'))
    require(all(type(v) is int and v > 0 for v in (n, width, height)), 'positive integer packet dimensions')
    require(receipt.get('status') == 'complete' and type(receipt.get('frame_count')) is int and
            receipt['frame_count'] >= n, 'complete guide receipt covering packet required')
    require(receipt.get('source_frame_interval') == [38, 38 + receipt['frame_count']], 'source000 guide interval required')
    require(len(records) == n and all(r.get('status') == 'complete' and
            r.get('frame_index') == i and r.get('source_frame_index') == 38+i
            for i, r in enumerate(records)), 'ordered complete frame records required')
    guide_objects = receipt.get('objects', [])
    require(len({o['object_id'] for o in guide_objects}) == len(guide_objects) and
            all(o.get('object_class') == 'player' and o['object_id'] in ROLES for o in guide_objects),
            'unique explicit fresh guide player roles required')
    guides = {o['object_id']: o for o in guide_objects}
    require(all(type(o.get('frame_index')) is int and 0 <= o['frame_index'] < receipt['frame_count'] for o in guide_objects), 'guide initialization frame range')
    originals = metadata.get('placements')
    require(isinstance(originals, list) and originals, 'initial sender placements required')
    require(len({p['object_id'] for p in originals}) == len(originals), 'one initial placement per role required')
    initial = {}
    for p in originals:
        role = p['object_id']
        require(role in guides, 'sender role absent from fresh receipt')
        f = p.get('frame_index')
        require(type(f) is int and 0 <= f < n and f == guides[role]['frame_index'], 'appearance initialization frame mismatch')
        require(box(p['bbox'], width, height) == box(guides[role]['bbox'], width, height), 'initial bbox differs from fresh receipt')
        key = p.get('encoded_crop_key') or p.get('crop_key')
        require(key in arrays, 'shared appearance payload missing')
        initial[role] = copy.deepcopy(p)
        initial[role]['bbox'] = box(p['bbox'], width, height)
    require(set(initial) == {role for role, o in guides.items() if o['frame_index'] < n},
            'sender must carry every role initialized in this packet prefix')
    result = dict(arrays)
    meta = copy.deepcopy(metadata)
    rows, coverage, decisions = [], [0]*n, []
    template_keys = {}
    if policy == 'alpha':
        for role, p in initial.items():
            require(p.get('mask_key') in arrays, 'initial alpha source mask missing')
            m = np.asarray(arrays[p['mask_key']])
            f = p['frame_index']; x1,y1,x2,y2 = p['bbox']
            if m.ndim == 3:
                require(m.shape == (n,height,width), 'initial mask stack must match packet raster')
                template = m[f,y1:y2,x1:x2]
            elif m.ndim == 2 and m.shape == (height,width):
                template = m[y1:y2,x1:x2]
            elif m.ndim == 2 and m.shape == (y2-y1,x2-x1):
                template = m
            else:
                raise ValueError('unsupported initial mask geometry')
            require(template.size and np.isin(template, [0,1]).all() and template.any(), 'nonempty binary initial template required')
            key = 'fresh_static_alpha_' + role
            require(key not in result, 'alpha template key collision')
            result[key] = template.astype(np.uint8, copy=True)
            template_keys[role] = key
    for i, record in enumerate(records):
        tracked = record.get('tracked')
        require(isinstance(tracked, list) and len({r['track_id'] for r in tracked}) == len(tracked), 'unique tracked roles per frame required')
        by_role = {}
        for row in tracked:
            role = row['track_id']
            require(role in guides and row.get('class_name') == 'player', 'unexpected tracked role/class')
            geometry = box(row.get('bbox'), width, height)
            require(all(type(row.get(k)) is bool for k in ('mask_missing','history_held','tracker_history_held','stale')), 'explicit missing/held/stale flags required')
            held = row['history_held'] or row['tracker_history_held']
            require(not held or row['stale'], 'held role must be declared stale')
            require(not held or held_policy != 'reject', 'registered policy rejects held role')
            if role in initial and i == initial[role]['frame_index']:
                require(geometry == list(initial[role]['bbox']), 'tracked initialization bbox differs from sender')
            by_role[role] = (row, geometry, held)
        for role, p in initial.items():
            state = by_role.get(role)
            reason = None
            if i < p['frame_index']:
                reason = 'before_appearance_initialization'
            elif state is None:
                reason = 'role_absent'
            elif state[0]['mask_missing']:
                reason = 'mask_missing'
            elif state[2] and held_policy == 'skip_recorded':
                reason = 'held_role_skipped'
            elif policy == 'null':
                reason = 'foreground_disabled'
            if reason is None:
                item = copy.deepcopy(p)
                item.update(frame_index=i, bbox=state[1], mask_key=template_keys.get(role))
                rows.append(item); coverage[i] += 1
            decisions.append({'frame_index':i, 'object_id':role,
                              'placed':reason is None, 'reason':reason,
                              'held':state[2] if state else None,
                              'stale':state[0]['stale'] if state else None})
    meta['placements'] = rows
    meta['mask_policy'] = 'alpha' if policy == 'alpha' else 'opaque'
    meta['temporal_adapter'] = {'version':VERSION, 'policy':policy,
        'geometry':'recorded_integral_tracked_bbox_no_rounding_no_mask_bound_inference',
        'missing_policy':missing_policy, 'held_policy':held_policy,
        'appearance':'shared_initial_reference_static',
        'alpha':'initial_crop_local_static_template' if policy == 'alpha' else None,
        'required_receiver':'pinned_legacy_decoder_with_packet_mask_policy_wrapper',
        'required_crop_resize':'cv2.INTER_LINEAR',
        'required_alpha_resize':'cv2.INTER_LINEAR_to_tracked_bbox' if policy == 'alpha' else None,
        'all_original_arrays_retained':True, 'coverage_by_frame':coverage,
        'role_decisions':decisions,
        'scope':'Placement adaptation only; model-derived guides are not truth; static appearance/template does not restore true pose or dynamic appearance.'}
    # metadata is the sole replaced array. Every payload array is retained.
    result['metadata'] = np.frombuffer(json.dumps(meta,allow_nan=False).encode(),dtype=np.uint8)
    return result, meta, coverage


def encode_packet(arrays):
    """Charge the returned complete NPZ length; add external manifest separately."""
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()
