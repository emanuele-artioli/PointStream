"""Preregistered offline plate experiment; closed payload-only reconstruction.

The original prepared RGB and parent masks are encoder inputs, not receiver
assets or independent annotations. All native plate, appearance, mask, map,
geometry and container bytes reside in each charged package. No reset policy
is inferred. Fixed frame-zero canvas registration can clip camera margins.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import warnings
import zipfile

import cv2
import numpy as np

from src.runner.packet_packing import _archive, _json, _npy, pack_client_envelope, unpack_client_envelope
from src.runner.mask_wire import decode_mask
from src.runner.client import reconstruct_serialized_client

MANIFEST = 'registered_plate.json'
MAPS = 'registered_homographies.npy'
ARMS = (
    ('first_identity_q32', 'first', False, 'first', 32),
    ('median_identity_q32', 'median', False, 'first', 32),
    ('first_registered_q32', 'first', True, 'first', 32),
    ('median_registered_q32', 'median', True, 'first', 32),
    ('median_registered_q44', 'median', True, 'first', 44),
    ('median_registered_perframe_q32', 'median', True, 'perframe', 32),
    ('median_registered_reset12_q32', 'median', True, 'first', 32),
    ('median_registered_reset24_q32', 'median', True, 'first', 32),
)
REGISTRATION = {'arms': ARMS, 'plate_codec': 'av1', 'plate_rate_control': 'QP',
    'appearance': 'original parent residual-off bytes; no free appearance',
    'mask_scale': 1, 'mask_codec': 'rle', 'batch_masks': True,
    'registration': 'masked ORB frame-to-frame0, ratio0.75 RANSAC1px, identity fallback',
    'maps': 'one little-endian float32 3x3 per emitted frame; charged NPY bytes',
    'plate': 'frame0 canvas, masked temporal median, uncovered pixels fall back to original frame0',
    'source': 'prepared cached RGB; no native PTS or independent task truth',
    'quality': 'whole RGB and uint8 BT601 Y pooled PSNR; parent-mask FG/BG diagnostics only',
    'exposure': 'offline complete-window encoder access; one development window',
    'selection': 'eight fixed arms retained; no quality-driven plate/map selection',
    'resets': 'new complete coded plate every12/24 prepared frames; all repetitions and segment manifest charged',
    'compute': 'CPU only; representative full48 reset12 and identity smoke before eight-arm full48; root phase3 gate'}


def members(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('duplicate package member')
        return {key: z.read(key) for key in z.namelist()}


def metadata(data):
    return json.loads(np.load(io.BytesIO(data['metadata.npy']), allow_pickle=False).tobytes())


def parent_masks(payload, count):
    data = members(unpack_client_envelope(payload)); meta = metadata(data)
    if count > meta['frame_count'] or count < 1:
        raise ValueError('invalid selected frame count')
    masks = np.zeros((count, meta['height'], meta['width']), dtype=bool)
    for item in meta['placements']:
        t = item['frame_index']
        if t >= count: continue
        key = item.get('mask_key')
        if not key: raise ValueError('parent placement requires transmitted mask')
        mask = decode_mask(np.load(io.BytesIO(data[key + '.npy']), allow_pickle=False).tobytes())
        if mask.ndim == 3: mask = mask[t]
        if mask.shape != masks.shape[1:]: raise ValueError('parent masks must cover original frame raster')
        masks[t] |= mask != 0
    return masks


def register(frames, masks):
    """Encoder-only deterministic masked ORB maps, with explicit fallback."""
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    orb = cv2.ORB_create(nfeatures=2000)
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    valid = [np.asarray(~mask, dtype=np.uint8) * 255 for mask in masks]
    kp0, des0 = orb.detectAndCompute(gray[0], valid[0])
    maps = [np.eye(3, dtype=np.float32)]; receipts = [{'frame': 0, 'status': 'reference'}]
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    height, width = frames.shape[1:3]
    corners = np.float32([[0,0], [width,0], [width,height], [0,height]]).reshape(-1,1,2)
    for i in range(1, len(frames)):
        kp, des = orb.detectAndCompute(gray[i], valid[i])
        good = [] if des is None or des0 is None else [pair[0] for pair in matcher.knnMatch(des, des0, k=2)
                       if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance]
        h = None; inliers = None
        if len(good) >= 8:
            src = np.float32([kp[m.queryIdx].pt for m in good]); dst = np.float32([kp0[m.trainIdx].pt for m in good])
            h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 1.0, maxIters=5000, confidence=.999)
        accepted = h is not None and np.isfinite(h).all() and abs(h[2,2]) > 1e-8
        if accepted:
            h = h / h[2,2]
            projected = cv2.perspectiveTransform(corners, h)
            area = abs(cv2.contourArea(projected)) / (width * height)
            accepted = (int(inliers.sum()) >= 8 and np.linalg.cond(h) < 1e8 and
                        .25 <= area <= 4 and np.max(np.abs(projected)) <= 4 * max(width,height))
        maps.append(np.asarray(h if accepted else np.eye(3), dtype=np.float32))
        receipts.append({'frame':i, 'matches':len(good), 'inliers':int(inliers.sum()) if inliers is not None else 0,
                         'status':'registered' if accepted else 'identity_fallback'})
    return np.stack(maps), receipts


def build_plate(frames, masks, maps, kind):
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError('prepared frames must be uint8 THW3')
    if masks.shape != frames.shape[:3] or maps.shape != (len(frames),3,3):
        raise ValueError('plate frame/mask/map coverage mismatch')
    if kind == 'first': return frames[0].copy(), {'uncovered_pixels':0, 'construction':'original_frame0'}
    if kind != 'median': raise ValueError('unknown plate kind')
    height, width = frames.shape[1:3]; samples = []
    for frame, mask, h in zip(frames, masks, maps, strict=True):
        warped = cv2.warpPerspective(frame, h, (width,height), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(np.asarray(~mask.astype(bool),dtype=np.uint8), h, (width,height), flags=cv2.INTER_NEAREST) != 0
        sample = warped.astype(np.float32); sample[~valid] = np.nan; samples.append(sample)
    stack = np.stack(samples)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
    missing = ~np.isfinite(median[...,0]); median[missing] = frames[0][missing]
    return np.clip(np.rint(median),0,255).astype(np.uint8), {'uncovered_pixels':int(missing.sum()),
        'construction':'mask_excluded_temporal_median; uncovered_frame0_fallback'}


def encode_package(parent, plate_rgb, maps, *, qp=32, codec='av1', encode_plate=None, arm='unspecified'):
    """Replace background only; preserve every retained parent foreground byte."""
    data = members(unpack_client_envelope(parent)); meta = metadata(data)
    count, height, width = meta['frame_count'], meta['height'], meta['width']
    maps = np.asarray(maps,dtype='<f4')
    if maps.shape != (count,3,3) or not np.isfinite(maps).all() or any(abs(np.linalg.det(h)) < 1e-8 for h in maps):
        raise ValueError('invalid per-frame homography coverage')
    if plate_rgb.shape != (height,width,3) or height % 2 or width % 2:
        raise ValueError('plate must match even original frame raster')
    if (meta.get('residual') or {}).get('present') or any(key.startswith('residual_') for key in data):
        raise ValueError('registered plate arms require residual-off parent')
    if codec not in ('av1','vvc'): raise ValueError('unsupported native plate codec')
    if encode_plate is None:
        from src.components.background.sidecar import IntraCodecSidecar
        encode_plate = IntraCodecSidecar(codec,qp=qp).encode
    payload = encode_plate(np.ascontiguousarray(plate_rgb[...,::-1]))
    if not payload: raise ValueError('empty native plate payload')
    for key in list(data):
        if key.startswith(('background_', 'registered_')): del data[key]
    header = _json({'schema':1,'kind':'registered_plate','plate_codec':codec,'qp':qp,'frame_count':count,
                    'height':height,'width':width,'maps':MAPS,'map_encoding':'float32_le_3x3'})
    bg = dict(meta.get('background') or {})
    bg.update(plate_key=None,homographies=[],mode='full',deferred_to_residual=False,
        scene_id='registered-prepared-window',width=width,height=height,payload_bytes=len(payload),
        geometry_header=header.hex(),geometry_header_bytes=len(header),wire_payload_keys=['background_payload_0'],
        wire_header_keys=['background_header_0'],wire_codec=None,wire_codec_id=None,sidecar_codec=codec)
    meta['background'] = bg
    data['metadata.npy'] = _npy(np.frombuffer(_json(meta),dtype=np.uint8))
    data['background_payload_0.npy'] = _npy(np.frombuffer(payload,dtype=np.uint8))
    data['background_header_0.npy'] = _npy(np.frombuffer(header,dtype=np.uint8))
    # The client ordinary schema needs maps; temporarily add them to validate
    # packing, then remove JSON maps in favor of the charged binary map member.
    meta['background']['homographies'] = maps.reshape(count,9).tolist()
    data['metadata.npy'] = _npy(np.frombuffer(_json(meta),dtype=np.uint8))
    packed = members(pack_client_envelope(_archive(data),mask_codec='rle',batch_masks=True))
    meta['background']['homographies'] = []
    packed['metadata.npy'] = _npy(np.frombuffer(_json(meta),dtype=np.uint8))
    packed[MAPS] = _npy(maps)
    packed[MANIFEST] = _json({'format':'pointstream.registered-plate','version':1,'arm':arm,
        'frames':count,'height':height,'width':width,'maps':MAPS,'native_codec':codec,'qp':qp,
        'parent_sha256':hashlib.sha256(parent).hexdigest(),'residual':False,'rate_boundary':'complete archive bytes'})
    return _archive(packed)


def unpack_package(payload):
    data = members(payload)
    spec = json.loads(data.pop(MANIFEST))
    if spec.get('format') != 'pointstream.registered-plate' or spec.get('version') != 1 or spec.get('maps') != MAPS:
        raise ValueError('unsupported registered plate package')
    maps = np.load(io.BytesIO(data.pop(MAPS)),allow_pickle=False)
    meta = metadata(data)
    if maps.dtype != np.dtype('<f4') or maps.shape != (meta['frame_count'],3,3) or not np.isfinite(maps).all():
        raise ValueError('invalid charged maps')
    if any(spec.get(key) != meta[mkey] for key,mkey in [('frames','frame_count'),('height','height'),('width','width')]):
        raise ValueError('registered package frame/raster mismatch')
    if any(abs(np.linalg.det(h)) < 1e-8 for h in maps): raise ValueError('singular charged map')
    if meta['background']['homographies']: raise ValueError('unexpected redundant maps')
    meta['background']['homographies'] = maps.reshape(len(maps),9).tolist()
    data['metadata.npy'] = _npy(np.frombuffer(_json(meta),dtype=np.uint8))
    return unpack_client_envelope(_archive(data))



def subset_parent(parent, start, end):
    """Keep charged global appearance references, reindex selected placements."""
    data = members(unpack_client_envelope(parent)); meta = metadata(data)
    if not 0 <= start < end <= meta['frame_count']: raise ValueError('invalid parent span')
    selected = [dict(p, frame_index=p['frame_index']-start) for p in meta['placements']
                if start <= p['frame_index'] < end]
    mask_keys = {p['mask_key'] for p in selected if p.get('mask_key')}
    for key in list(data):
        if key.startswith('mask_') and key[:-4] not in mask_keys: del data[key]
    for key in mask_keys:
        arr = np.load(io.BytesIO(data[key+'.npy']),allow_pickle=False)
        mask = decode_mask(arr.tobytes())
        if mask.ndim == 3:
            from src.runner.mask_wire import encode_mask
            mask = mask[start:end]
            data[key+'.npy'] = _npy(np.frombuffer(encode_mask(mask),dtype=np.uint8))
            for p in selected:
                if p.get('mask_key') == key: p['mask_wire']['shape'] = list(mask.shape)
    meta['frame_count'] = end-start; meta['placements'] = selected
    if meta.get('background'):
        meta['background']['homographies'] = meta['background']['homographies'][start:end]
    data['metadata.npy'] = _npy(np.frombuffer(_json(meta),dtype=np.uint8))
    return _archive(data)


def encode_reset_bundle(parent, frames, masks, *, interval, qp=32, codec='av1', encode_plate=None):
    if interval not in (12,24): raise ValueError('unsupported registered reset interval')
    meta = metadata(members(unpack_client_envelope(parent)))
    if frames.shape != (meta['frame_count'],meta['height'],meta['width'],3) or masks.shape != frames.shape[:3]:
        raise ValueError('reset source coverage mismatch')
    data = {}; segments = []; prep = []
    for start in range(0,len(frames),interval):
        end = min(start+interval,len(frames)); maps, fit = register(frames[start:end],masks[start:end])
        plate, construction = build_plate(frames[start:end],masks[start:end],maps,'median')
        name = f'background_payload_segment_{start:04d}.packet'
        payload = encode_package(subset_parent(parent,start,end),plate,maps,qp=qp,codec=codec,
            encode_plate=encode_plate,arm=f'reset{interval}_segment{start}')
        data[name] = payload
        segments.append({'start':start,'end':end,'member':name,'bytes':len(payload),
                         'sha256':hashlib.sha256(payload).hexdigest()})
        prep.append({'start':start,'end':end,'fit':fit,'plate':construction})
    data['reset_manifest.json'] = _json({'format':'pointstream.registered-plate-reset','version':1,
        'frames':len(frames),'height':frames.shape[1],'width':frames.shape[2],
        'interval_frames':interval,'fps':'12','segments':segments,
        'rate_boundary':'complete bundle bytes, complete segment packets including repeated global appearance'})
    return _archive(data), prep


def reconstruct_reset_bundle(payload, timings=None):
    data = members(payload); spec = json.loads(data.pop('reset_manifest.json'))
    if spec.get('format') != 'pointstream.registered-plate-reset' or spec.get('version') != 1:
        raise ValueError('unsupported reset bundle')
    frames = []; cursor = 0
    for item in spec['segments']:
        if item['start'] != cursor or not cursor < item['end'] <= spec['frames']:
            raise ValueError('reset segment gap/overlap')
        part = data.pop(item['member'])
        if len(part) != item['bytes'] or hashlib.sha256(part).hexdigest() != item['sha256']:
            raise ValueError('reset segment identity mismatch')
        decoded = _reconstruct_rgb_background(part,timings=timings)
        if decoded.shape != (item['end']-cursor,spec['height'],spec['width'],3):
            raise ValueError('reset decoded frame count mismatch')
        frames.append(decoded); cursor = item['end']
    if cursor != spec['frames'] or data: raise ValueError('reset incomplete/unreferenced payloads')
    return np.concatenate(frames)


def _reconstruct_rgb_background(payload, timings=None):
    # Existing still sidecars return OpenCV BGR. Convert explicitly to the
    # documented delivered RGB frame space before ordinary client composition.
    # These pixels are decoder intermediates derived solely from charged bytes.
    import time
    from src.components.background.sidecar import build_sidecar
    data = members(unpack_package(payload)); meta = metadata(data); bg = meta['background']
    keys = bg['wire_payload_keys']
    if len(keys) != 1: raise ValueError('registered still plate requires one charged native packet')
    native = np.load(io.BytesIO(data[keys[0]+'.npy']),allow_pickle=False).tobytes()
    started = time.monotonic()
    plate_bgr = build_sidecar(bg['sidecar_codec']).decode(native)
    if timings is not None: timings['registered_native_plate_decode_s'] = time.monotonic()-started
    plate_rgb = np.ascontiguousarray(plate_bgr[...,::-1])
    if plate_rgb.shape != (meta['height'],meta['width'],3): raise ValueError('decoded native plate raster mismatch')
    data['background_plate.npy'] = _npy(plate_rgb)
    bg['plate_key'] = 'background_plate'; bg['wire_payload_keys'] = []; bg['wire_header_keys'] = []
    data['metadata.npy'] = _npy(np.frombuffer(_json(meta),dtype=np.uint8))
    return reconstruct_serialized_client(_archive(data),timings=timings,require_compressed=True)

def reconstruct_package(payload, timings=None):
    if 'reset_manifest.json' in members(payload): return reconstruct_reset_bundle(payload,timings=timings)
    return _reconstruct_rgb_background(payload,timings=timings)



def encode_arm(parent, frames, *, arm, codec='av1', encode_plate=None):
    selected = next((row for row in ARMS if row[0] == arm), None)
    if selected is None: raise ValueError('unregistered arm')
    name, kind, registered, appearance, qp = selected
    meta = metadata(members(unpack_client_envelope(parent)))
    if frames.shape != (meta['frame_count'],meta['height'],meta['width'],3):
        raise ValueError('original source and parent frame/raster mismatch')
    masks = parent_masks(parent,len(frames))
    if 'reset12' in name or 'reset24' in name:
        interval = 12 if 'reset12' in name else 24
        return encode_reset_bundle(parent,frames,masks,interval=interval,qp=qp,codec=codec,encode_plate=encode_plate)
    if registered: maps, fits = register(frames,masks)
    else:
        maps = np.repeat(np.eye(3,dtype=np.float32)[None],len(frames),axis=0)
        fits = [{'frame':i,'status':'identity_control'} for i in range(len(frames))]
    plate, construction = build_plate(frames,masks,maps,kind)
    packet = encode_package(parent,plate,maps,qp=qp,codec=codec,encode_plate=encode_plate,arm=name)
    return packet, {'fit':fits,'plate':construction,'appearance_parent':appearance,
        'source_rgb_sha256':hashlib.sha256(frames.tobytes()).hexdigest(),
        'plate_rgb_sha256':hashlib.sha256(plate.tobytes()).hexdigest()}


def score(source, decoded, masks):
    """Full-frame metrics and parent-mask diagnostics; exact zero MSE is None."""
    from src.components.codec.frames import rgb_to_luma
    if source.shape != decoded.shape or masks.shape != source.shape[:3]:
        raise ValueError('quality requires all original-target frames/pixels')
    rgb_error = source.astype(np.float64)-decoded.astype(np.float64)
    y_error = rgb_to_luma(source).astype(np.float64)-rgb_to_luma(decoded).astype(np.float64)
    def db(mse): return float(10*np.log10(255**2/mse)) if mse > 0 else None
    per_y = np.mean(y_error**2,axis=(1,2))
    result = {'frames':len(source),'whole_rgb_pooled_mse':float(np.mean(rgb_error**2)),
        'whole_rgb_pooled_psnr_db':db(float(np.mean(rgb_error**2))),
        'whole_y_pooled_mse':float(per_y.mean()),'whole_y_pooled_psnr_db':db(float(per_y.mean())),
        'whole_y_mean_frame_psnr_db':float(np.mean([db(v) for v in per_y])) if (per_y>0).all() else None,
        'whole_y_mse_per_frame':per_y.tolist(),'rgb_mae':float(np.mean(np.abs(rgb_error))),
        'mask_scope':'original parent masks; diagnostics, no independent ROI truth',
        'zero_mse_representation':'null PSNR means exact reconstruction when MSE=0'}
    for label, selected in [('parent_fg',masks.astype(bool)),('parent_bg',~masks.astype(bool))]:
        count = int(selected.sum()); mse = float(np.mean(y_error[selected]**2)) if count else None
        result[label] = {'pixels':count,'pooled_mse_y':mse,'pooled_psnr_y_db':db(mse) if mse is not None else None}
    return result

def main():
    p = argparse.ArgumentParser(); p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--decoded',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True)
    a = p.parse_args()
    from experiments.tier.receiver_replay import receiver_access_guard
    audit, reads, commands = receiver_access_guard(a.packet,a.decoded,a.data_root,Path(__file__).resolve().parents[2])
    sys.addaudithook(audit)
    frames = reconstruct_package(a.packet.read_bytes()); np.save(a.decoded,frames)
    print(json.dumps({'shape':list(frames.shape),'rgb_sha256':hashlib.sha256(frames.tobytes()).hexdigest(),
                      'observed_data_reads':reads,'native_commands':commands,'access_boundary':'observed Python guard/native argv, not OS sandbox'}))

if __name__ == '__main__': main()
