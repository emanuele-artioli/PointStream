"""Posthoc parent-mask transport changes; no human labels or task accuracy.

Every registered packet identity is checked. Six lossy mask interventions are
compared to their original delivered union masks across all48 full rasters.
No source RGB, learned segmentation model, or annotation file is opened.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

SHAPE=(48,360,640)
SOURCE_RGB_SHA='1f02475a5bbc3d94e4bae2e904dc29c3af3082be0c0c160e027b706a6950f6f8'
REPORT_SHA='9318c324bbe520a309e4495d5700dcec715b91477d1bd1dbb54449e74a565d5f'
REPORT_REVISION='bfab96c3549ba6ed9acfb709caf796e07a8ee67d'
PARENTS=('bbox_resized_first_reference_residual_off','per_frame_crop_residual_off')
SETTINGS=(*PARENTS,'bbox_resized_first_reference_residual_on','per_frame_crop_residual_on')
EXPECTED_KEYS={(setting,variant) for setting in SETTINGS for variant in ('original','compact_psm1','rle_lossless')}
EXPECTED_KEYS |= {(setting,f'rle_scale{scale}') for setting in PARENTS for scale in (2,4,8)}
EXPECTED_KEYS.add((PARENTS[0],'retained_floor_rle'))


def sha(data):return hashlib.sha256(data).hexdigest()


def checked_payload(identity):
    if not isinstance(identity,dict) or not isinstance(identity.get('path'),str):raise ValueError('explicit package identity required')
    data=Path(identity['path']).read_bytes()
    if type(identity.get('bytes')) is not int or len(data)!=identity['bytes'] or sha(data)!=identity.get('sha256'):
        raise ValueError('package bytes/SHA256 differ from frozen report')
    return data


def report_index(report):
    if report.get('code_revision')!=REPORT_REVISION or report.get('source_rgb_sha256')!=SOURCE_RGB_SHA or report.get('source_shape')!=[48,360,640,3]:
        raise ValueError('frozen packet report revision/source/raster mismatch')
    rows=report.get('rows')
    if not isinstance(rows,list) or len(rows)!=19:raise ValueError('exact19 report rows required')
    indexed={}
    for row in rows:
        key=(row.get('setting'),row.get('variant'))
        if key in indexed:raise ValueError('duplicate report arm')
        if key not in EXPECTED_KEYS:raise ValueError('unregistered report arm')
        if row.get('quality',{}).get('frames')!=48:raise ValueError('all48 scored frames required')
        scale=int(key[1][-1]) if key[1] in ('rle_scale2','rle_scale4','rle_scale8') else 1
        if row.get('mask_scale')!=scale:raise ValueError('report mask scale mismatch')
        indexed[key]=row
    if set(indexed)!=EXPECTED_KEYS:raise ValueError('incomplete registered arm inventory')
    return indexed


def decoded_union(payload):
    from src.runner.packet_packing import unpack_client_envelope
    from experiments.background.registered_plate_codec import parent_masks,members,metadata
    ordinary=unpack_client_envelope(payload);meta=metadata(members(ordinary))
    if (meta['frame_count'],meta['height'],meta['width'])!=SHAPE:
        raise ValueError('decoded mask package requires exact48x360x640 coverage')
    masks=parent_masks(ordinary,48)
    if masks.shape!=SHAPE:raise ValueError('decoded union raster mismatch')
    return masks


def compare_masks(original,changed,*,shape=SHAPE):
    if original.shape!=shape or changed.shape!=shape:raise ValueError('all registered mask frames and pixels required')
    for array in (original,changed):
        if array.dtype not in (np.dtype(bool),np.dtype(np.uint8)) or not np.isin(array,[0,1]).all():
            raise ValueError('binary parent/delivered masks required')
    a=original.astype(bool);b=changed.astype(bool)
    tp=np.count_nonzero(a&b,axis=(1,2));fn=np.count_nonzero(a&~b,axis=(1,2));fp=np.count_nonzero(~a&b,axis=(1,2))
    positive=np.count_nonzero(a,axis=(1,2));delivered=np.count_nonzero(b,axis=(1,2));union=tp+fn+fp
    original_count=int(positive.sum());total_pixels=int(np.prod(shape));tp_all=int(tp.sum());fn_all=int(fn.sum());fp_all=int(fp.sum());union_all=int(union.sum())
    fraction=lambda count:float(count/original_count) if original_count else None
    return {'scope':'posthoc changes relative to original transmitted parent segmentation; no human/task truth',
        'frames':shape[0],'shape':list(shape),'original_mask_sha256':sha(np.ascontiguousarray(a).tobytes()),
        'delivered_mask_sha256':sha(np.ascontiguousarray(b).tobytes()),
        'whole_raster_pixels':total_pixels,'fixed_original_positive_pixels':original_count,
        'intersection_pixels':tp_all,'union_pixels':union_all,
        'pooled_binary_iou':float(tp_all/union_all) if union_all else None,
        'fn_pixels_relative_to_parent':fn_all,'fp_pixels_relative_to_parent':fp_all,
        'fn_fraction_of_fixed_original_positive':fraction(fn_all),
        'fp_fraction_of_fixed_original_positive':fraction(fp_all),
        'changed_pixels':fn_all+fp_all,'changed_fraction_of_fixed_original_positive':fraction(fn_all+fp_all),
        'changed_fraction_of_whole_raster':float((fn_all+fp_all)/total_pixels),
        'nonempty_parent_to_empty_delivered_frames':np.flatnonzero((positive>0)&(delivered==0)).tolist(),
        'empty_parent_to_nonempty_delivered_frames':np.flatnonzero((positive==0)&(delivered>0)).tolist(),
        'empty_union_frames':np.flatnonzero(union==0).tolist(),
        'zero_denominator_policy':'null IoU for empty pooled union; null parent-normalized fractions for zero original positives',
        'per_frame':[{'frame':i,'original_positive_pixels':int(positive[i]),'delivered_positive_pixels':int(delivered[i]),
            'intersection_pixels':int(tp[i]),'union_pixels':int(union[i]),
            'fn_pixels_relative_to_parent':int(fn[i]),'fp_pixels_relative_to_parent':int(fp[i]),
            'changed_pixels':int(fn[i]+fp[i])} for i in range(shape[0])]}


def diagnose(report):
    indexed=report_index(report)
    # Validate all19 retained transport identities before decoding any mask.
    # The floor control is inventoried, not interpreted by the modern adapter.
    inventories=[]
    for key,row in indexed.items():
        checked_payload(row['packet'])
        inventories.append({'setting':key[0],'variant':key[1],'packet':row['packet']})
    results=[]
    for setting in PARENTS:
        parent=indexed[(setting,'original')]
        parent_mask=decoded_union(checked_payload(parent['packet']))
        for scale in (2,4,8):
            row=indexed[(setting,f'rle_scale{scale}')]
            delivered=decoded_union(checked_payload(row['packet']))
            results.append({'setting':setting,'variant':row['variant'],'mask_scale':scale,
                'original_packet':parent['packet'],'delivered_packet':row['packet'],
                'whole_y_psnr_delta_db_from_packet_report':row['quality']['pooled_y_psnr_db']-parent['quality']['pooled_y_psnr_db'],
                **compare_masks(parent_mask,delivered)})
    return {'status':'complete','analysis':'posthoc mask transport diagnostic; no source RGB access or human labels',
        'packet_code_revision':REPORT_REVISION,'source_rgb_sha256_identity_only':SOURCE_RGB_SHA,
        'inventoried_packets':inventories,'rows':results,'expected_rows':6,
        'model_scope':'original parent masks are model outputs, not independent truth',
        'interpretation_boundary':'small whole-frame PSNR changes may coexist with substantial parent-mask changes; no task-accuracy conclusion'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=a.report.read_bytes()
    if sha(data)!=REPORT_SHA:raise ValueError('exact frozen full packet report SHA256 required')
    if not os.environ.get('PS_CODE_REVISION'):raise ValueError('frozen diagnostic code revision required')
    result=diagnose(json.loads(data));result.update(report={'path':str(a.report.resolve()),'bytes':len(data),'sha256':sha(data)},
        code_revision=os.environ['PS_CODE_REVISION'],worker={'path':__file__,'sha256':sha(Path(__file__).read_bytes())},
        environment={'python':sys.version,'numpy':np.__version__,'command':sys.argv,'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'thread_environment':{key:os.environ[key] for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'gpu_allocated':False})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
