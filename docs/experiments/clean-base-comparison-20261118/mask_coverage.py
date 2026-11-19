"""Audit reviewed body pixels against the paid receiver's projected masks."""
import base64
from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'docs/experiments/receiver-replay-20261118'))
from independent_crop_audit import array_frames, file_sha, read_small, require  # noqa: E402
from independent_finish_audit import UNITS, bind_generation, overlay  # noqa: E402

STAGE=Path('/private/tmp/ps-joint-campus-stage-20261118')
AUDIT=Path('/private/tmp/pointstream-independent-idle-nov18')
LABELS=Path('/Users/manu/Datasets/pointstream-submission-20261118/foreground-human-reviewed-20261118/9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a.json')
SCORES=Path('/private/tmp/pointstream-20261118-clean-score')
VIS=Path('/Users/manu/.codex/visualizations/2026/11/18/01a1edc1-dd4e-7ab3-b819-fbd88da8db05/clean-court-rally-20261118')
OUTPUT=Path(__file__).with_name('mask-coverage.json')


def main():
    cv2.setNumThreads(1)
    require(file_sha(LABELS)=='9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a','labels')
    labels=json.loads(LABELS.read_text())['frames']
    wanted={v['tick']:v for v in labels}
    composition=json.loads((SCORES/'composition.json').read_text())
    records={row['tick']:row for row in composition['rows']}
    report=json.loads((STAGE/'finish/finish.json').read_text())
    rows=[]
    with ExitStack() as stack:
        gens=dict(bind_generation(stack,AUDIT/f'player{i}-new-full.tar',STAGE/f'player{i}-full.json',report) for i in (1,2))
        masks={}
        for unit in UNITS:
            path=STAGE/'finish'/(unit+'-masks.npz')
            require(file_sha(path)==report['mask_files'][unit],'mask input')
            z=stack.enter_context(zipfile.ZipFile(path))
            require(np.array_equal(read_small(z,'t'),np.arange(240)),'mask tick order')
            masks[unit]=stack.enter_context(array_frames(z,'masks',(240,784,512),np.bool_))
        for tick in range(max(wanted)+1):
            current={unit:next(masks[unit]) for unit in UNITS}
            if tick not in wanted:
                continue
            canvas=np.zeros((1080,1920,3),np.uint8)
            source=np.asarray(Image.open(SCORES/'source'/f'{tick:03d}.png').convert('RGB'))
            original=np.asarray(Image.open(SCORES/'original'/f'{tick:03d}.png').convert('RGB'))
            clean=np.asarray(Image.open(SCORES/'clean'/f'{tick:03d}.png').convert('RGB'))
            require(all(hashlib.sha256(img.tobytes()).hexdigest()==records[tick][kind+'_rgb_sha256']
                        for kind,img in [('original',original),('clean',clean)]),'composition identity')
            signed=((clean.astype(np.float32)-source)**2-(original.astype(np.float32)-source)**2).sum(axis=2)
            omitted=np.zeros((1080,1920),bool)
            player_rows={}
            for p in wanted[tick]['players']:
                index=p['id'].split(':')[1]
                unit='nadal_kyrgios__000_at120__person_'+index
                mask=np.asarray(Image.open(BytesIO(base64.b64decode(p['mask']['png_base64']))).convert('L'))>0
                support=np.zeros((1080,1920),bool)
                area,status=overlay(canvas,support,np.zeros((784,512,3),np.uint8),gens[unit][0][tick],current[unit])
                require(status==report['rows'][tick]['status'][unit] and area==report['rows'][tick]['projected_pixels'][unit],'projection')
                missing=mask & ~support
                omitted |= missing
                player_rows[p['id']]={'body_pixels':int(mask.sum()),'covered_pixels':int((mask&support).sum()),
                                     'omitted_pixels':int(missing.sum())}
            rows.append({'tick':tick,'players':player_rows,'omitted_union_pixels':int(omitted.sum()),
                         'signed_excess_sse_inside_omitted':float(signed[omitted].sum(dtype=np.float64)),
                         'signed_excess_sse_outside_omitted':float(signed[~omitted].sum(dtype=np.float64))})
    inside=sum(r['signed_excess_sse_inside_omitted'] for r in rows)
    outside=sum(r['signed_excess_sse_outside_omitted'] for r in rows)
    aggregate={}
    for unit in ('person:000000000000','person:000000000001'):
        total=sum(r['players'][unit]['body_pixels'] for r in rows)
        missed=sum(r['players'][unit]['omitted_pixels'] for r in rows)
        aggregate[unit]={'body_pixels':total,'omitted_pixels':missed,'omitted_fraction':missed/total,
                         'mean_omitted_pixels_per_frame':missed/len(rows)}
    result={'scope':'reviewed 12 frames, body masks only, signed RGB squared-error sums; descriptive paired diagnostic',
            'players':aggregate,'signed_excess_sse_inside':inside,'signed_excess_sse_outside':outside,
            'signed_excess_sse_total':inside+outside,'fraction_total_signed_excess_inside':inside/(inside+outside),
            'rows':rows,'composition_sha256':file_sha(SCORES/'composition.json')}
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
    # Fixed crop chosen before inspection from the established player box at tick 90.
    images=[Image.open(SCORES/kind/'090.png').convert('RGB').crop((850,500,1250,980))
            for kind in ('source','original','clean')]
    out=Image.new('RGB',(1200,520),(16,20,28))
    draw=ImageDraw.Draw(out)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24)
    for i,(image,label) in enumerate(zip(images,('SOURCE','FULL-SOURCE BASE','CLEANED BASE'),strict=True)):
        out.paste(image,(i*400,40))
        draw.text((i*400+12,7),label,font=font,fill='white')
    VIS.mkdir(parents=True,exist_ok=True)
    out.save(VIS/'tick-090-player-racket-crop.png')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    main()
