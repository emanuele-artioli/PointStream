"""Validate and import explicit human decisions from the private Al028 review packet.

Only accepted or replacement player decisions produce masks. Missing/unsure are
retained as unresolved, and visible ball points must have been explicitly marked.
"""
import argparse
import csv
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_review(root):
    value=json.loads((root/'review.json').read_text())
    digest=value.pop('review_sha256')
    canonical=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if sha(canonical)!=digest:
        raise ValueError('review manifest digest mismatch')
    value['review_sha256']=digest
    return value


def vertices(polygons, size=(1920, 1080)):
    if not isinstance(polygons,list) or not polygons:
        raise ValueError('edited player requires replacement polygons')
    for poly in polygons:
        if not isinstance(poly,list) or len(poly)<3:
            raise ValueError('polygon needs three vertices')
        for point in poly:
            if not isinstance(point,list) or len(point)!=2 or any(type(n) is not int for n in point) or not (0<=point[0]<size[0] and 0<=point[1]<size[1]):
                raise ValueError('polygon vertex outside the source')
        area=sum(poly[i][0]*poly[(i+1)%len(poly)][1]-poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly)))
        if not area:
            raise ValueError('degenerate polygon')
    return polygons


def official_positions(review, decisions, csv_path):
    expected = review.get("official_ball_sha256")
    if not expected:
        if csv_path is not None or "official_ball_sha256" in decisions:
            raise ValueError("unexpected official ball input for this review")
        return None
    if csv_path is None or decisions.get("official_ball_sha256") != expected or sha(csv_path.read_bytes()) != expected:
        raise ValueError("published ball annotation identity mismatch")
    marks = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            tick = int(row["Frame"])
            if tick in marks:
                raise ValueError("duplicate official ball timestamp")
            visible = int(row["Visibility"])
            if visible in (0, 1):
                marks[tick] = {"status": "visible" if visible else "missing",
                               "point": [int(row["X"]), int(row["Y"])] if visible else None}
    return marks


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--official-ball',type=Path)
    a=p.parse_args()
    review=read_review(a.review)
    decisions=json.loads(a.decisions.read_text())
    if decisions.get('version')!=1 or decisions.get('review_sha256')!=review['review_sha256'] or not isinstance(decisions.get('reviewer'),str) or not decisions['reviewer'].strip():
        raise ValueError('review identity or reviewer missing')
    if not isinstance(decisions.get('frames'),list) or len(decisions['frames'])!=len(review['frames']):
        raise ValueError('incomplete frame decisions')
    from_source = review.get("draw_from_source") is True
    size = (review["width"], review["height"]) if from_source else (1920, 1080)
    if any(type(n) is not int or n <= 0 for n in size):
        raise ValueError("invalid native dimensions")
    marks=official_positions(review,decisions,a.official_ball)
    out=[]
    for frame,chosen in zip(review['frames'],decisions['frames'],strict=True):
        if (chosen.get('id'),chosen.get('frame_index'),chosen.get('source_sha256'))!=(frame['id'],frame['frame_index'],frame['source_sha256']):
            raise ValueError('source frame identity mismatch')
        if sha((a.review/frame['source']).read_bytes())!=frame['source_sha256']:
            raise ValueError('source frame changed')
        if not isinstance(chosen.get('players'),list) or len(chosen['players'])!=2:
            raise ValueError('two distinct player decisions required')
        people=[]
        for proposal,answer in zip(frame['players'],chosen['players'],strict=True):
            status=answer.get('status')
            if from_source:
                if status not in ('edit', 'missing', 'unsure') or (status != 'edit' and answer.get('polygons')):
                    raise ValueError('source-only player decision requires explicit geometry or an empty status')
                result={'id':proposal['id'],'status':status,'mask':None}
            else:
                content=(a.review/proposal['proposal']).read_bytes()
                if sha(content)!=proposal['sha256'] or answer.get('proposal_sha256')!=proposal['sha256']:
                    raise ValueError('proposal identity mismatch')
                if status not in ('accept','edit','missing','unsure'):
                    raise ValueError('invalid player decision')
                result={'id':proposal['id'],'status':status,'proposal_sha256':proposal['sha256'],'mask':None}
            if status in ('accept','edit'):
                if status=='accept':
                    image=Image.open(io.BytesIO(content)).convert('L')
                else:
                    image=Image.new('L',size,0)
                    draw=ImageDraw.Draw(image)
                    for poly in vertices(answer.get('polygons'), size):
                        draw.polygon([tuple(x) for x in poly],fill=255)
                if image.size!=size:
                    raise ValueError('mask size mismatch')
                count=int(np.count_nonzero(np.asarray(image)))
                if not count:
                    raise ValueError('empty approved player mask')
                buf=io.BytesIO()
                image.save(buf,format='PNG')
                result['mask']={'png_base64':base64.b64encode(buf.getvalue()).decode(),'foreground_pixels':count,'sha256':sha(buf.getvalue())}
                if from_source:
                    result['method']='human_source_polygons'
                else:
                    result['method']='human_accept_source_model_proposal' if status=='accept' else 'human_replacement_polygons'
            people.append(result)
        if marks is not None and ('ball' in chosen or frame['frame_index'] not in marks):
            raise ValueError('an official-only ball tick is missing or has a separate reviewer decision')
        ball=chosen.get('ball') if marks is None else marks[frame['frame_index']]
        if not isinstance(ball,dict) or ball.get('status') not in ('visible','missing','unsure'):
            raise ValueError('invalid ball status')
        point=ball.get('point')
        if ball['status']=='visible':
            if not isinstance(point,list) or len(point)!=2 or any(type(x) is not int for x in point) or not (0<=point[0]<size[0] and 0<=point[1]<size[1]):
                raise ValueError('visible ball requires an in-bounds point')
        elif point is not None:
            raise ValueError('missing or unsure ball cannot have a point')
        out.append({'tick':frame['frame_index'],'source_png_sha256':frame['source_sha256'],'players':people,'ball':{'status':ball['status'],'center_xy':point},'note':chosen.get('note','')})
        if from_source:
            out[-1].update(absolute_tick=frame['absolute_tick'], source_rgb_sha256=frame['source_rgb_sha256'])
    if a.out.exists():
        raise ValueError('output already exists')
    a.out.mkdir(parents=True)
    result={'schema':'pointstream-al028-human-review-v1','reviewer':decisions['reviewer'],'source_sha256':review['source_video_sha256'],'review_sha256':review['review_sha256'],'width':1920,'height':1080,'fps':review['source_video_fps'],'frames':out}
    if from_source:
        result.update(schema="pointstream-source-polygons-and-ball-v1", width=size[0], height=size[1],
                      rally_id=review["source_group"], original_video_sha256=review["original_video_sha256"])
    if marks is not None:
        result.update(schema="pointstream-reviewed-player-official-ball-v1",
                      ball_source="official dataset",official_ball_sha256=review["official_ball_sha256"])
    (a.out/'annotations.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'frames':len(out),'reviewer':result['reviewer'],'output':str(a.out/'annotations.json')}))


if __name__=='__main__':
    main()
