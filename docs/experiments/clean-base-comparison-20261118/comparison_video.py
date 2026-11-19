"""Make a same-frame visual from verified generated players and two decoded bases."""
import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT / 'docs/experiments/receiver-replay-20261118'))
from independent_crop_audit import array_frames, file_sha, read_small, require  # noqa: E402
from independent_finish_audit import UNITS, bind_generation, overlay  # noqa: E402
from src.pointstream.tracked_patch import yuv_to_rgb  # noqa: E402

STAGE = Path('/private/tmp/ps-joint-campus-stage-20261118')
AUDIT = Path('/private/tmp/pointstream-independent-idle-nov18')
BALL = Path('/private/tmp/ps-low-ball-stage-20261118/ball-input.npz')
BALL_SHA = '3a010a990d7cb434e7027e35216b41ba21a64829c0ad0281998117c4147786fc'


def run(original,clean,scores,output):
    cv2.setNumThreads(1)
    info=json.loads((scores/'composition.json').read_text())
    require(info['original_base_sha256']==file_sha(original) and info['clean_base_sha256']==file_sha(clean),'paired inputs')
    require(file_sha(BALL)==BALL_SHA,'ball')
    videos = [np.memmap(p,np.uint8,'r',shape=(240,816,960)) for p in (original,clean)]
    data=np.load(BALL,allow_pickle=False)
    report=json.loads((STAGE/'finish/finish.json').read_text())
    labels=Image.new('RGB',(1920,64),(16,20,28))
    draw=ImageDraw.Draw(labels)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',28)
    for x,text in [(480,'FULL-SOURCE BASE + PLAYERS + BALL'),(1440,'CLEANED BASE + PLAYERS + BALL')]:
        box=draw.textbbox((0,0),text,font=font)
        draw.text((x-(box[2]-box[0])/2,15),text,font=font,fill='white')
    title=np.asarray(labels)
    output.parent.mkdir(parents=True,exist_ok=True)
    require(not output.exists(),'output exists')
    ffmpeg=['/opt/homebrew/bin/ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1920x604','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','2',
            '-preset','ultrafast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(output)]
    rows={r['tick']:r for r in info['rows']}
    with ExitStack() as stack:
        gens=dict(bind_generation(stack,AUDIT/f'player{i}-new-full.tar',STAGE/f'player{i}-full.json',report) for i in (1,2))
        masks={}
        for unit in UNITS:
            path=STAGE/'finish'/(unit+'-masks.npz')
            require(file_sha(path)==report['mask_files'][unit], 'mask identity')
            z=stack.enter_context(zipfile.ZipFile(path))
            require(np.array_equal(read_small(z,'t'),np.arange(240)),'mask ticks')
            masks[unit]=stack.enter_context(array_frames(z,'masks',(240,784,512),np.bool_))
        process=subprocess.Popen(ffmpeg,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            for tick in range(240):
                crops={u:next(gens[u][1]) for u in UNITS}
                mask={u:next(masks[u]) for u in UNITS}
                panels=[]
                for kind,base in zip(('original','clean'),videos,strict=True):
                    canvas=cv2.resize(yuv_to_rgb(base[tick],960,544,False),(1920,1080),interpolation=cv2.INTER_CUBIC)
                    support=np.zeros((1080,1920),bool)
                    for unit in UNITS:
                        area,status=overlay(canvas,support,crops[unit],gens[unit][0][tick],mask[unit])
                        require(status==report['rows'][tick]['status'][unit] and area==report['rows'][tick]['projected_pixels'][unit], 'player projection')
                    x,y=data['coords'][tick]
                    if x>=0:
                        canvas[y:y+32,x:x+32]=data['patches'][tick]
                    if tick in rows:
                        require(hashlib.sha256(canvas.tobytes()).hexdigest()==rows[tick][kind+'_rgb_sha256'],'scored tick')
                    panels.append(cv2.resize(canvas,(960,540),interpolation=cv2.INTER_AREA))
                frame=np.concatenate((title,np.concatenate(panels,axis=1)),axis=0)
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
        error=process.stderr.read().decode()
        require(process.wait(timeout=30)==0,'video encode '+error[-500:])
    probe=subprocess.run(['/opt/homebrew/bin/ffprobe','-v','error','-select_streams','v:0','-count_frames',
                          '-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(output)],
                         capture_output=True,text=True,check=True)
    stream=json.loads(probe.stdout)['streams'][0]
    require(stream=={'width':1920,'height':604,'r_frame_rate':'50/1','nb_read_frames':'240'},'video geometry')
    result={'video_sha256':file_sha(output),'stream':stream,'composition_sha256':file_sha(scores/'composition.json'),
            'base_sha256':[file_sha(original),file_sha(clean)],'scored_ticks_checked':len(rows)}
    (scores/'video-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','clean','scores','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.original,args.clean,args.scores,args.output)
