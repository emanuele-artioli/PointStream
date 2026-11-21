"""Create source-derived 32px ball patches in an actual 64px SVT-AV1 stream."""
import argparse, hashlib, json, os, shutil, struct, subprocess, tarfile, time
from fractions import Fraction
from pathlib import Path
import numpy as np
from src.codecs import svtav1

SOURCE='f0551443eaff4e4b3539f21cd0c6ac66f55601d480ad099a571e8626db735e4a'
TRACK='43a816146a72daebfab62db5a87dcb651d431aa408feb859f7df9b7dc60f0859'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def run(a):
 assert len(os.sched_getaffinity(0))==1 and os.environ['CUDA_VISIBLE_DEVICES']==''
 assert sha(a.source)==SOURCE and sha(a.track)==TRACK
 n=a.frames; assert n==(24 if os.environ['PS_STAGE']=='smoke' else 240)
 rows=json.loads(a.track.read_text()); assert rows['source_sha256']==SOURCE and len(rows['centers'])==240
 w,h=1920,1080; work=Path(os.environ['PS_SCRATCH_DIR'])/'tiles';work.mkdir();beg=time.monotonic(); src=work/'source.yuv'
 ff=svtav1.tool('ffmpeg'); enc=svtav1.tool('SvtAv1EncApp'); dec=svtav1.tool('dav1d')
 assert (ff['sha256'],enc['sha256'],dec['sha256'])==('7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2','4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4','e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6')
 subprocess.run([ff['path'],'-v','error','-nostdin','-i',str(a.source),'-frames:v',str(n),'-pix_fmt','yuv420p','-threads','1','-f','rawvideo',str(src)],check=True,timeout=500)
 assert src.stat().st_size==n*w*h*3//2
 source=np.memmap(src,np.uint8,'r',shape=(n,h*3//2,w)); patches=np.empty((n,64*64*3//2),np.uint8); positions=[]
 for t,point in enumerate(rows['centers'][:n]):
  p=patches[t];p[:4096]=16;p[4096:]=128
  if point is None:positions.append(None);continue
  x=max(0,min(w-32,int(point[0])-16))//2*2; y=max(0,min(h-32,int(point[1])-16))//2*2
  raw=source[t].reshape(-1);ysize=w*h;csize=ysize//4;Y=raw[:ysize].reshape(h,w);U=raw[ysize:ysize+csize].reshape(h//2,w//2);V=raw[ysize+csize:].reshape(h//2,w//2)
  p[:4096].reshape(64,64)[16:48,16:48]=Y[y:y+32,x:x+32]
  p[4096:5120].reshape(32,32)[8:24,8:24]=U[y//2:y//2+16,x//2:x//2+16];p[5120:].reshape(32,32)[8:24,8:24]=V[y//2:y//2+16,x//2:x//2+16];positions.append([x,y])
 raw=work/'tiles.yuv';raw.write_bytes(patches.tobytes());out=work/'codec';out.mkdir()
 codec=svtav1.code(raw,out,width=64,height=64,fps=Fraction(50),frames=n,crf=20,preset=4,threads=1,full_range=False,encoder=enc['path'],decoder=dec['path'],timeout=500)
 stream=Path(codec['stream']); decoded=Path(codec['decoded']);assert codec['decoded_frames']==n
 pub=work/'publish';pub.mkdir();shutil.copyfile(stream,pub/'tile.ivf');binary=b''.join(struct.pack('>HH',*(point if point is not None else (65535,65535))) for point in positions);(pub/'positions.bin').write_bytes(binary)
 report={'source_sha256':SOURCE,'track_sha256':TRACK,'frames':n,'placements':sum(x is not None for x in positions),'tile_bytes':stream.stat().st_size,'positions_bytes':len(binary),'tile_sha256':sha(stream),'decoded_sha256':sha(decoded),'elapsed_seconds':time.monotonic()-beg,'encoder':enc,'decoder':dec,'settings':{'tile':[64,64],'patch':[32,32],'fps':[50,1],'crf':20,'preset':4},'positions':positions}
 (pub/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');stage=Path(os.environ['PS_STAGE_DIR']);(stage/'tile-report.json').write_text(json.dumps(report,indent=2)+'\n')
 with tarfile.open(stage/'published.tar','w') as tar:tar.add(pub,arcname='publish')
def validate():
 stage=Path(os.environ['PS_STAGE_DIR']);j=json.loads((stage/'tile-report.json').read_text());smoke=os.environ['PS_STAGE']=='smoke';n=24 if smoke else 240
 with tarfile.open(stage/'published.tar') as tar:tile=tar.extractfile('publish/tile.ivf').read();positions=tar.extractfile('publish/positions.bin').read()
 checks={'source':j['source_sha256']==SOURCE,'track':j['track_sha256']==TRACK,'frames':j['frames']==n,'tile':hashlib.sha256(tile).hexdigest()==j['tile_sha256'] and len(svtav1.ivf_frames(tile))==n,'positions':len(positions)==4*n,'settings':j['settings']['crf']==20,'forecast':not smoke or 1.5*j['elapsed_seconds']*10+60<=600}
 result={'passed':all(checks.values()),'checks':checks};Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(result,indent=2)+'\n');assert result['passed']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--track',type=Path);p.add_argument('--frames',type=int);p.add_argument('--validate',action='store_true');a=p.parse_args();validate() if a.validate else run(a)
