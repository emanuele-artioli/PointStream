"""Bind paid /002 tiles to the paid cleaned native carrier and replay all ticks."""
import argparse,hashlib,json,os,struct,tarfile,tempfile,time
from pathlib import Path
import cv2,numpy as np
from experiments.tennis import common_player_packet as player, common_player_ball as ball
from src.codecs import svtav1
from src.pointstream import tracked_patch

def sha(raw):return hashlib.sha256(raw).hexdigest()
def member(tar,name):
 with tarfile.open(tar) as f:return f.extractfile(name).read()
def run(a):
 assert a.ticks == 240
 assert len(os.sched_getaffinity(0))==1 and os.environ['CUDA_VISIBLE_DEVICES']==''
 assert sha(a.base.read_bytes())=='776b1ad6f7558ad26b21de26f4d885b24013cd00785ebbe665276239886860d9'
 assert sha(a.tiles.read_bytes())=='a9e0a5e837a7d04bad4a7a0e73fd3e76d14174ceaf3c6977476a5391ba19480b'
 stage=Path(os.environ['PS_STAGE_DIR']);work=Path(os.environ['PS_SCRATCH_DIR']);started=time.monotonic()
 base=member(a.base,'publish/clean-crf60.ivf');tile=member(a.tiles,'publish/tile.ivf');binary=member(a.tiles,'publish/positions.bin'); assert sha(base)=='2fa1f8d752b527707a5e0c1e1be6e070ef34efb91a37aa8ae20ca66215ec2ab9';assert sha(tile)=='f30860d5924a4d2b9980168a587a343eab8edb17479d4db4aa612997ee6f7b25'
 positions=[]
 for x,y in struct.iter_unpack('>HH',binary):positions.append(None if (x,y)==(65535,65535) else [x,y])
 dec=svtav1.tool('dav1d');assert dec['sha256']=='e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6'
 profile={'mode':'native_only','recipe':player.NATIVE_RECIPE,'resources':{'native_decoder':dec['sha256']},'coded_size':[240,136],'canvas_size':[1920,1080],'cuts':[],'runs':[],'order':[]}
 carrier=player.build(profile,base,None,{});side=ball.pack(carrier,positions,tile,decoder_sha256=dec['sha256']);parsed,tile2,size=ball._open(carrier,side,dec['sha256']);assert parsed==[tuple(p) if p else None for p in positions] and tile2==tile and size==(1920,1080)
 native=work/'base.yuv';tracked_patch.decode_native(base,native,{'width':240,'height':136,'fps':[50,1],'frames':240},dec['real_path']);assert sha(native.read_bytes())=='ab8d5890ae3d0365e5dab07e9ee8f2eba934feb165b4cc1a20e35b20be2a5b5c'
 native_frames=np.memmap(native,np.uint8,'r',shape=(240,204,240));hashes=[];current={}
 def canvases():
  for t,frame in enumerate(native_frames):
   rgb=cv2.resize(tracked_patch.yuv_to_rgb(frame,240,136,False),(1920,1080),interpolation=cv2.INTER_CUBIC);current[t]=hashlib.sha256(rgb.tobytes()).hexdigest();yield t,rgb
 for t,frame in ball.render(carrier,side,canvases(),scratch=work,decoder=dec['real_path']):
  digest=hashlib.sha256(frame.tobytes()).hexdigest();hashes.append(digest)
  if positions[t] is None:assert digest==current[t]
 assert len(hashes)==240
 stats=ball.inspect(carrier,side,decoder_sha256=dec['sha256']);stats.update({'frames':240,'placements':sum(p is not None for p in positions),'rendered_hashes':hashes,'elapsed_seconds':time.monotonic()-started,'carrier_sha256':sha(carrier),'sidecar_sha256':sha(side)})
 (stage/'native.pscp').write_bytes(carrier);(stage/'ball.sidecar').write_bytes(side);(stage/'binding.json').write_text(json.dumps(stats,indent=2)+'\n')
def validate():
 s=Path(os.environ['PS_STAGE_DIR']);j=json.loads((s/'binding.json').read_text());checks={'frames':j['frames']==240 and len(j['rendered_hashes'])==240,'placements':j['placements']==231,'carrier':sha((s/'native.pscp').read_bytes())==j['carrier_sha256'],'sidecar':sha((s/'ball.sidecar').read_bytes())==j['sidecar_sha256'],'forecast':os.environ['PS_STAGE']!='smoke' or 2*j['elapsed_seconds']+60<=500};r={'passed':all(checks.values()),'checks':checks};Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(r));assert r['passed']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path);p.add_argument('--tiles',type=Path);p.add_argument('--ticks',type=int);p.add_argument('--validate',action='store_true');a=p.parse_args();validate() if a.validate else run(a)
