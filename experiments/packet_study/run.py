"""Registered bounded packet rate-quality study; all artifacts remain external."""
from __future__ import annotations
import argparse, hashlib, io, json, os, resource, socket, subprocess, sys, time
from pathlib import Path
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
 os.environ[_key]="1"
import numpy as np

SOURCE='outputs/evaluation-20260914/e03b/run-20260916-federer007/prepared_rgb.npy'
PACKETS='outputs/evaluation-20260914/e06/run-20260916-federer007-perframe-bbox'
SETTINGS=['bbox_resized_first_reference_residual_off','bbox_resized_first_reference_residual_on','per_frame_crop_residual_off','per_frame_crop_residual_on']
REGISTRATION={'source':SOURCE,'settings':SETTINGS,'fps':'12','lossy_mask_scales':[2,4,8],'mask_codecs':['psm1','rle'],'anchor_av1_crf':[32,44,52,58,63],'anchor_vvc_qp':[32,44,52,58,63],'quality':'complete registered RGB frames; BT601 uint8 Y pooled and mean-frame PSNR, RGB MAE; no independent task truth','access':'offline complete-window encoding for all arms; fresh package/native receiver; preinstalled decoding software, no learned weights','exposure':'one retained development prepared cache; raw extraction and held-out exposure not newly certified','accounting':'every persisted transport byte, no free source-side assets; native anchors include file headers and exact deployment manifest','selection':'all registered variants retained, no post-result selection; no BD-rate without quality overlap; narrow scenario allowed'}

def sha(path):
 p=Path(path);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def command(argv,log,stdin=None):
 if os.getloadavg()[0]>40:raise RuntimeError('host load exceeds registered guard')
 started=time.monotonic();p=subprocess.Popen(argv,stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 # communicate with repeated bounded timeouts so load is inspected during native work.
 first=True
 while True:
  try:out,err=p.communicate(input=stdin if first else None,timeout=1);break
  except subprocess.TimeoutExpired:
   first=False
   if os.getloadavg()[0]>40 or time.monotonic()-started>600:
    p.terminate()
    try:p.communicate(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.communicate()
    raise RuntimeError('own child stopped for native time/load guard')
 log.append({'argv':argv,'returncode':p.returncode,'seconds':time.monotonic()-started,'stderr':err.decode('utf8','replace')})
 if p.returncode:raise RuntimeError(err.decode('utf8','replace')[-2500:])
 return out

def quality(source,decoded):
 from src.components.codec.frames import rgb_to_luma
 if source.shape!=decoded.shape:raise ValueError('all registered frames required')
 a=rgb_to_luma(source).astype(np.float64);b=rgb_to_luma(decoded).astype(np.float64);mse=np.mean((a-b)**2,axis=(1,2))
 return {'frames':len(mse),'mse_per_frame_y':mse.tolist(),'pooled_mse_y':float(mse.mean()),'pooled_y_psnr_db':float(10*np.log10(255**2/mse.mean())) if mse.mean() else None,'mean_frame_y_psnr_db':float(np.mean(10*np.log10(255**2/mse))) if (mse>0).all() else None,'rgb_mae':float(np.mean(np.abs(source.astype(np.float64)-decoded)))}

def child(args):
 from experiments.tier.receiver_replay import receiver_access_guard
 guard,reads,commands=receiver_access_guard(args.packet,args.decoded,args.data_root,Path(__file__).resolve().parents[2]);sys.addaudithook(guard)
 from src.runner.packet_packing import unpack_client_envelope
 from src.runner.client import reconstruct_serialized_client
 packed=args.packet.read_bytes();envelope=packed if args.original else unpack_client_envelope(packed)
 frames=np.asarray(reconstruct_serialized_client(envelope,require_compressed=True),dtype=np.uint8)
 np.save(args.decoded,frames,allow_pickle=False)
 print(json.dumps({'packet':sha(args.packet),'shape':list(frames.shape),'decoded_rgb_sha256':hashlib.sha256(frames.tobytes()).hexdigest(),'observed_registered_root_reads':reads,'native_commands':commands,'boundary':'observed Python access guard and native argv, not OS sandbox'}))

def study(args):
 from src.runner.packet_packing import pack_client_envelope
 args.out.mkdir(parents=True,exist_ok=False);write(args.out/'registration-before-run.json',REGISTRATION)
 os.nice(19);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:]);resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3));subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True)
 source_path=args.data_root/SOURCE;source=np.load(source_path,allow_pickle=False)
 if source.shape!=(48,360,640,3) or source.dtype!=np.uint8:raise ValueError('registered prepared input mismatch')
 if hashlib.sha256(source.tobytes()).hexdigest()!='1f02475a5bbc3d94e4bae2e904dc29c3af3082be0c0c160e027b706a6950f6f8':raise ValueError('source hash mismatch')
 log=[];ffmpeg='/opt/local/bin/ffmpeg';report={'registration':REGISTRATION,'stage':args.stage,'smoke':args.smoke,'code_revision':args.code_revision,'worker':sha(__file__),'hostname':socket.getfqdn(),'affinity':sorted(os.sched_getaffinity(0)),'source':sha(source_path),'source_rgb_sha256':hashlib.sha256(source.tobytes()).hexdigest(),'source_rgb_frame_sha256':[hashlib.sha256(f.tobytes()).hexdigest() for f in source],'source_shape':list(source.shape),'environment':{'python':sys.version,'numpy':np.__version__,'threads':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}},'ffmpeg':sha(ffmpeg),'ffmpeg_version':subprocess.run([ffmpeg,'-version'],capture_output=True,text=True).stdout,'gpu_allocated':False,'commands':log,'rows':[]}
 if args.stage=='packing':
  for setting in SETTINGS[:1] if args.smoke else SETTINGS:
   original=(args.data_root/PACKETS/setting/'transport.npz').read_bytes();original_pixels=None
   variants=[('original',None,1),('compact_psm1','psm1',1),('rle_lossless','rle',1)]
   if setting.endswith('_off'):variants += [(f'rle_scale{scale}','rle',scale) for scale in ([2] if args.smoke else REGISTRATION['lossy_mask_scales'])]
   for name,codec,scale in variants:
    directory=args.out/setting/name;directory.mkdir(parents=True);packet=directory/'packet.zip';packet.write_bytes(original if codec is None else pack_client_envelope(original,mask_scale=scale,mask_codec=codec))
    decoded=directory/'decoded.npy';receipt=directory/'receiver.json';cmd=[sys.executable,'-m','experiments.packet_study.run','--data-root',str(args.data_root),'--packet',str(packet),'--decoded',str(decoded),'--receipt',str(receipt)]
    if codec is None:cmd.append('--original')
    write(receipt,json.loads(command(cmd,log)));frames=np.load(decoded,allow_pickle=False);digest=hashlib.sha256(frames.tobytes()).hexdigest()
    if codec is None:original_pixels=digest
    if scale==1 and digest!=original_pixels:raise ValueError('lossless output parity failure')
    report['rows'].append({'setting':setting,'variant':name,'mask_scale':scale,'lossless':scale==1,'input_package':sha(args.data_root/PACKETS/setting/'transport.npz'),'packet':sha(packet),'decoded_rgb_sha256':digest,'parent_decoded_rgb_sha256':original_pixels,'quality':quality(source,frames),'receiver_receipt':sha(receipt)})
    write(args.out/'report.partial.json',report)
 else:
  raw=args.out/'source.rgb';source.tofile(raw)
  for codec,qs in [('av1',REGISTRATION['anchor_av1_crf']),('vvc',REGISTRATION['anchor_vvc_qp'])]:
   for q in [52] if args.smoke else qs:
    directory=args.out/f'{codec}_{q}';directory.mkdir();stream=directory/('stream.ivf' if codec=='av1' else 'stream.266')
    cmd=[ffmpeg,'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x360','-r','12','-i',str(raw),'-an','-frames:v','48','-threads','4']
    if codec=='av1':cmd += ['-c:v','libaom-av1','-cpu-used','6','-row-mt','1','-crf',str(q),'-b:v','0','-g','9999','-pix_fmt','yuv420p','-f','ivf',str(stream)]
    else:cmd += ['-c:v','libvvenc','-preset','medium','-qp',str(q),'-pix_fmt','yuv420p10le','-f','vvc',str(stream)]
    command(cmd,log)
    # Persist and charge explicit common geometry/cadence/placement instead of using source at receiver.
    manifest=directory/'manifest.json';write(manifest,{'schema':'pointstream.native-anchor.v1','codec':codec,'stream':sha(stream),'width':640,'height':360,'frames':48,'fps':'12','output':'rgb24'})
    probe=json.loads(command(['/opt/local/bin/ffprobe','-v','error','-threads','1','-count_frames','-show_entries','stream=codec_name,width,height,pix_fmt,nb_read_frames','-of','json',str(stream)],log))['streams']
    if len(probe)!=1 or probe[0]['width']!=640 or probe[0]['height']!=360 or int(probe[0]['nb_read_frames'])!=48:raise ValueError('native geometry/frame denominator mismatch')
    decoded=command([ffmpeg,'-v','error','-threads','4','-i',str(stream),'-pix_fmt','rgb24','-f','rawvideo','pipe:1'],log)
    frames=np.frombuffer(decoded,np.uint8).reshape(-1,360,640,3);qvalues=quality(source,frames);np.save(directory/'decoded.npy',frames,allow_pickle=False)
    report['rows'].append({'codec':codec,'quantizer':q,'stream':sha(stream),'manifest':sha(manifest),'complete_bytes':stream.stat().st_size+manifest.stat().st_size,'native_probe':probe,'quality':qvalues,'decoded_rgb_sha256':hashlib.sha256(decoded).hexdigest(),'access':'continuous complete-window native encoding, decode arguments contain charged stream only'})
    write(args.out/'report.partial.json',report)
 report['status']='complete';write(args.out/'report.json',report);print(json.dumps({'status':'complete','report':sha(args.out/'report.json')}))

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--stage',choices=['packing','anchors'],default='packing');p.add_argument('--smoke',action='store_true');p.add_argument('--code-revision',default='unfrozen');p.add_argument('--packet',type=Path);p.add_argument('--decoded',type=Path);p.add_argument('--receipt',type=Path);p.add_argument('--original',action='store_true');args=p.parse_args()
 if args.packet:child(args)
 else:study(args)
if __name__=='__main__':main()
