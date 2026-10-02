"""Bounded native VMAF controls on the registered Gate A source prefixes."""
import argparse,hashlib,json,os,pathlib,subprocess,sys,time

def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,a.legacy_root);os.environ['CUDA_VISIBLE_DEVICES']='';os.nice(19)
 import numpy as np,cv2
 from src.components.metrics.vmaf import _write_y4m_clip,_read_vmaf_mean
 base=pathlib.Path(a.data_root)/'outputs/gate-a-vvc-webp-n96-run2/points/C2.run';expected=['09fbaee1f343b6046718853ef82b8b257c3bb8a9bbff8c52c23c368aeedce573','2009a1bb6934c4109f16f73951acb2af5a22d91a3cb15f7f79fc3dc883bb6c6b'];prefixes=[];identities=[]
 for i,wanted in enumerate(expected):
  path=base/f'chunk_{i:02d}/source.npy';arr=np.load(path,mmap_mode='r',allow_pickle=False)
  if arr.shape!=(96,2160,3840,3) or arr.dtype!=np.uint8:raise ValueError('source raster')
  got=hashlib.sha256(np.ascontiguousarray(arr).data).hexdigest()
  if got!=wanted:raise ValueError('source pixels')
  identities.append({'path':str(path),'rgb_sha256':got,'file_sha256':digest(path)});prefixes.append(np.asarray(arr[:2]))
 source=np.concatenate(prefixes);ff='/opt/local/bin/ffmpeg';ref=out/'reference.y4m';_write_y4m_clip(ref,source,ff);rows=[]
 for label,sigma in [('identical',None),('mild_blur',1.0),('severe_blur',16.0),('black_null',0)]:
  frames=source.copy() if sigma is None else np.zeros_like(source) if sigma==0 else np.stack([cv2.GaussianBlur(x,(0,0),sigmaX=sigma,sigmaY=sigma) for x in source])
  predicted=out/(label+'.y4m');log=out/(label+'.json');_write_y4m_clip(predicted,frames,ff)
  filt=f'[1:v]format=yuv420p[dist];[0:v]format=yuv420p[ref];[dist][ref]libvmaf=model=version=vmaf_v0.6.1:log_path={log}:log_fmt=json:n_threads=8'
  cmd=[ff,'-hide_banner','-loglevel','error','-i',str(ref),'-i',str(predicted),'-filter_complex',filt,'-f','null','-'];start=time.monotonic();subprocess.run(cmd,check=True,timeout=600,capture_output=True);data=json.loads(log.read_text());values=[x['metrics']['vmaf'] for x in data['frames']]
  if len(values)!=4 or not all(np.isfinite(values)):raise ValueError('all-frame metric control')
  rows.append({'name':label,'mean':_read_vmaf_mean(log),'per_frame':values,'command':cmd,'seconds':time.monotonic()-start,'log_sha256':digest(log),'predicted_rgb_sha256':hashlib.sha256(frames.data).hexdigest()})
 by={x['name']:x['mean'] for x in rows};passed=by['identical']>95 and by['identical']>by['mild_blur']>by['severe_blur'] and by['black_null']<by['mild_blur']
 report={'scope':'instrument controls, not codec evidence','source':identities,'registered_prefix_frames_each':2,'model':'explicit built-in vmaf_v0.6.1','ffmpeg_sha256':digest(ff),'rows':rows,'passed':passed}
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 if not passed:raise RuntimeError('metric controls failed; full comparison prohibited')
if __name__=='__main__':main()
