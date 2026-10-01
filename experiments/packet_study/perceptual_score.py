"""Common full-frame LPIPS/DISTS scorer; inputs explicitly declared and hashed.

Primary evaluation always uses all 48 prepared frames, 360x640, RGB. No crops,
resizing, available-frame filtering, or score-dependent input selection. Model
files must already exist; this entry point cannot download or train weights.
GPU execution is an explicit caller choice, reserved for fleet dispatch.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

SOURCE_RGB_SHA256 = '1f02475a5bbc3d94e4bae2e904dc29c3af3082be0c0c160e027b706a6950f6f8'
SHAPE = (48,360,640,3)
FORMATS = ('uint8_nhwc_rgb','float32_nchw_rgb01')
WEIGHT_IDS = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'lpips_alex_v01': 'https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/alex.pth',
    'dists': 'https://github.com/dingkeyan93/DISTS/blob/master/weights.pt',
}
REGISTRATION = {'shape':list(SHAPE),'source_rgb_sha256':SOURCE_RGB_SHA256,
    'primary':'complete full-frame RGB; no resize/crop; every48 frame included',
    'quantization':'float32 RGB[0,1] -> float64 multiply255 -> np.rint (ties-to-even) -> uint8; no clipping',
    'lpips':{'distribution':'lpips==0.1.4','network':'alex','model_version':'0.1','normalize':True},
    'dists':{'distribution':'DISTS-pytorch==0.1.0','input_range':[0,1],'require_grad':False,'batch_average':False},
    'aggregation':'per-frame perceptual distances and arithmetic mean; lower better',
    'y_metric':'BT601 0.299R+0.587G+0.114B, clipped uint8 truncation, pooled MSE then PSNR',
    'weights':'preinstalled evaluator assets, not codec payload; four explicit paths and complete SHA256s',
    'selection':'registered independently; all declared reconstructions required, no partial result',
    'native_float_parity':'outside this scoring contract; common uint8 conversion applies to all float methods'}


def sha_bytes(data): return hashlib.sha256(data).hexdigest()


def file_identity(path):
    path=Path(path).resolve(); h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def checked_file(spec):
    if not isinstance(spec,dict) or not isinstance(spec.get('path'),str) or not isinstance(spec.get('sha256'),str):
        raise ValueError('every input requires explicit path and SHA256')
    identity=file_identity(spec['path'])
    if identity['sha256'] != spec['sha256']:raise ValueError('file SHA256 mismatch: '+identity['path'])
    if 'bytes' in spec and identity['bytes'] != spec['bytes']:raise ValueError('file byte count mismatch')
    return identity


def common_rgb(array, encoding, *, shape=SHAPE):
    """Strict shared conversion; float overshoot is rejected rather than clipped."""
    if encoding == 'uint8_nhwc_rgb':
        if array.dtype != np.uint8 or array.shape != shape:raise ValueError('uint8 reconstruction requires complete NHWC RGB raster')
        return np.ascontiguousarray(array)
    if encoding == 'float32_nchw_rgb01':
        if array.dtype != np.float32 or array.shape != (shape[0],3,shape[1],shape[2]):
            raise ValueError('float reconstruction requires complete float32 NCHW RGB raster')
        if not np.isfinite(array).all() or array.min()<0 or array.max()>1:
            raise ValueError('float reconstruction must be finite RGB in [0,1]')
        return np.ascontiguousarray(np.rint(array.astype(np.float64).transpose(0,2,3,1)*255).astype(np.uint8))
    raise ValueError('unsupported explicit reconstruction encoding')


def load_inputs(source_path, manifest):
    if manifest.get('schema') != 1 or not isinstance(manifest.get('reconstructions'),list) or not manifest['reconstructions']:
        raise ValueError('manifest schema1 requires nonempty reconstruction list')
    source_identity=checked_file(manifest['source'])
    if Path(source_path).resolve()!=Path(source_identity['path']):raise ValueError('CLI source differs from manifest source')
    source=common_rgb(np.load(source_identity['path'],allow_pickle=False),'uint8_nhwc_rgb')
    pixel_sha=sha_bytes(source.tobytes())
    if pixel_sha != SOURCE_RGB_SHA256:raise ValueError('source differs from frozen prepared RGB pixel identity')
    rows=[];names=set()
    for spec in manifest['reconstructions']:
        method=spec.get('method')
        if not isinstance(method,str) or not method or method in names:raise ValueError('method identities must be nonempty and unique')
        names.add(method)
        if type(spec.get('frame_count')) is not int or spec['frame_count'] != SHAPE[0]:raise ValueError('all48 declared frames required')
        identity=checked_file(spec)
        raw=np.load(identity['path'],allow_pickle=False)
        rgb=common_rgb(raw,spec.get('encoding'))
        pixel_sha=sha_bytes(rgb.tobytes())
        if 'common_rgb_sha256' in spec and spec['common_rgb_sha256']!=pixel_sha:raise ValueError('common reconstructed RGB SHA mismatch')
        rows.append((method,rgb,{'input':identity,'encoding':spec['encoding'],'raw_shape':list(raw.shape),
            'raw_dtype':str(raw.dtype),'raw_array_sha256':sha_bytes(np.ascontiguousarray(raw).tobytes()),'common_rgb_shape':list(rgb.shape),'common_rgb_sha256':pixel_sha}))
    return source,{'input':source_identity,'rgb_sha256':sha_bytes(source.tobytes()),'shape':list(source.shape)},rows


def luma_quality(source, decoded):
    from src.components.codec.frames import rgb_to_luma
    if source.shape != decoded.shape:raise ValueError('matched full raster required')
    a=rgb_to_luma(source).astype(np.float64);b=rgb_to_luma(decoded).astype(np.float64)
    per=np.mean((a-b)**2,axis=(1,2));pooled=float(per.mean())
    psnr=lambda mse:float(10*np.log10(255**2/mse)) if mse>0 else None
    return {'mse_per_frame_y':per.tolist(),'pooled_mse_y':pooled,'pooled_y_psnr_db':psnr(pooled),
            'mean_frame_y_psnr_db':float(np.mean([psnr(v) for v in per])) if (per>0).all() else None,
            'zero_mse_psnr':'null represents positive infinity at zero MSE'}


def score_arrays(source,decoded,engine,batch_size=1):
    if type(batch_size) is not int or not 1<=batch_size<=48:raise ValueError('bounded batch size required')
    if source.shape!=SHAPE or source.dtype!=np.uint8 or decoded.shape!=SHAPE or decoded.dtype!=np.uint8:
        raise ValueError('primary scorer requires complete registered uint8 RGB raster')
    lpips_values=[];dists_values=[]
    for start in range(0,len(source),batch_size):
        end=min(start+batch_size,len(source))
        a=np.ascontiguousarray(source[start:end].transpose(0,3,1,2).astype(np.float32)/255)
        b=np.ascontiguousarray(decoded[start:end].transpose(0,3,1,2).astype(np.float32)/255)
        lp,di=engine(a,b)
        for values,dest in [(lp,lpips_values),(di,dists_values)]:
            values=np.asarray(values,dtype=np.float64).reshape(-1)
            if len(values)!=end-start or not np.isfinite(values).all():raise ValueError('perceptual scorer did not emit every finite per-frame distance')
            dest.extend(values.tolist())
    return {**luma_quality(source,decoded),'lpips_per_frame':lpips_values,'dists_per_frame':dists_values,
            'lpips_mean':float(np.mean(lpips_values)),'dists_mean':float(np.mean(dists_values)),'frames':len(source)}


def state_identity(model):
    h=hashlib.sha256();tensors=[]
    for key,value in sorted(model.state_dict().items()):
        value=value.detach().cpu().contiguous(); raw=value.numpy().tobytes()
        descriptor={'key':key,'dtype':str(value.dtype),'shape':list(value.shape),'bytes':len(raw),'sha256':sha_bytes(raw)}
        h.update(json.dumps(descriptor,sort_keys=True,separators=(',',':')).encode()+b'\n');tensors.append(descriptor)
    return {'tensor_manifest_sha256':h.hexdigest(),'tensors':tensors}


def distribution_identity(name,module):
    dist=importlib.metadata.distribution(name)
    files={}
    for file in dist.files or ():
        if str(file).endswith(('/RECORD','/METADATA')):
            files[str(file)]=file_identity(dist.locate_file(file))
    return {'distribution':name,'version':dist.version,'module':file_identity(inspect.getfile(module)),
            'installed_distribution_manifests':files}


@contextlib.contextmanager
def no_weight_downloads(torch):
    original=torch.hub.download_url_to_file
    def reject(*args,**kwargs):raise RuntimeError('scoring forbids model downloads; prepare all declared assets first')
    torch.hub.download_url_to_file=reject
    try:yield
    finally:torch.hub.download_url_to_file=original


class OfficialEngine:
    def __init__(self,weights,device='cpu',gpu_uuid=None):
        import torch
        import torchvision
        import lpips
        from DISTS_pytorch import DISTS
        if importlib.metadata.version('lpips')!='0.1.4':raise ValueError('requires lpips0.1.4')
        if importlib.metadata.version('DISTS-pytorch') not in ('0.1','0.1.0'):raise ValueError('requires DISTS-pytorch0.1.0')
        if set(weights)!=set(WEIGHT_IDS):raise ValueError('four official evaluator weight identities required')
        assets={key:checked_file(value) for key,value in weights.items()}
        for key in assets:
            if weights[key].get('official_id')!=WEIGHT_IDS[key]:raise ValueError('declared official model identity mismatch')
        checkpoint_dir=Path(torch.hub.get_dir()).resolve()/'checkpoints'
        for key in ('alexnet','vgg16'):
            expected=checkpoint_dir/WEIGHT_IDS[key].rsplit('/',1)[-1]
            if Path(assets[key]['path'])!=expected:raise ValueError('backbone must occupy declared Torch Hub cache path: '+str(expected))
        if device not in ('cpu','cuda:0'):raise ValueError('device must be cpu or fleet-isolated cuda:0')
        if device.startswith('cuda') and (not gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu_uuid):
            raise ValueError('CUDA scoring requires explicit fleet-isolated GPU UUID')
        torch.set_num_threads(1);torch.manual_seed(0)
        with no_weight_downloads(torch):
            self.lp=lpips.LPIPS(net='alex',version='0.1',pretrained=True,pnet_rand=False,
                spatial=False,model_path=assets['lpips_alex_v01']['path'],eval_mode=True,verbose=False).eval()
            self.di=DISTS(load_weights=False).eval()
            calibration=torch.load(assets['dists']['path'],map_location='cpu',weights_only=True)
            if set(calibration)!= {'alpha','beta'}:raise ValueError('unexpected DISTS calibration tensors')
            with torch.no_grad():
                for name in ('alpha','beta'):
                    tensor=calibration[name]
                    if tensor.shape!=getattr(self.di,name).shape or not torch.isfinite(tensor).all():raise ValueError('invalid DISTS calibration')
                    getattr(self.di,name).copy_(tensor)
        for model in (self.lp,self.di):
            model.requires_grad_(False)
        self.provenance={'assets':assets,'weight_official_ids':WEIGHT_IDS,
            'lpips_state':state_identity(self.lp),'dists_state':state_identity(self.di),
            'libraries':{'torch':distribution_identity('torch',torch),'torchvision':distribution_identity('torchvision',torchvision),
                'lpips':distribution_identity('lpips',lpips),'dists':distribution_identity('DISTS-pytorch',inspect.getmodule(DISTS))},
            'device':device,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'declared_gpu_uuid':gpu_uuid,
            'torch_cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
            'model_boundary':'preinstalled evaluation models; no learned codec asset supplied by scorer'}
        if device.startswith('cuda'):
            prop=torch.cuda.get_device_properties(0)
            observed=str(getattr(prop,'uuid','unavailable'))
            if observed!='unavailable' and observed.removeprefix('GPU-')!=gpu_uuid.removeprefix('GPU-'):
                raise ValueError('observed CUDA UUID differs from fleet declaration')
            self.provenance['gpu']={'name':prop.name,'uuid_observed':observed,'total_memory':prop.total_memory}
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
        self.torch=torch;self.device=device;self.lp=self.lp.to(device);self.di=self.di.to(device)

    def __call__(self,a,b):
        torch=self.torch
        with torch.inference_mode():
            x=torch.from_numpy(a).to(self.device);y=torch.from_numpy(b).to(self.device)
            lp=self.lp(x,y,normalize=True);di=self.di(x,y,require_grad=False,batch_average=False)
            return lp.detach().cpu().numpy(),di.detach().cpu().numpy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu')
    p.add_argument('--gpu-uuid');p.add_argument('--batch-size',type=int,default=1)
    a=p.parse_args()
    if not os.environ.get('PS_CODE_REVISION'):raise ValueError('frozen PS_CODE_REVISION receipt required')
    manifest=json.loads(a.manifest.read_text());weights=json.loads(a.weights.read_text())
    source,source_receipt,inputs=load_inputs(a.source,manifest)
    engine=OfficialEngine(weights,a.device,a.gpu_uuid)
    rows=[{'method':method,**receipt,**score_arrays(source,rgb,engine,a.batch_size)} for method,rgb,receipt in inputs]
    result={'registration':REGISTRATION,'manifest':file_identity(a.manifest),'weights_manifest':file_identity(a.weights),
        'registration_sha256':sha_bytes(json.dumps(REGISTRATION,sort_keys=True,separators=(',',':')).encode()),
        'batch_size':a.batch_size,'source':source_receipt,'models':engine.provenance,'rows':rows,'code_revision':os.environ.get('PS_CODE_REVISION'),
        'scorer_file':file_identity(__file__),'environment':{'python':sys.version,'executable':sys.executable,
            'platform':platform.platform(),'numpy_version':np.__version__,'command':sys.argv}}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
