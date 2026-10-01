"""Common scoring input, aggregation and official-call contracts, no downloads."""
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.packet_study import perceptual_score as p


def test_common_float_quantization_same_contract_and_no_crop():
    shape=(2,3,4,3)
    raw=np.zeros((2,3,3,4),np.float32);raw[:,0]=.5;raw[:,1]=1
    rgb=p.common_rgb(raw,'float32_nchw_rgb01',shape=shape)
    assert rgb.shape==shape and rgb.dtype==np.uint8
    assert np.all(rgb[...,0]==128) and np.all(rgb[...,1]==255) and np.all(rgb[...,2]==0)
    assert np.array_equal(p.common_rgb(rgb,'uint8_nhwc_rgb',shape=shape),rgb)
    with pytest.raises(ValueError,match='complete'):p.common_rgb(rgb,'uint8_nhwc_rgb')


@pytest.mark.parametrize('bad',[np.nan,np.inf,-.01,1.01])
def test_nonfinite_or_out_of_range_float_rejected(bad):
    raw=np.zeros((2,3,3,4),np.float32);raw[0,0,0,0]=bad
    with pytest.raises(ValueError,match='finite'):p.common_rgb(raw,'float32_nchw_rgb01',shape=(2,3,4,3))


def test_wrong_dtype_layout_and_encoding_rejected():
    with pytest.raises(ValueError):p.common_rgb(np.zeros((2,3,3,4),np.float64),'float32_nchw_rgb01',shape=(2,3,4,3))
    with pytest.raises(ValueError):p.common_rgb(np.zeros((2,3,4,3),np.float32),'float32_nchw_rgb01',shape=(2,3,4,3))
    with pytest.raises(ValueError):p.common_rgb(np.zeros((2,3,4,3),np.uint8),'guess',shape=(2,3,4,3))


@pytest.fixture
def inputs(tmp_path,monkeypatch):
    shape=(2,3,4,3);monkeypatch.setattr(p,'SHAPE',shape)
    # common_rgb's production default shape is frozen; replace only in this
    # small file-I/O fixture, preserving the tested converter semantics.
    common=p.common_rgb
    monkeypatch.setattr(p,'common_rgb',lambda array,encoding:common(array,encoding,shape=shape))
    source=np.zeros(shape,np.uint8);monkeypatch.setattr(p,'SOURCE_RGB_SHA256',p.sha_bytes(source.tobytes()))
    src=tmp_path/'source.npy';recon=tmp_path/'recon.npy';np.save(src,source);np.save(recon,source)
    manifest={'schema':1,'source':p.file_identity(src),'reconstructions':[{
        **p.file_identity(recon),'method':'codec','frame_count':2,'encoding':'uint8_nhwc_rgb'}]}
    return src,manifest,recon


def test_explicit_file_and_pixel_identity_checked(inputs):
    src,manifest,recon=inputs
    source,receipt,rows=p.load_inputs(src,manifest)
    assert receipt['rgb_sha256']==p.SOURCE_RGB_SHA256 and len(rows)==1
    manifest['reconstructions'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='SHA256'):p.load_inputs(src,manifest)


def test_missing_frame_source_hash_and_duplicate_method_fail_closed(inputs,monkeypatch):
    src,manifest,recon=inputs
    manifest['reconstructions'][0]['frame_count']=1
    with pytest.raises(ValueError,match='all48'):p.load_inputs(src,manifest)
    manifest['reconstructions'][0]['frame_count']=2
    manifest['reconstructions'].append(dict(manifest['reconstructions'][0]))
    with pytest.raises(ValueError,match='unique'):p.load_inputs(src,manifest)
    manifest['reconstructions'].pop()
    monkeypatch.setattr(p,'SOURCE_RGB_SHA256','0'*64)
    with pytest.raises(ValueError,match='frozen'):p.load_inputs(src,manifest)


def test_actual_missing_reconstruction_and_partial_raster_fail(inputs):
    src,manifest,recon=inputs
    np.save(recon,np.zeros((1,3,4,3),np.uint8))
    manifest['reconstructions'][0].update(p.file_identity(recon))
    with pytest.raises(ValueError,match='complete'):p.load_inputs(src,manifest)
    recon.unlink()
    with pytest.raises(FileNotFoundError):p.load_inputs(src,manifest)


def test_all48_full_frame_batches_and_distances():
    source=np.zeros(p.SHAPE,np.uint8);decoded=np.full(p.SHAPE,20,np.uint8);calls=[]
    def engine(a,b):
        calls.append(a.shape)
        assert a.dtype==np.float32 and a.shape[1:]==(3,360,640)
        assert np.allclose(b,20/255)
        return np.arange(len(a),dtype=np.float32),np.full(len(a),.25)
    scores=p.score_arrays(source,decoded,engine,batch_size=7)
    assert len(scores['lpips_per_frame'])==len(scores['dists_per_frame'])==48
    assert calls[-1][0]==6 and len(calls)==7
    assert scores['dists_mean']==.25 and scores['pooled_mse_y']==400
    assert scores['pooled_y_psnr_db']==pytest.approx(10*np.log10(255**2/400))


@pytest.mark.parametrize('mode',['missing','nan'])
def test_metric_partial_or_nonfinite_results_rejected(mode):
    source=np.zeros(p.SHAPE,np.uint8)
    def engine(a,b):return ([0]*(len(a)-1) if mode=='missing' else [np.nan]*len(a)),[0]*len(a)
    with pytest.raises(ValueError,match='every finite'):p.score_arrays(source,source,engine,batch_size=8)


def test_zero_mse_explicit_and_bt601_existing_uint8_contract():
    source=np.zeros((2,1,1,3),np.uint8);decoded=source.copy();decoded[0,0,0]=[255,0,0]
    metric=p.luma_quality(source,decoded)
    assert metric['mse_per_frame_y']==[76**2,0]  # .299*255 truncates, does not round.
    assert metric['mean_frame_y_psnr_db'] is None
    exact=p.luma_quality(source,source)
    assert exact['pooled_mse_y']==0 and exact['pooled_y_psnr_db'] is None


def test_official_forward_arguments_and_input_range_without_model_downloads():
    calls=[]
    class Tensor:
        def __init__(self,array):self.array=array
        def to(self,device):return self
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.array
    def model(name):
        def call(x,y,**kwargs):
            calls.append((name,kwargs,x.array.shape,float(x.array.max())))
            return Tensor(np.zeros(len(x.array),np.float32))
        return call
    engine=p.OfficialEngine.__new__(p.OfficialEngine)
    engine.device='cpu';engine.torch=SimpleNamespace(from_numpy=Tensor,inference_mode=contextlib.nullcontext)
    engine.lp=model('lpips');engine.di=model('dists')
    lp,di=engine(np.zeros((1,3,360,640),np.float32),np.ones((1,3,360,640),np.float32))
    assert calls[0][1]=={'normalize':True}
    assert calls[1][1]=={'require_grad':False,'batch_average':False}
    assert lp.shape==di.shape==(1,)


def test_missing_or_wrong_declared_weight_file_rejected(tmp_path):
    weight=tmp_path/'weight';weight.write_bytes(b'weight')
    with pytest.raises(ValueError,match='explicit'):p.checked_file({'path':str(weight)})
    with pytest.raises(ValueError,match='SHA256'):p.checked_file({'path':str(weight),'sha256':'0'*64})
