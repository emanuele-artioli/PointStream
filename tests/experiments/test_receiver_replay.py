import numpy as np
import pytest
from experiments.tier.receiver_replay import psnr

def test_exact_frame_denominator():
    source = np.full((3,4,5,3),100,dtype=np.uint8)
    with pytest.raises(ValueError,match='incomplete output'):
        psnr(source, source[:2])

def test_mean_frame_and_pooled_are_distinct():
    source=np.full((2,4,5,3),100,dtype=np.uint8)
    decoded=source.copy();decoded[0]+=3;decoded[1]+=10
    result=psnr(source,decoded)
    assert result['frames']==2
    assert result['mean_frame_y_psnr_db']>result['pooled_y_psnr_db']

def test_source_cache_read_rejected_and_package_allowed(tmp_path):
    from experiments.tier.receiver_replay import receiver_access_guard
    data=tmp_path/'data';code=tmp_path/'code'
    package=data/'copied/package.npz';output=data/'copied/decoded.npy'
    hook,reads,commands=receiver_access_guard(package,output,data,code)
    hook('open',(str(package),'rb',0))
    with pytest.raises(PermissionError,match='receiver denied'):
        hook('open',(str(data/'outputs/source.npy'),'rb',0))
    hook('open',(str(code/'src/client.py'),'rb',0))
    assert len(reads)==2
