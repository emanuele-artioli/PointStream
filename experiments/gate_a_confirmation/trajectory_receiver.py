"""Explicit packet-controlled mask adaptation of the pinned legacy receiver."""
import argparse, hashlib, importlib.util, io, json, pathlib, sys, os

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-receiver',required=True);p.add_argument('--legacy-root',required=True);p.add_argument('--package',required=True);p.add_argument('--output',required=True);a,rest=p.parse_known_args()
    base=pathlib.Path(a.base_receiver);assert hashlib.sha256(base.read_bytes()).hexdigest()=='45d68968b9d8263569021470513a45ca9a6651f141ba38e658c5ce709d8ebbc7'
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,a.legacy_root)
    import numpy as np
    with np.load(a.package,allow_pickle=False) as packet:meta=json.loads(packet['metadata'].tobytes())
    policy=meta['mask_policy']
    if policy not in ['opaque','alpha']:raise ValueError('unsupported packet mask policy')
    import src.runner.client as client
    original=client.composite_clip
    def composite(*args,**kwargs):
        kwargs['use_heuristic_mask']=policy=='opaque'
        return original(*args,**kwargs)
    client.composite_clip=composite
    spec=importlib.util.spec_from_file_location('pinned_base_receiver',base);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.argv=[str(base),*rest,'--legacy-root',a.legacy_root,'--package',a.package,'--output',a.output];module.main()
    receipt=pathlib.Path(a.output).with_suffix('.receipt.json');r=json.loads(receipt.read_text());r['packet_mask_policy']=policy;r['adaptation']='packet-controlled alpha/opaque compositing; pinned legacy native background/appearance decoder';receipt.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
