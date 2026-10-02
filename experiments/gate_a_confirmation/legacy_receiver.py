"""Fresh-process receiver. Intentionally imports only pinned legacy receiver code."""
import argparse, hashlib, json, os, pathlib, sys, tempfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-root',required=True);p.add_argument('--package',required=True);p.add_argument('--output',required=True);p.add_argument('--deny-root',action='append',default=[]);a=p.parse_args()
    root=pathlib.Path(a.legacy_root).resolve(); package=pathlib.Path(a.package).resolve(); output=pathlib.Path(a.output).resolve()
    sys.path.insert(0,str(root)); os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from src.runner.client import reconstruct_serialized_client
    denied=[pathlib.Path(x).resolve() for x in a.deny_root]
    violations=[]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=pathlib.Path(os.fsdecode(args[0])).resolve()
            if path not in (package, output, output.with_suffix('.receipt.json')) and any(path==d or d in path.parents for d in denied):
                violations.append(str(path));raise PermissionError('receiver forbidden input: '+str(path))
    sys.addaudithook(audit)
    payload=package.read_bytes(); clip=reconstruct_serialized_client(payload)
    frames=np.asarray(clip,dtype=np.uint8); np.save(output,frames,allow_pickle=False)
    receipt={'package_sha256':hashlib.sha256(payload).hexdigest(),'package_bytes':len(payload),'frames_shape':list(frames.shape),'frames_rgb_sha256':hashlib.sha256(np.ascontiguousarray(frames).data).hexdigest(),'denied_roots':[str(x) for x in denied],'violations':violations,'isolation':'fresh Python process with open audit denial; native OS isolation not asserted'}
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
