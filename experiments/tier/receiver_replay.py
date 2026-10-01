"""Bounded source-hidden fresh-process replay of retained E06 packages.

Python audit hooks guard all reads below the external data root. Native
subprocess argv is recorded; this is a reproducibility boundary, not an OS sandbox.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receiver_access_guard(package, output, data_root, code_root):
    allowed = {Path(package).resolve(), Path(output).resolve()}
    code_root = Path(code_root).resolve()
    reads, commands = [], []
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(Path(data_root).resolve()):
                reads.append(str(p))
                if p not in allowed and not p.is_relative_to(code_root):
                    raise PermissionError(f'receiver denied external data read: {p}')
        if event == 'subprocess.Popen':
            commands.append(args[1])
    return audit, reads, commands


def hidden_receiver(package, output, data_root):
    audit, reads, commands = receiver_access_guard(package, output, data_root, Path(__file__).resolve().parents[2])
    sys.addaudithook(audit)
    from experiments.tier.e06_transport import reconstruct_standalone
    frames = reconstruct_standalone(Path(package).read_bytes())
    np.save(output, frames, allow_pickle=False)
    print(json.dumps({'shape': list(frames.shape), 'pixels_sha256': hashlib.sha256(frames.tobytes()).hexdigest(), 'external_opens': reads, 'native_commands': commands}))


def psnr(source, decoded):
    if source.shape != decoded.shape:
        raise ValueError(f'incomplete output {decoded.shape} expected {source.shape}')
    # Original E06 headline uses 8-bit BT.601 luma; keep mean frame dB separate.
    from src.components.codec.frames import rgb_to_luma
    a, b = rgb_to_luma(source), rgb_to_luma(decoded)
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2, axis=(1,2))
    if np.any(mse <= 0):
        raise ValueError('zero-MSE frame needs explicit infinity handling')
    return {'frames':len(mse),'frame_mse_y':mse.tolist(),'mean_frame_y_psnr_db':float(np.mean(10*np.log10(255**2/mse))), 'pooled_y_psnr_db':float(10*np.log10(255**2/np.mean(mse)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--child-package', type=Path)
    p.add_argument('--child-output', type=Path)
    args=p.parse_args()
    if args.child_package:
        hidden_receiver(args.child_package,args.child_output,args.data_root);return
    os.nice(19)
    affinity=sorted(os.sched_getaffinity(0))[:2];os.sched_setaffinity(0,affinity)
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    if os.getloadavg()[0] > 32:
        raise RuntimeError('host load outside bounded CPU policy')
    args.out.mkdir(parents=True,exist_ok=False)
    root=args.data_root/'outputs/evaluation-20260914'
    source=root/'e03b/run-20260916-federer007/prepared_rgb.npy'
    families=[('original','run-20260916-federer007-perframe-bbox','transport.npz'),('compact','audit-20260916-lossless-pack','transport_compact.npz'),('floor','probe-20260917-floor-arms','transport_floor.npz')]
    candidates=[(family,path) for family,directory,name in families for path in sorted((root/'e06'/directory).glob('*/'+name))]
    if len(candidates) != 12 or any(sum(f == family for f, _ in candidates) != 4 for family, _, _ in families):
        raise ValueError("expected all twelve retained packages (four per family)")
    if args.smoke:candidates=candidates[:1]
    report={'host':socket.getfqdn(),'python':sys.version,'worker_sha256':digest(__file__),'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() if (Path.cwd()/'.git').exists() else os.environ.get('PS_CODE_REVISION'),'source':str(source),'source_file_sha256':digest(source),'cpu_affinity':affinity,'gpu_allocated':False,'smoke':args.smoke,'rows':[]}
    for family,path in candidates:
        if os.getloadavg()[0] > 32:raise RuntimeError('host load exceeded policy')
        # Source arrays enter the scorer only, after the receiver has exited.
        isolated=args.out/(family+'-'+path.parent.name);isolated.mkdir()
        package=isolated/'package.npz';shutil.copyfile(path,package)
        output=isolated/'decoded.npy'
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='ps-hidden-') as cwd:
            command=[sys.executable,'-m','experiments.tier.receiver_replay','--data-root',str(args.data_root),'--out',str(args.out),'--child-package',str(package),'--child-output',str(output)]
            child=subprocess.run(command,cwd=cwd,env=env,text=True,capture_output=True,timeout=180)
        row={'family':family,'package':str(path),'package_sha256':digest(package),'complete_file_bytes':package.stat().st_size,'returncode':child.returncode,'stderr':child.stderr,'command':command}
        if child.returncode==0:
            row['receiver']=json.loads(child.stdout.strip().splitlines()[-1]);row['quality']=psnr(np.load(source,allow_pickle=False),np.load(output,allow_pickle=False))
        report['rows'].append(row)
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if child.returncode:raise RuntimeError('receiver instrument failed; preserve report')
    print(args.out/'report.json')

if __name__=='__main__':main()
