"""Compare paid PNG and candidate WebP pixels with the campus receiver OpenCV."""
import base64
import hashlib
import json
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from experiments.tennis import common_player_packet as packet

ROOT = Path('docs/experiments/lossless-reference-codecs-20261118')
CARRIER = Path('/Users/manu/Documents/PointStream-prepared/2026-11-18-receiver-replay-zlib/qpng.pscp')
PYTHON = '/local/users/emanuele/pointstream/cache/bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e/tree/bin/python'
WORKER = '''import base64,hashlib,json,os,sys,time
start=time.process_time()
import cv2,numpy as np
cv2.setNumThreads(1)
r=json.load(sys.stdin)
def sha(x): return hashlib.sha256(x).hexdigest()
def rgb(blob):
    a=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
    if a is None: raise ValueError('image decode failed')
    return a[:,:,::-1]
rows=[]
for item in r:
    png=base64.b64decode(item['png'],validate=True)
    webp=base64.b64decode(item['webp'],validate=True)
    if sha(webp)!=item['webp_sha256']: raise ValueError('candidate sha')
    a,b=rgb(png),rgb(webp)
    if a.shape != (784,512,3) or b.shape != a.shape or not np.array_equal(a,b): raise ValueError('pixel mismatch')
    rows.append(dict(unit=item['unit'],png_sha256=sha(png),webp_sha256=sha(webp),rgb_sha256=sha(a.tobytes()),equal=True,shape=list(a.shape)))
info=cv2.getBuildInformation()
print(json.dumps(dict(python=sys.executable,opencv=cv2.__version__,numpy=np.__version__,opencv_extension=cv2.__file__,opencv_build_sha256=sha(info.encode()),opencv_build_webp=[x.strip() for x in info.splitlines() if 'WEBP:' in x],results=rows,cpu_seconds=time.process_time()-start,cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))))
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    start = time.monotonic()
    receipt = json.loads((ROOT / 'result.json').read_text())
    data = CARRIER.read_bytes()
    if sha(data) != receipt['carrier_sha256']:
        raise ValueError('carrier identity')
    sections, _, _ = packet.unpack(data)
    expected = {r['section']: r for r in receipt['references']}
    request = []
    with tempfile.TemporaryDirectory(prefix='ps-webp-linux-', dir='/private/tmp') as tmp:
        for ref in ('ref0', 'ref1'):
            meta, png = packet._bounded_transport(sections[ref], b'REF1', packet.MAX_REF)
            source, output = Path(tmp) / f'{ref}.png', Path(tmp) / f'{ref}.webp'
            source.write_bytes(png)
            subprocess.run(['/opt/homebrew/bin/cwebp', '-quiet', '-lossless', '-m', '6', '-metadata', 'none', str(source), '-o', str(output)], check=True, timeout=30)
            webp = output.read_bytes()
            expected_sha = expected[ref]['candidates']['webp']['payload_sha256']
            if sha(webp) != expected_sha:
                raise ValueError('candidate image identity')
            request.append({'unit': meta['unit'], 'png': base64.b64encode(png).decode(),
                            'webp': base64.b64encode(webp).decode(), 'webp_sha256': expected_sha})
    command = f'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {shlex.quote(PYTHON)} -c {shlex.quote(WORKER)}'
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'gpu5', command],
                            input=json.dumps(request), capture_output=True, text=True, timeout=120, check=True)
    report = json.loads(result.stdout)
    expected_build = json.loads(Path('docs/experiments/joint-readiness-20261118/linux-jpeg.json').read_text())
    report['same_python'] = report['python'] == expected_build['python']
    report['same_opencv_build'] = report['opencv_build_sha256'] == expected_build['opencv_build_sha256']
    report['controller_elapsed_seconds'] = time.monotonic() - start
    report['scope'] = 'installed library parity; not a serialized WebP packet or isolated receiver'
    (ROOT / 'linux-preflight-result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
