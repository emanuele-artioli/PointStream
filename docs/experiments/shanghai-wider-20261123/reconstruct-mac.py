import argparse,json,sys
from pathlib import Path
import cv2
sys.path.insert(0,'docs/experperiments')
sys.path.insert(0,'docs/experiments/test-crop-joint-20261122')
import build_joint as b
from src.pointstream import ball
root=Path('/private/tmp/ps-shanghai-clean-inputs-20261123')
out=Path('/private/tmp/ps-shanghai-clean-result-20261123')
out.mkdir(exist_ok=False)
a=argparse.Namespace(out=out,source=root/'match102-000-fc5d3ee431f8.mp4',features=root/'published.tar',ball=root/'ball.payload')
r=json.loads((b.HERE/'match102-result.json').read_text()); old=json.loads((b.HERE/'match102-sender.json').read_text())
assert b.sha(a.source)==r['source_sha256'] and b.sha(a.features)==r['prior_feature_archive_sha256']
assert b.sha(a.ball)=='3ad726835c0e51c11daa80d9b54630a1e52494227680b434254e583950426918'
assert cv2.__version__==old['source_opencv']
cv2.setNumThreads(1)
p,app=ball.decode_frames(a.ball.read_bytes(),240,960,544)
src=b.raster_source(a); masks=b.spatial_masks(a,r,p)
d=b.donor_base(a,src,masks,smoke_only=False)
(out/'receipt.json').write_text(json.dumps(d,indent=2))
for key in ('smoke','donors','holes','painted_area','mask_sha256','clean_yuv_sha256'):
 assert d[key]==old['donor'][key], key
print('EXACT', b.sha(out/'clean.yuv'))
