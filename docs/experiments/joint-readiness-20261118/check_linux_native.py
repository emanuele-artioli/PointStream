"""Validate the finish-side native decode on campus CPU before GPU masking."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root/'source'))
from experiments.tennis import common_receiver_isolation as isolation  # noqa: E402
from experiments.tennis import common_receiver_manifest as manifest  # noqa: E402

identity = json.loads((root/'linux-carriers.json').read_text())['carriers']['40']['sha256']
paths = {'carrier':root/'q40.pscp', 'manifest':root/'q40.manifest',
         'ffmpeg':Path('/local/users/emanuele/pointstream/cache')/manifest.SAM_ENV/'tree/bin/ffmpeg'}
receipt = isolation.launch(source_root=root/'source', output=root/'native-out', inputs=list(paths.values()),
                           worker_config={**{k:str(v) for k,v in paths.items()},'carrier_sha256':identity},
                           denied=[Path(v) for v in json.loads((root/'denied.json').read_text())], seconds=90, kind='native')
print(receipt['native_yuv_sha256'], receipt['native_bytes'], receipt['frame_count'])
