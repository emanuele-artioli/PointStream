"""Measure the frozen four-bit adapter-plus-guider charge from a saved fit."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from experiments.tennis import k6b_fit, w7_select


def bill(archive: Path, checkpoint: str, guider: Path, archive_sha: str, guider_sha: str) -> dict[str, Any]:
    if w7_select.file_sha256(archive) != archive_sha or w7_select.file_sha256(guider) != guider_sha:
        raise ValueError('billing input sha256 mismatch')
    import torch
    with tarfile.open(archive) as tar:
        member = tar.extractfile(f'publish/{checkpoint}')
        if member is None:
            raise ValueError('adapter checkpoint missing')
        payload = member.read()
        state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    base = torch.load(guider, map_location='cpu', weights_only=True)
    # Match apply_fit's exact pre-quantization conversions and original guider.
    converted = {**state, **{k: {n: v.float() for n, v in state[k].items()}
                            for k in ('reference_unet', 'denoising_unet')}}
    _, billing = k6b_fit.quantize_adapters(converted, 4, {k: v.detach().cpu() for k, v in base.items()})
    if billing['bits'] != 4 or not billing['pose_guider_billed'] or billing['pose_guider_bytes'] <= 0 or billing['bytes'] <= 0:
        raise ValueError('invalid adapter/guider billing')
    return {'archive_sha256': w7_select.file_sha256(archive), 'checkpoint': checkpoint,
            'checkpoint_sha256': hashlib.sha256(payload).hexdigest(),
            'guider_sha256': w7_select.file_sha256(guider), 'billing': billing}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--guider', type=Path, required=True)
    p.add_argument('--archive-sha', required=True)
    p.add_argument('--guider-sha', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    result = bill(a.archive, a.checkpoint, a.guider, a.archive_sha, a.guider_sha)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print('four-bit adapter+guider bytes', result['billing']['bytes'], flush=True)


if __name__ == '__main__':
    main()
