"""Shared-Ada W3 with enforced parent/child allocator limits and receipts."""
import json
import os
from pathlib import Path
import sys
import tarfile

from experiments.tennis import w3_neural_baseline as w3


def main():
    review = '--review12' in sys.argv
    if review:
        sys.argv.remove('--review12')
        from experiments.tennis import w3_publish_review12 as review12
    if '--validate' in sys.argv:
        w3.main()
        if review:
            review12.check_frames()
        archive = Path(os.environ['PS_STAGE_DIR']) / 'published.tar'
        with tarfile.open(archive) as tar:
            resources = json.load(tar.extractfile('publish/shared-resource-receipts.json'))
        if len(resources) < 3 or any(r['budget_mib'] != 6144 or r['action'] not in ('encode', 'decode') for r in resources):
            raise ValueError('shared child allocator receipt validation')
        return
    import torch
    if os.environ.get('PS_GPU_SHARING') != 'shared' or os.environ.get('PS_GPU_MEMORY_LIMIT_MIB') != '10240':
        raise ValueError('shared fleet claim mismatch')
    if torch.cuda.memory_allocated() != 0:
        raise ValueError('parent allocator already initialized')
    torch.cuda.set_per_process_memory_fraction(4096 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    receipts = scratch / 'shared-resource-receipts'
    receipts.mkdir()
    os.environ['PS_K5D_SHARED_NUMERICAL'] = 'pointstream-parent4-child6-v1'
    os.environ['PS_K5D_RESOURCE_RECEIPTS'] = str(receipts)
    if review:
        w3.code_point = review12.record_frames
    w3.main()
    rows = [json.loads(p.read_text()) for p in receipts.glob('child-*.json')]
    if len(rows) < 3 or any(r['budget_mib'] != 6144 for r in rows):
        raise ValueError('DCVC subprocess allocator receipts missing')
    (scratch / 'publish' / 'shared-resource-receipts.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
