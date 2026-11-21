"""Use the validated player worker under an exclusive fleet GPU claim."""
import os
from experiments.tennis import player_requalification

if __name__ == '__main__':
    if 'PS_GPU_SHARING' not in os.environ and os.environ.get('PS_CLAIMED_GPU_UUID'):
        # Exclusive claims do not set the shared-claim memory cap. Retain the
        # original job's declared 18,000 MiB allocator cap and validation.
        os.environ.setdefault('PS_GPU_MEMORY_LIMIT_MIB', '18000')
    player_requalification.main()
