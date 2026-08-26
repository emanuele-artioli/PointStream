"""`python -m experiments.tier` — run the shipped tier configs on one clip."""

from __future__ import annotations

import sys

from experiments.tier.run import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
