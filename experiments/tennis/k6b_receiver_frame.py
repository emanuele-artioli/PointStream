"""Frozen every-frame fresh-prompt SAM; no temporal mask propagation."""
from experiments.tennis.k6b_receiver_sam import main

if __name__ == "__main__":
    raise SystemExit(main(independent_default=True))
