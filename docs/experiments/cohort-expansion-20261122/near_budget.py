"""Choose integer neural quantizers solely from actual native packet rates."""

import math


def choose(prices, cap=20.4):
    pairs = sorted((int(q), float(rate)) for q, rate in prices.items())
    assert len(pairs) >= 2 and all(rate > 0 and 0 <= q <= 63 for q, rate in pairs)
    below = [(q, rate) for q, rate in pairs if rate <= cap]
    over = [(q, rate) for q, rate in pairs if rate > cap]
    if below and 1 - max(rate for _, rate in below) / cap <= 0.02:
        return None
    if below and over:
        lower = max(below, key=lambda x: x[1])
        upper = min(over, key=lambda x: x[1])
        lo, hi = lower[0] + 1, upper[0] - 1
    elif below:
        lower, upper = sorted(below, key=lambda x: x[1])[-2:]
        lo, hi = max(q for q, _ in below) + 1, 63
    else:
        lower, upper = sorted(over, key=lambda x: x[1])[:2]
        lo, hi = 0, min(q for q, _ in over) - 1
    possible = [q for q in range(lo, hi + 1) if q not in prices]
    if not possible:
        return None
    est = lower[0] + (upper[0] - lower[0]) * math.log(cap / lower[1]) / math.log(
        upper[1] / lower[1]
    )
    return min(possible, key=lambda q: (abs(q - est), q))
