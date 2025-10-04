# alphabet/bootstrap.py

"""
Moving-block bootstrap:
- Resamples *indices* in contiguous blocks to respect local correlation.
- Mean and [2.5%, 97.5%] quantiles are empirical percentiles over bootstrap draws.
- p_le_0 is the empirical Pr(mean <= 0) under the bootstrap measure (one-sided).
"""

from __future__ import annotations
from typing import Callable, List, Dict, Optional
import math, random

def block_bootstrap_mean(
    x: List[float],
    B: int = 200,
    block_len: int = 128,
    rng: Optional[random.Random] = None,
) -> Dict[str, float]:
    """
    Moving-block bootstrap for a single series x (e.g., per-token deltas).
    Returns mean, 95% CI, and p(mean <= 0) under the bootstrap.
    """
    assert len(x) > 0
    if rng is None: rng = random.Random(0)
    n = len(x)
    L = max(1, min(block_len, n))
    m = math.ceil(n / L)

    draws = []
    for _ in range(B):
        idx = []
        
        if (_ + 1) % 50 == 0:   # every 50 iterations, tune to taste
            print(f"{_+1}/{B} ", flush=True, end="" )        
        
        
        
        for _ in range(m):
            s = rng.randrange(0, n)
            idx.extend(((s + t) % n) for t in range(L))
        idx = idx[:n]
        draws.append(sum(x[i] for i in idx) / n)

    draws.sort()
    mean = sum(draws) / B
    lo_idx = max(0, int(math.floor(0.025 * B)) - 1)
    hi_idx = min(B - 1, int(math.ceil(0.975 * B)) - 1)
    lo95, hi95 = draws[lo_idx], draws[hi_idx]
    p_le_0 = sum(1 for d in draws if d <= 0.0) / B
    return {"mean": mean, "lo95": lo95, "hi95": hi95, "p_le_0": p_le_0, "B": B, "block_len": L}



def paired_block_bootstrap(
    a: List[float],
    b: List[float],
    B: int = 200,
    block_len: int = 128,
    stat_fn: Optional[Callable[[List[float], List[float]], float]] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, float]:
    """
    Moving-block bootstrap for paired sequences (same length).
    Resamples *indices* in blocks and applies them to both lists to preserve pairing.
    Returns mean, 95% CI, and p(Δ<=0).
    """
    assert len(a) == len(b) and len(a) > 0, "paired sequences must be same nonzero length"
    if rng is None: rng = random.Random(0)
    n = len(a)
    L = max(1, min(block_len, n))
    m = math.ceil(n / L)

    def default_stat(x, y):
        # mean difference (y - x)
        return (sum(y) / len(y)) - (sum(x) / len(x))

    stat = stat_fn or default_stat

    draws = []
    for _ in range(B):
        idx: List[int] = []
        for _ in range(m):
            s = rng.randrange(0, n)               # start index of block
            idx.extend(((s + t) % n) for t in range(L))
        idx = idx[:n]                              # trim to exact length
        xa = [a[i] for i in idx]
        yb = [b[i] for i in idx]
        draws.append(stat(xa, yb))

    draws.sort()
    mean = sum(draws) / B
    # empirical quantiles
    lo_idx = max(0, int(math.floor(0.025 * B)) - 1)
    hi_idx = min(B - 1, int(math.ceil(0.975 * B)) - 1)
    lo95 = draws[lo_idx]
    hi95 = draws[hi_idx]
    p_le_0 = sum(1 for d in draws if d <= 0.0) / B

    return {"mean": mean, "lo95": lo95, "hi95": hi95, "p_le_0": p_le_0, "B": B, "block_len": L}
