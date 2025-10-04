
# omega/controls.py
from __future__ import annotations
from typing import List, Optional
import random

def global_shuffle(seq: List[str], seed: int = 0) -> List[str]:
    """Return a globally shuffled copy of seq (destroys all order)."""
    rng = random.Random(seed)
    out = list(seq)
    rng.shuffle(out)
    return out

def block_shuffle(seq: List[str], block_len: int = 50, seed: int = 0) -> List[str]:
    """
    Shuffle contiguous blocks of length block_len.
    Preserves within-block order; destroys long-range structure.
    """
    if block_len <= 1:
        return global_shuffle(seq, seed=seed)
    n = len(seq)
    blocks = [seq[i:i+block_len] for i in range(0, n, block_len)]
    rng = random.Random(seed)
    rng.shuffle(blocks)
    return [tok for b in blocks for tok in b]

def permute_labels(labels: List[str], seed: int = 0) -> List[str]:
    """Histogram-preserving permutation of labels."""
    rng = random.Random(seed)
    idx = list(range(len(labels)))
    rng.shuffle(idx)
    out = [None] * len(labels)
    for i, j in enumerate(idx):
        out[i] = labels[j]
    return out

# Simple dissipation proxy for SCR: LZMA compressed size (bytes) of the token stream
def lzma_energy(tokens: List[str]) -> int:
    import lzma
    data = (" ".join(tokens)).encode("utf-8", errors="ignore")
    comp = lzma.compress(data, preset=3)
    return len(comp)
