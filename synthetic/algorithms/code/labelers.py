# alphabet/labelers.py
from __future__ import annotations
from typing import List, Tuple, Optional
import random, math
from collections import Counter, defaultdict

def _hashed_bow(tokens: List[str], windows: List[Tuple[int,int]], D: int = 4096) -> List[List[float]]:
    """Hashed bag-of-words for each window; returns dense lists length D."""
    vecs = []
    for (a,b) in windows:
        v = [0.0]*D
        for t in tokens[a:b]:
            h = (hash(t) & 0x7fffffff) % D
            v[h] += 1.0
        # L2 normalize to avoid length bias
        s = math.sqrt(sum(x*x for x in v)) or 1.0
        vecs.append([x/s for x in v])
    return vecs

def _kmeans(X: List[List[float]], K: int, iters: int = 10, rng: Optional[random.Random] = None) -> Tuple[List[List[float]], List[int]]:
    """Tiny k-means with cosine distance (via normalized vectors)."""
    if rng is None: rng = random.Random(0)
    n = len(X); d = len(X[0])
    # init: pick K random points
    idx = list(range(n)); rng.shuffle(idx)
    C = [X[i][:] for i in idx[:K]]  # centroids
    assign = [0]*n
    for _ in range(iters):
        changed = 0
        # assign
        for i in range(n):
            xi = X[i]
            bestk, best = 0, -1e9
            for k in range(K):
                ck = C[k]
                # cosine similarity (both L2 normalized): dot
                s = 0.0
                for j in range(d):
                    s += xi[j]*ck[j]
                if s > best:
                    best, bestk = s, k
            if assign[i] != bestk:
                assign[i] = bestk; changed += 1
        if changed == 0: break
        # update
        sums = [[0.0]*d for _ in range(K)]
        counts = [0]*K
        for i in range(n):
            k = assign[i]
            counts[k] += 1
            xi = X[i]
            for j in range(d): sums[k][j] += xi[j]
        for k in range(K):
            if counts[k] == 0: continue
            s = math.sqrt(sum(v*v for v in sums[k])) or 1.0
            C[k] = [v/s for v in sums[k]]
    return C, assign

def _sliding_windows(n: int, win: int, stride: int) -> List[Tuple[int,int]]:
    ws = []
    i = 0
    while i < n:
        a = i
        b = min(n, i+win)
        ws.append((a,b))
        i += stride
    if ws and ws[-1][1] < n:
        ws.append((max(0, n-win), n))
    return ws

def _assign_tokens_by_nearest_window(n_tokens: int, windows: List[Tuple[int,int]]) -> List[int]:
    """For each token index, pick the index of the covering/nearest window."""
    # Precompute centers for distance
    centers = [ (a+b)//2 for (a,b) in windows ]
    out = [0]*n_tokens
    for i in range(n_tokens):
        # nearest center
        bestk, bestd = 0, 1<<30
        for k, c in enumerate(centers):
            d = abs(i - c)
            if d < bestd:
                bestd, bestk = d, k
        out[i] = bestk
    return out

def make_topic_labels(
    train_tokens: List[str],
    test_tokens:  List[str],
    win: int = 100,
    stride: int = 50,
    K: int = 16,
    D: int = 4096,
    iters: int = 10,
    rng: Optional[random.Random] = None
) -> Tuple[List[str], List[str]]:
    """
    Build 'slow' topic labels by clustering hashed bag-of-words over sliding windows on TRAIN.
    TEST windows are projected onto TRAIN centroids. Label at position t is the id of its window.
    Returns (Z_train, Z_test) as strings like 'T7'.
    """
    if rng is None: rng = random.Random(0)
    n_tr = len(train_tokens); n_te = len(test_tokens)
    w_tr = _sliding_windows(n_tr, win, stride)
    w_te = _sliding_windows(n_te, win, stride)
    X_tr = _hashed_bow(train_tokens, w_tr, D=D)
    C, assign_tr = _kmeans(X_tr, K=K, iters=iters, rng=rng)

    # project TEST windows to nearest centroid (cosine via dot)
    d = len(C[0])
    X_te = _hashed_bow(test_tokens, w_te, D=D)
    assign_te = []
    for x in X_te:
        bestk, best = 0, -1e9
        for k in range(K):
            s = 0.0
            ck = C[k]
            for j in range(d): s += x[j]*ck[j]
            if s > best: best, bestk = s, k
        assign_te.append(bestk)

    # per-token label via nearest window
    win_of_tok_tr = _assign_tokens_by_nearest_window(n_tr, w_tr)
    win_of_tok_te = _assign_tokens_by_nearest_window(n_te, w_te)
    Z_train = [f"T{assign_tr[win_of_tok_tr[i]]}" for i in range(n_tr)]
    Z_test  = [f"T{assign_te[win_of_tok_te[i]]}" for i in range(n_te)]
    return Z_train, Z_test


# --- Supervised lag labels ---------------------------------------------------
def make_lag_labels(train_tokens, test_tokens, lag: int = 8, null: str = "L-1"):
    """26-way label: 'L'+token_at_(t-lag); null for first 'lag' positions."""
    def mk(seq):
        Z = []
        for i in range(len(seq)):
            if i < lag:
                Z.append(null)
            else:
                Z.append("L" + seq[i - lag])
        return Z
    return mk(train_tokens), mk(test_tokens)

def make_lagbin_labels(train_tokens, test_tokens, lag: int = 8, letter: str = "Q", null: str = "LB-1"):
    """Binary label: LB1 iff seq[t-lag]==letter else LB0; null for first 'lag'."""
    def mk(seq):
        Z = []
        for i in range(len(seq)):
            if i < lag:
                Z.append(null)
            else:
                Z.append("LB1" if seq[i - lag] == letter else "LB0")
        return Z
    return mk(train_tokens), mk(test_tokens)