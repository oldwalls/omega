# alphabet/cond_coder.py

"""
We construct augmented streams by interleaving labels and tokens (⟨Z, x⟩ pairs).
Training sees true label–token pairs on TRAIN only (no TEST leakage).
At evaluation, we compute code length only at the real-token positions in TEST.
Thus H_base = H(x_t | x_{<t}); H_cond = H(x_t | x_{<t}, z_t). Δ = H_cond - H_base.
Negative Δ implies useful information in z_t (Ω-positive evidence).
"""
# cond_coder.py  (replace the conditional parts with this gated version)
from __future__ import annotations
import math, collections
from typing import List, Dict, Tuple, Optional

def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], Dict[str, int]]:
    """Counts[context]->next-token counts for base model (train only)."""
    C: Dict[Tuple[str, ...], Dict[str, int]] = {}
    if n <= 0: return C
    for i in range(n, len(tokens)):
        ctx = tuple(tokens[i-n:i])
        nxt = tokens[i]
        d = C.setdefault(ctx, {})
        d[nxt] = d.get(nxt, 0) + 1
    return C

def _ngram_counts_gated(tokens: List[str], labels: List[str], n: int) -> Dict[str, Dict[Tuple[str, ...], Dict[str, int]]]:
    """Label-gated counts: Counts[label][context]->next-token counts (train only)."""
    G: Dict[str, Dict[Tuple[str, ...], Dict[str, int]]] = {}
    if n <= 0: return G
    assert len(tokens) == len(labels)
    for i in range(n, len(tokens)):
        l  = labels[i]
        ctx = tuple(tokens[i-n:i])
        nxt = tokens[i]
        L = G.setdefault(l, {})
        d = L.setdefault(ctx, {})
        d[nxt] = d.get(nxt, 0) + 1
    return G

def _prob_from_counts(d: Dict[str,int], vocab: List[str], alpha: float = 1.0) -> Dict[str, float]:
    V = max(1, len(vocab))
    tot = sum(d.values())
    denom = tot + alpha * V
    if denom <= 0:  # uniform fallback
        p = 1.0 / V
        return {y: p for y in vocab}
    out = {}
    for y in vocab:
        out[y] = (d.get(y, 0) + alpha) / denom
    return out

def _collect_vocab(*streams: List[str]) -> List[str]:
    S = set()
    for s in streams:
        S.update(s)
    return sorted(S)

###################################################


# --- Label-wise mixture gated coder ------------------------------------------
def compute_conditional_codelength_mixture_labelwise(
    train_tokens, test_tokens, train_labels, test_labels,
    n: int = 3, alpha: float = 1.0, lam_map: dict | None = None, default_lam: float = 0.0
) -> dict:
    """
    H_cond under label-wise mixture:
      P(x|ctx,z)= (1-λ(z))*P_base(x|ctx) + λ(z)*P_gate(x|ctx,z).
    lam_map: dict from label -> λ in [0,1]; others use default_lam.
    """
    import math
    assert train_labels is not None and test_labels is not None
    assert len(train_labels) == len(train_tokens)
    assert len(test_labels)  == len(test_tokens)

    vocab = _collect_vocab(train_tokens, test_tokens)
    V = len(vocab)

    Cb = _ngram_counts(train_tokens, n)
    Gc = _ngram_counts_gated(train_tokens, train_labels, n)

    def p_base(ctx, y):
        d = Cb.get(ctx, {})
        return _prob_from_counts(d, vocab, alpha)[y]

    def p_gate(lab, ctx, y):
        d = Gc.get(lab, {}).get(ctx, {})
        return _prob_from_counts(d, vocab, alpha)[y]

    def nll_base(i):
        if i < n: return math.log2(V)
        ctx = tuple(test_tokens[i-n:i])
        return -math.log2(max(p_base(ctx, test_tokens[i]), 1e-300))

    def nll_mix(i):
        if i < n: return math.log2(V)
        lab = test_labels[i]
        lam = (lam_map or {}).get(lab, default_lam)
        ctx = tuple(test_tokens[i-n:i]); y = test_tokens[i]
        pb = p_base(ctx, y); pg = p_gate(lab, ctx, y)
        p  = (1.0 - lam)*pb + lam*pg
        return -math.log2(max(p, 1e-300))

    H_base = sum(nll_base(i) for i in range(len(test_tokens))) / max(1, len(test_tokens))
    H_mix  = sum(nll_mix(i)  for i in range(len(test_tokens))) / max(1, len(test_tokens))
    return {"H_base": H_base, "H_cond": H_mix}

def pointwise_cond_losses_mixture_labelwise(
    train_tokens, test_tokens, train_labels, test_labels,
    n: int = 3, alpha: float = 1.0, lam_map: dict | None = None, default_lam: float = 0.0
):
    """Per-token losses for label-wise mixture."""
    import math
    assert len(train_labels) == len(train_tokens)
    assert len(test_labels)  == len(test_tokens)

    vocab = _collect_vocab(train_tokens, test_tokens)
    V = len(vocab)
    Cb = _ngram_counts(train_tokens, n)
    Gc = _ngram_counts_gated(train_tokens, train_labels, n)

    def p_base(ctx, y):
        d = Cb.get(ctx, {})
        return _prob_from_counts(d, vocab, alpha)[y]

    def p_gate(lab, ctx, y):
        d = Gc.get(lab, {}).get(ctx, {})
        return _prob_from_counts(d, vocab, alpha)[y]

    out = []
    for i in range(len(test_tokens)):
        if i < n:
            out.append(math.log2(V)); continue
        lab = test_labels[i]
        lam = (lam_map or {}).get(lab, default_lam)
        ctx = tuple(test_tokens[i-n:i]); y = test_tokens[i]
        pb = p_base(ctx, y); pg = p_gate(lab, ctx, y)
        p  = (1.0 - lam)*pb + lam*pg
        out.append(-math.log2(max(p, 1e-300)))
    return out








###################################################
def compute_conditional_codelength(train_tokens: List[str], test_tokens: List[str],
                                   train_labels: Optional[List[str]],
                                   test_labels:  Optional[List[str]],
                                   n: int = 4, alpha: float = 1.0) -> Dict[str, float]:
    """
    Base: H_base = mean -log2 P(x_t | x_{t-n:t-1}) on TEST using n-gram trained on TRAIN.
    Cond (gated): H_cond = mean -log2 P(x_t | x_{t-n:t-1}, z_t) where z_t gates *which* n-gram table is used.
    No interleaving; labels do not enter the context.
    """
    assert len(train_tokens) > 0 and len(test_tokens) > 0
    vocab = _collect_vocab(train_tokens, test_tokens)

    # --- base model trained on TRAIN ---
    C_base = _ngram_counts(train_tokens, n)

    # --- conditional model (gated) trained on TRAIN ---
    if train_labels is not None and test_labels is not None:
        assert len(train_labels) == len(train_tokens)
        assert len(test_labels)  == len(test_tokens)
        G_cond = _ngram_counts_gated(train_tokens, train_labels, n)
    else:
        G_cond = None

    # --- evaluate on TEST ---
    def _neglogp_base(i: int) -> float:
        if i < n:  # cold start: uniform
            return math.log2(len(vocab))
        ctx = tuple(test_tokens[i-n:i])
        d = C_base.get(ctx, {})
        p = _prob_from_counts(d, vocab, alpha)[test_tokens[i]]
        return -math.log2(max(p, 1e-300))

    def _neglogp_cond(i: int) -> float:
        if G_cond is None:  # no labels -> degenerate to base
            return _neglogp_base(i)
        if i < n:
            return math.log2(len(vocab))
        l = test_labels[i]
        ctx = tuple(test_tokens[i-n:i])
        d = G_cond.get(l, {}).get(ctx, {})
        p = _prob_from_counts(d, vocab, alpha)[test_tokens[i]]
        return -math.log2(max(p, 1e-300))

    # means over all test positions
    H_base = sum(_neglogp_base(i) for i in range(len(test_tokens))) / max(1, len(test_tokens))
    H_cond = sum(_neglogp_cond(i) for i in range(len(test_tokens))) / max(1, len(test_tokens))

    return {"H_base": H_base, "H_cond": H_cond}

def pointwise_base_losses(train_tokens: List[str], test_tokens: List[str], n: int = 4, alpha: float = 1.0) -> List[float]:
    """Per-token losses for base model."""
    vocab = _collect_vocab(train_tokens, test_tokens)
    C_base = _ngram_counts(train_tokens, n)
    out = []
    for i in range(len(test_tokens)):
        if i < n:
            out.append(math.log2(len(vocab))); continue
        ctx = tuple(test_tokens[i-n:i]); d = C_base.get(ctx, {})
        p = _prob_from_counts(d, vocab, alpha)[test_tokens[i]]
        out.append(-math.log2(max(p, 1e-300)))
    return out

def pointwise_cond_losses(train_tokens: List[str], test_tokens: List[str],
                          train_labels: List[str], test_labels: List[str],
                          n: int = 4, alpha: float = 1.0) -> List[float]:
    """Per-token losses for label-gated conditional model."""
    assert len(train_labels) == len(train_tokens)
    assert len(test_labels)  == len(test_tokens)
    vocab = _collect_vocab(train_tokens, test_tokens)
    G_cond = _ngram_counts_gated(train_tokens, train_labels, n)
    out = []
    for i in range(len(test_tokens)):
        if i < n:
            out.append(math.log2(len(vocab))); continue
        l = test_labels[i]; ctx = tuple(test_tokens[i-n:i])
        d = G_cond.get(l, {}).get(ctx, {})
        p = _prob_from_counts(d, vocab, alpha)[test_tokens[i]]
        out.append(-math.log2(max(p, 1e-300)))
    return out

# --- Global mixture gated coder ---------------------------------------------
def compute_conditional_codelength_mixture(
    train_tokens, test_tokens, train_labels, test_labels,
    n: int = 3, alpha: float = 1.0, lam: float = 0.5
) -> dict:
    """
    H_cond under global mixture:
      P(x|ctx,z)= (1-λ)*P_base(x|ctx) + λ*P_gate(x|ctx,z).
    """
    assert train_labels is not None and test_labels is not None
    assert len(train_labels) == len(train_tokens)
    assert len(test_labels)  == len(test_tokens)

    vocab = _collect_vocab(train_tokens, test_tokens)
    V = len(vocab)
    Cb = _ngram_counts(train_tokens, n)
    Gc = _ngram_counts_gated(train_tokens, train_labels, n)

    def p_base(ctx, y):
        d = Cb.get(ctx, {})
        return _prob_from_counts(d, vocab, alpha)[y]

    def p_gate(lab, ctx, y):
        d = Gc.get(lab, {}).get(ctx, {})
        return _prob_from_counts(d, vocab, alpha)[y]

    def nll_base(i):
        if i < n: return math.log2(V)
        ctx = tuple(test_tokens[i-n:i])
        return -math.log2(max(p_base(ctx, test_tokens[i]), 1e-300))

    def nll_mix(i):
        if i < n: return math.log2(V)
        lab = test_labels[i]
        ctx = tuple(test_tokens[i-n:i]); y = test_tokens[i]
        pb = p_base(ctx, y); pg = p_gate(lab, ctx, y)
        p  = (1.0 - lam)*pb + lam*pg
        return -math.log2(max(p, 1e-300))

    H_base = sum(nll_base(i) for i in range(len(test_tokens))) / max(1, len(test_tokens))
    H_mix  = sum(nll_mix(i)  for i in range(len(test_tokens))) / max(1, len(test_tokens))
    return {"H_base": H_base, "H_cond": H_mix}

def pointwise_cond_losses_mixture(
    train_tokens, test_tokens, train_labels, test_labels,
    n: int = 3, alpha: float = 1.0, lam: float = 0.5
):
    """Per-token losses for global λ mixture."""
    assert len(train_labels) == len(train_tokens)
    assert len(test_labels)  == len(test_tokens)

    vocab = _collect_vocab(train_tokens, test_tokens)
    V = len(vocab)
    Cb = _ngram_counts(train_tokens, n)
    Gc = _ngram_counts_gated(train_tokens, train_labels, n)

    out = []
    for i in range(len(test_tokens)):
        if i < n:
            out.append(math.log2(V)); continue
        lab = test_labels[i]
        ctx = tuple(test_tokens[i-n:i]); y = test_tokens[i]
        pb = _prob_from_counts(Cb.get(ctx, {}), vocab, alpha)[y]
        pg = _prob_from_counts(Gc.get(lab, {}).get(ctx, {}), vocab, alpha)[y]
        p  = (1.0 - lam)*pb + lam*pg
        out.append(-math.log2(max(p, 1e-300)))
    return out
