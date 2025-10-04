# omega/alphametrics.py
from __future__ import annotations
import math, random, json, lzma, brotli, itertools, collections, struct, heapq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# ----------------------------- Tokenization -----------------------------
def tokenize_text(text: str, mode: str = "words", k: Optional[int] = None,
                  randomized: bool = False, seed: int = 0) -> List[str]:
    """
    Convert raw text into a token stream.
    mode='words' -> whitespace split; mode='chars' -> characters.
    If k>1, build sliding k-grams as composite tokens.
    If randomized=True, permute the base alphabet deterministically (seed) to test representation invariance.
    """
    if mode not in {"words", "chars"}:
        raise ValueError("mode must be 'words' or 'chars'")
    base = text.split() if mode == "words" else list(text)
    if randomized:
        rng = random.Random(seed)
        uniq = sorted(set(base)); perm = uniq[:]; rng.shuffle(perm); mp = dict(zip(uniq, perm))
        base = [mp[t] for t in base]
    if k and k > 1:
        toks = []
        for i in range(len(base) - k + 1):
            toks.append("⟨" + "·".join(base[i:i+k]) + "⟩")
        return toks
    return base

# ----------------------------- Compression / MDL -----------------------------
def _pack_tokens_as_bytes(tokens: List[str], width: int = 2) -> bytes:
    vocab = sorted(set(tokens))
    idmap = {t:i for i,t in enumerate(vocab)}
    fmt = ">H" if width == 2 else ">I"
    return b"".join(struct.pack(fmt, idmap[t]) for t in tokens)

def _pack_tokens_as_bytes_with_vocab(tokens: List[str], vocab: List[str], width: int = 2) -> bytes:
    idmap = {t:i for i,t in enumerate(vocab)}
    fmt = ">H" if width == 2 else ">I"
    return b"".join(struct.pack(fmt, idmap[t]) for t in tokens)

def _bytes_len_lzma(tokens: List[str]) -> int:
    data = _pack_tokens_as_bytes(tokens, width=2)
    return len(lzma.compress(data, preset=6))

def _bytes_len_brotli(tokens: List[str]) -> int:
    data = _pack_tokens_as_bytes(tokens, width=2)
    return len(brotli.compress(data, quality=5))

def mdl_bytes(tokens: List[str], compressors=("lzma","brotli")) -> Dict[str,int]:
    out = {}
    for c in compressors:
        if c == "lzma": out["lzma"] = _bytes_len_lzma(tokens)
        elif c == "brotli": out["brotli"] = _bytes_len_brotli(tokens)
    return out

def ncd(tokensA: List[str], tokensB: List[str], compressor: str = "lzma") -> float:
    """
    Normalized Compression Distance with a SHARED vocab to avoid mapping artifacts.
    """
    vocab = sorted(set(tokensA) | set(tokensB) | {"▮"})
    if compressor == "lzma":
        def C(ts): return len(lzma.compress(_pack_tokens_as_bytes_with_vocab(ts, vocab), preset=6))
    else:
        def C(ts): return len(brotli.compress(_pack_tokens_as_bytes_with_vocab(ts, vocab), quality=5))
    A = C(tokensA); B = C(tokensB); AB = C(tokensA + ["▮"] + tokensB)
    return (AB - min(A, B)) / max(A, B) if max(A, B) > 0 else 0.0

# ----------------------------- Entropy / MI (holdout) -----------------------------
def _entropy_from_counts(counts: Dict[str,int]) -> float:
    total = sum(counts.values())
    if total == 0: return 0.0
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0: H -= p * math.log2(p)
    return H

def predictive_info(tokens: List[str], k_ctx: int = 3,
                    holdout_frac: float = 0.2, alpha: float = 0.5) -> Tuple[float,float,float]:
    """
    Estimate I(X_past; next) on a test split with Laplace smoothing α.
    Returns (H(next), H(next|context), I) in bits.
    """
    n = len(tokens)
    if n < k_ctx + 2:
        return (0.0, 0.0, 0.0)
    split = max(k_ctx, int((1.0 - holdout_frac) * n))
    train = tokens[:split]; test = tokens[split:] or tokens
    if len(test) < k_ctx + 1:
        test = tokens; train = tokens

    ctx_counts: Dict[Tuple[str,...], Dict[str,int]] = {}
    vocab_next: Dict[str,int] = {}
    for i in range(k_ctx-1, len(train)-1):
        ctx = tuple(train[i-(k_ctx-1):i]); nxt = train[i+1]
        ctx_counts.setdefault(ctx, collections.Counter())[nxt] += 1
        vocab_next[nxt] = vocab_next.get(nxt, 0) + 1

    test_next_vocab: Dict[str,int] = {}
    for i in range(k_ctx-1, len(test)-1):
        nxt = test[i+1]; test_next_vocab[nxt] = test_next_vocab.get(nxt, 0) + 1

    keysY = sorted(set(vocab_next.keys()) | set(test_next_vocab.keys()))
    V = max(1, len(keysY))

    H_ctx: Dict[Tuple[str,...], float] = {}
    for ctx, d in ctx_counts.items():
        tot = sum(d.values()); denom = tot + alpha * V
        if denom <= 0:
            H_ctx[ctx] = math.log2(V) if V > 1 else 0.0; continue
        Hc = 0.0
        for y in keysY:
            p = (d.get(y,0) + alpha) / denom
            if p > 0: Hc -= p * math.log2(p)
        H_ctx[ctx] = Hc

    ctx_freq_test: Dict[Tuple[str,...], int] = {}
    for i in range(k_ctx-1, len(test)-1):
        ctx = tuple(test[i-(k_ctx-1):i])
        ctx_freq_test[ctx] = ctx_freq_test.get(ctx, 0) + 1

    total_events = sum(ctx_freq_test.values()) or 1
    Hcond = 0.0
    for ctx, c in ctx_freq_test.items():
        Hcond += (c / total_events) * H_ctx.get(ctx, math.log2(V) if V>1 else 0.0)

    Hnext = _entropy_from_counts(test_next_vocab)
    I = max(0.0, Hnext - Hcond)
    return (Hnext, Hcond, I)

def efficiency(tokens: List[str], k_ctx: int = 3, compressor: str = "lzma",
               holdout_frac: float = 0.2, alpha: float = 0.5) -> float:
    """Predictive bits per compressed bit (proxy), with holdout MI."""
    Hn, Hc, I = predictive_info(tokens, k_ctx=k_ctx, holdout_frac=holdout_frac, alpha=alpha)
    Cbytes = _bytes_len_lzma(tokens) if compressor == "lzma" else _bytes_len_brotli(tokens)
    Cbits_per_token = (8.0 * Cbytes) / max(1, len(tokens))
    return I / Cbits_per_token if Cbits_per_token > 0 else 0.0

# ----------------------------- JS/HAC IB -----------------------------
def _normalize(counter: Dict[str,int], keys: List[str]) -> List[float]:
    tot = float(sum(counter.values()))
    if tot <= 0.0: 
        # return float32 zeros
        return [0.0 for _ in range(len(keys))]
    inv = 1.0 / tot
    # ensure float32-like magnitudes; (stays as Python float but downstream we’ll cast)
    return [counter.get(k, 0) * inv for k in keys]

def _kl_divergence(p, q, eps: float = 1e-12) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0: continue
        s += pi * math.log((pi + eps)/(qi + eps))
    return s

def _js_divergence(p, q) -> float:
    m = [(pi+qi)*0.5 for pi, qi in zip(p, q)]
    return 0.5*_kl_divergence(p, m) + 0.5*_kl_divergence(q, m)

def ib_train_apply(tokens: List[str], k_ctx: int, n_clusters: int,
                   max_contexts: int = 2000, stride: int = 1,
                   rng: Optional[random.Random] = None,
                   max_edges_per_context: int = 512):
    """
    Train HAC over next-token distributions of contexts of length k_ctx (stride sampling for stability).
    Returns (mapping: context_tuple -> cluster_id, keysY).
    """
    if rng is None: rng = random.Random(0)
    assert k_ctx >= 1 and stride >= 1
    ctx_counts: Dict[Tuple[str, ...], Dict[str, int]] = {}
    vocab_next: Dict[str, int] = {}
    N = len(tokens)
    if N <= k_ctx: return {}, []
    for i in range(k_ctx, N-1, stride):
        ctx = tuple(tokens[i-k_ctx:i]); nxt = tokens[i+1]
        d = ctx_counts.setdefault(ctx, collections.Counter()); d[nxt] += 1
        vocab_next[nxt] = vocab_next.get(nxt, 0) + 1
    if not ctx_counts: return {}, []
    ctx_items = sorted(ctx_counts.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:max(1, max_contexts)]
    contexts = [c for c,_ in ctx_items]; keysY = sorted(vocab_next.keys())
    
#    P = [_normalize(d, keysY) for _, d in ctx_items]
    

    # compact float32 storage
    try:
        import array
        P = [array.array('f', _normalize(d, keysY)) for _, d in ctx_items]
    except Exception:
        # fallback if array is unavailable
        P = [_normalize(d, keysY) for _, d in ctx_items]

    
    if len(contexts) <= n_clusters:
        mapping = {c: i for i, c in enumerate(contexts)}; return mapping, keysY

    parent  = {i: i for i in range(len(contexts))}
    clusters = {i: [i] for i in range(len(contexts))}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def merge(i: int, j: int):
        wi, wj = len(clusters[i]), len(clusters[j]); Pi, Pj = P[i], P[j]
        P[i] = [ (wi*Pi[k] + wj*Pj[k])/(wi+wj) for k in range(len(Pi)) ]
        clusters[i].extend(clusters[j]); parent[j] = i

    heap = []
    for i in range(len(contexts)):
        for j in range(i+1, len(contexts)):
            heapq.heappush(heap, (_js_divergence(P[i], P[j]), i, j))
            
#    heap = []
#    M = len(contexts)
#    band = max(8, int(max_edges_per_context))  # cap per context
#    for i in range(M):
#        # compare only with the next 'band' contexts (sorted by frequency already)
#        j_max = min(M, i + 1 + band)
#        for j in range(i+1, j_max):
#            heapq.heappush(heap, (_js_divergence(P[i], P[j]), i, j))            
            
            
    n_active = len(contexts)

    while n_active > n_clusters and heap:
        d, a, b = heapq.heappop(heap)
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if len(clusters[ra]) < len(clusters[rb]):
            ra, rb = rb, ra

        # record cluster size before merge
        prev_size = len(clusters[ra])

        merge(ra, rb)
        n_active -= 1

        new_size = len(clusters[ra])
        if new_size == prev_size:
            # nothing actually changed, skip pushing redundant distances
            continue

        for x in range(len(contexts)):
            rx = find(x)
            if rx == ra or rx != x:
                continue
            heapq.heappush(heap, (_js_divergence(P[ra], P[rx]), ra, rx))
        

    reps = sorted({find(i) for i in range(len(contexts))})
    rep_to_cid = {rep: cid for cid, rep in enumerate(reps)}  # fixed
    mapping: Dict[Tuple[str, ...], int] = {}
    for i, ctx in enumerate(contexts):
        mapping[ctx] = rep_to_cid[find(i)]
       
    suffix_majority = {}  # dict[int ell] -> dict[Tuple[str,...] suffix] -> cid
    for ctx, cid in mapping.items():
        for ell in range(1, k_ctx):
            suf = ctx[-ell:]
            layer = suffix_majority.setdefault(ell, {})
            cnt = layer.setdefault(suf, collections.Counter())
            cnt[cid] += 1

    # collapse counters to majority vote
    suffix_majority = {
        ell: {suf: cnt.most_common(1)[0][0] for suf, cnt in layer.items()}
        for ell, layer in suffix_majority.items()
    }

    return mapping, keysY, suffix_majority


def ib_apply(tokens: List[str], k_ctx: int,
         mapping: Dict[Tuple[str, ...], int],
         null_label: str = "C-1",
         suffix_majority: Optional[Dict[int, Dict[Tuple[str, ...], int]]] = None,
         backoff: bool = True) -> List[str]:
    Z = []
    for t in range(len(tokens)):
        if t < k_ctx:
            Z.append(null_label)
            continue
        ctx = tuple(tokens[t-k_ctx:t])
        cid = mapping.get(ctx, None)
        if cid is None and backoff and suffix_majority:
            for ell in range(k_ctx-1, 0, -1):
                suf = ctx[-ell:]
                cid = suffix_majority.get(ell, {}).get(suf, None)
                if cid is not None:
                    break
        Z.append(f"C{cid}" if cid is not None else null_label)
    return Z
    

def hash_layer_like(Z_like: List[str], n_clusters: int, rng: Optional[random.Random] = None) -> List[str]:
    """Shuffle a bag with the same label histogram to break alignment but preserve frequencies."""
    if rng is None: rng = random.Random(0)
    hist = collections.Counter(Z_like)
    bag = []
    for lab, cnt in hist.items():
        bag.extend([lab]*cnt)
    rng.shuffle(bag)
    return bag


# ----------------------------- Reporting helpers -----------------------------

def layer_metrics(tokens: List[str], label: str, k_ctx: int = 3,
                  holdout_frac: float = 0.2, alpha: float = 0.5) -> Dict[str, object]:
    m = {}
    m["label"] = label
    m["len"] = len(tokens)
    mdls = mdl_bytes(tokens, compressors=("lzma","brotli"))
    m.update({f"mdl_{k}": v for k,v in mdls.items()})
    # normalized rates (bits/token)
    m["rate_lzma"]   = (8.0 * mdls["lzma"]) / max(1, len(tokens))
    m["rate_brotli"] = (8.0 * mdls["brotli"]) / max(1, len(tokens))
    Hn, Hc, I = predictive_info(tokens, k_ctx=k_ctx, holdout_frac=holdout_frac, alpha=alpha)
    m["H_next"] = Hn
    m["H_next_cond"] = Hc
    m["I_pred"] = I
    m["eff_lzma"] = efficiency(tokens, k_ctx=k_ctx, compressor="lzma",
                               holdout_frac=holdout_frac, alpha=alpha)
    m["eff_brotli"] = efficiency(tokens, k_ctx=k_ctx, compressor="brotli",
                                 holdout_frac=holdout_frac, alpha=alpha)
    return m

def summarize_layers(layers: List[Dict[str,object]]) -> str:
    cols = ["label","len","rate_lzma","rate_brotli","mdl_lzma","mdl_brotli","I_pred","eff_lzma","eff_brotli"]
    header = " | ".join(f"{c:>12}" for c in cols)
    lines = [header, "-"*len(header)]
    for L in layers:
        def fmt(x):
            if isinstance(x, float):
                return f"{x:.6g}"
            return str(x)
        lines.append(" | ".join(f"{fmt(L.get(c))[:12]:>12}" for c in cols))
    return "\n".join(lines)



# ----------------------------- Reporting glue -----------------------------
@dataclass
class IBLayerResult:
    n_clusters: int
    tokens: List[str]
    contexts: List[Tuple[str,...]]
    cluster_of_ctx: Dict[Tuple[str,...], int]

# (ib_make_layer, bpe_tokens, layer_metrics, summarize_layers, AlphaRunConfig, AlphaReport, run_alphabetization) 
# keep your originals here with added docstrings as needed.

def ib_make_layer(tokens: List[str], k_ctx: int = 3,
                  max_contexts: int = 2000, n_clusters: int = 32,
                  rng: Optional[random.Random] = None) -> IBLayerResult:
    """
    Build an IB-style layer by clustering next-token distributions of contexts (past k_ctx-1 tokens).
    Greedy HAC with JS divergence; stops at n_clusters. For speed, trims to 'max_contexts' by frequency.
    """
    if rng is None: rng = random.Random(0)
    # collect context→next counts
    ctx_counts: Dict[Tuple[str,...], Dict[str,int]] = {}
    vocab_next: Dict[str,int] = {}

    for i in range(k_ctx, len(tokens)-1):
        ctx = tuple(tokens[i-k_ctx:i])   # length == k_ctx
        nxt = tokens[i+1]
        ctx_counts.setdefault(ctx, collections.Counter())[nxt] += 1
        vocab_next[nxt] = vocab_next.get(nxt, 0) + 1

    # keep most frequent contexts
    ctx_items = sorted(ctx_counts.items(), key=lambda kv: sum(kv[1].values()), reverse=True)
    ctx_items = ctx_items[:max_contexts]
    contexts = [c for c, _ in ctx_items]
    keysY = sorted(vocab_next.keys())
    P = [ _normalize(d, keysY) for _, d in ctx_items ]  # list of distributions

    # initialize clusters: each context alone
    clusters = {i:[i] for i in range(len(contexts))}
    import heapq
    heap = []
    for i in range(len(contexts)):
        for j in range(i+1, len(contexts)):
            # use JS as distance
            m = 0.5
            # directly compute JS on the normalized vectors
            # (we already built P as probabilities)
            def _js_ij(ii=i, jj=j): return _js_divergence(P[ii], P[jj])
            heapq.heappush(heap, (_js_ij(), i, j))
    parent = {i:i for i in range(len(contexts))}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def merge(i, j):
        # merge j into i, update prototype distribution by size-weighted average
        wi, wj = len(clusters[i]), len(clusters[j])
        Pi = P[i]; Pj = P[j]
        P[i] = [ (wi*Pi[k] + wj*Pj[k]) / (wi+wj) for k in range(len(Pi)) ]
        clusters[i].extend(clusters[j])
        parent[j] = i

    n_active = len(contexts)
    while n_active > n_clusters and heap:
        d, a, b = heapq.heappop(heap)
        ra, rb = find(a), find(b)
        if ra == rb: continue
        if len(clusters[ra]) < len(clusters[rb]):
            ra, rb = rb, ra
        merge(ra, rb)
        n_active -= 1
        # update distances to other active reps
        for x in range(len(contexts)):
            rx = find(x)
            if rx == ra or rx != x: continue
            heapq.heappush(heap, (_js_divergence(P[ra], P[rx]), ra, rx))

    reps = sorted({find(i) for i in range(len(contexts))})
    rep_to_cid = {rep: cid for cid, rep in enumerate(reps)}
    cluster_of_ctx = {}
    for i in range(len(contexts)):
        cid = rep_to_cid[find(i)]
        cluster_of_ctx[contexts[i]] = cid

    # produce Z sequence: for each position, assign cluster of its context (fallback -1)
    Z = []
    null = f"C{-1}"

    for i in range(k_ctx, len(tokens)-1):
        ctx = tuple(tokens[i-k_ctx:i])     # length == k_ctx
        cid = cluster_of_ctx.get(ctx, -1)
        Z.append(f"C{cid}" if cid >= 0 else null)

    return IBLayerResult(n_clusters=len(reps), tokens=Z,
                         contexts=contexts, cluster_of_ctx=cluster_of_ctx)

# ----------------------------- BPE (MDL-ish dictionary) -----------------------------

def bpe_tokens(tokens: List[str], merges: int = 100, min_count: int = 2) -> Tuple[List[str], Dict[Tuple[str,str], int]]:
    """
    Simple Byte-Pair Encoding over token stream (adjacent-pair merges).
    Returns new token stream and a dict of merges used.
    """
    seq = tokens[:]
    merges_used: Dict[Tuple[str,str], int] = {}
    for _ in range(merges):
        pair_counts = collections.Counter(zip(seq, seq[1:]))
        if not pair_counts: break
        (a,b), cnt = pair_counts.most_common(1)[0]
        if cnt < min_count: break
        new_tok = f"<{a}+{b}>"
        merges_used[(a,b)] = cnt
        out = []
        i = 0
        L = len(seq)
        while i < L:
            if i < L-1 and seq[i] == a and seq[i+1] == b:
                out.append(new_tok)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return seq, merges_used

# ----------------------------- Reporting -----------------------------

def layer_metrics(tokens: List[str], label: str, k_ctx: int = 3,
                  holdout_frac: float = 0.2, alpha: float = 0.5) -> Dict[str, object]:
    m = {}
    m["label"] = label
    m["len"] = len(tokens)
    mdls = mdl_bytes(tokens, compressors=("lzma","brotli"))
    m.update({f"mdl_{k}": v for k,v in mdls.items()})
    # normalized rates (bits/token)
    m["rate_lzma"]   = (8.0 * mdls["lzma"]) / max(1, len(tokens))
    m["rate_brotli"] = (8.0 * mdls["brotli"]) / max(1, len(tokens))
    Hn, Hc, I = predictive_info(tokens, k_ctx=k_ctx, holdout_frac=holdout_frac, alpha=alpha)
    m["H_next"] = Hn
    m["H_next_cond"] = Hc
    m["I_pred"] = I
    m["eff_lzma"] = efficiency(tokens, k_ctx=k_ctx, compressor="lzma",
                               holdout_frac=holdout_frac, alpha=alpha)
    m["eff_brotli"] = efficiency(tokens, k_ctx=k_ctx, compressor="brotli",
                                 holdout_frac=holdout_frac, alpha=alpha)
    return m

def summarize_layers(layers: List[Dict[str,object]]) -> str:
    cols = ["label","len","rate_lzma","rate_brotli","mdl_lzma","mdl_brotli","I_pred","eff_lzma","eff_brotli"]
    header = " | ".join(f"{c:>12}" for c in cols)
    lines = [header, "-"*len(header)]
    for L in layers:
        def fmt(x):
            if isinstance(x, float):
                return f"{x:.6g}"
            return str(x)
        lines.append(" | ".join(f"{fmt(L.get(c))[:12]:>12}" for c in cols))
    return "\n".join(lines)

@dataclass
class AlphaRunConfig:
    mode: str = "words"        # 'words' | 'chars'
    k_base: int = 1            # base k-gram for tokens
    k_ctx: int = 3             # context length for prediction
    ib_clusters: Tuple[int,...] = (64, 32, 16, 8)
    bpe_merges: Tuple[int,...] = (50, 100, 200)
    randomized: bool = False
    seed: int = 0

@dataclass
class AlphaReport:
    layers: List[Dict[str,object]]
    ncds: Dict[str,float]

def run_alphabetization(train_text: str,
                        codex_text: Optional[str] = None,
                        decoy_text: Optional[str] = None,
                        cfg: Optional[AlphaRunConfig] = None) -> AlphaReport:
    """
    Build layers with IB (contexts→clusters) and BPE merges; compute MDL/prediction/efficiency; optional NCDs.
    """
    if cfg is None: cfg = AlphaRunConfig()

    # base tokens and representations
    base = tokenize_text(train_text, mode=cfg.mode, k=cfg.k_base, randomized=cfg.randomized, seed=cfg.seed)

    layers: List[Dict[str,object]] = []
    # L0: raw/base
    layers.append(layer_metrics(base, f"raw(k={cfg.k_base})", k_ctx=cfg.k_ctx))

    # IB layers (stacked)
    tokens_ib = base

    for li, K in enumerate(cfg.ib_clusters, 1):
        ib = ib_make_layer(tokens_ib, k_ctx=cfg.k_ctx, n_clusters=K)
        Z = ib.tokens
        layers.append(layer_metrics(Z, f"IB{li}_K={ib.n_clusters}", k_ctx=cfg.k_ctx))
        tokens_ib = Z

    # BPE layers (fresh from base each time)
    for M in cfg.bpe_merges:
        Zb, _ = bpe_tokens(base, merges=M)
        layers.append(layer_metrics(Zb, f"BPE_M={M}", k_ctx=cfg.k_ctx))

    # NCD, if codex/decoy given (use shared vocab)
    ncds = {}
    if codex_text and decoy_text:
        codex = tokenize_text(codex_text, mode=cfg.mode, k=cfg.k_base, randomized=False)
        decoy = tokenize_text(decoy_text, mode=cfg.mode, k=cfg.k_base, randomized=False)
        ncds["ncd_lzma_codex"]   = ncd(base, codex, compressor="lzma")
        ncds["ncd_lzma_decoy"]   = ncd(base, decoy, compressor="lzma")
        ncds["ncd_brotli_codex"] = ncd(base, codex, compressor="brotli")
        ncds["ncd_brotli_decoy"] = ncd(base, decoy, compressor="brotli")

    return AlphaReport(layers=layers, ncds=ncds)

def as_json(report: AlphaReport) -> str:
    return json.dumps({"layers": report.layers, "ncds": report.ncds}, indent=2)
