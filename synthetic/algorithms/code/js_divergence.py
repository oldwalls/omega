def _js_divergence_from_counts(cntA, cntB, eps=1e-12):
    keys = set(cntA.keys()) | set(cntB.keys())
    totA = sum(cntA.values()) or 1
    totB = sum(cntB.values()) or 1
    P = {k: (cntA.get(k,0)/totA) for k in keys}
    Q = {k: (cntB.get(k,0)/totB) for k in keys}
    M = {k: 0.5*(P[k]+Q[k]) for k in keys}
    def KL(X,Y):
        s = 0.0
        for k in keys:
            if X[k] > 0:
                s += X[k]*math.log((X[k]+eps)/(Y[k]+eps))
        return s
    return 0.5*KL(P,M) + 0.5*KL(Q,M)

def ngram_counts(tokens, k):
    cnt = collections.Counter()
    for i in range(k, len(tokens)):
        cnt[tuple(tokens[i-k:i])] += 1
    return cnt

def ctx_stats(train_tokens, test_tokens, k_ctx, mapping):
    # existing fields...
    A = {tuple(train_tokens[i-k_ctx:i]) for i in range(k_ctx, len(train_tokens))}
    B = {tuple(test_tokens[i-k_ctx:i])  for i in range(k_ctx, len(test_tokens))}
    seen = sum(1 for i in range(k_ctx, len(test_tokens)) if tuple(test_tokens[i-k_ctx:i]) in mapping)
    denom = max(1, len(test_tokens)-k_ctx)
    jacc = (len(A & B) / (len(A | B) or 1))
    # NEW: JS divergence on k-gram frequencies
    js = _js_divergence_from_counts(ngram_counts(train_tokens, k_ctx),
                                    ngram_counts(test_tokens,  k_ctx))
    return {
        "train_unique_ctx": len(A),
        "test_unique_ctx": len(B),
        "test_seen_frac": seen/denom,
        "jaccard_unique": jacc,
        "js_freq": js
    }