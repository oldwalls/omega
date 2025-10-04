# omega/cli.py
from __future__ import annotations
import argparse, pathlib, sys, json, math, random, time
from typing import List, Dict, Tuple, Optional

from labelers import make_topic_labels
from bootstrap import paired_block_bootstrap, block_bootstrap_mean
from alphametrics import (
    AlphaRunConfig, run_alphabetization, summarize_layers, as_json,
    ib_train_apply, ib_apply, hash_layer_like, tokenize_text
)
from cond_coder import (            
    compute_conditional_codelength, compute_conditional_codelength_mixture_labelwise,
    pointwise_base_losses, pointwise_cond_losses, pointwise_cond_losses_mixture_labelwise,
)

# ------------------------------- utils -------------------------------


def _read_text(path: str | None) -> str:
    if not path or path == "-":
        return sys.stdin.read()
    return pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")

def _write_text(path: str, data: str) -> None:
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data, encoding="utf-8")

def _parse_int_list(s: str | None, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if s is None: return default
    s = s.strip()
    if not s: return tuple()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())

def _join_tokens(mode: str, tokens: List[str]) -> str:
    return ("".join(tokens)) if mode == "chars" else (" ".join(tokens))

def _percentile(xs: List[float], p: float) -> float:
    if not xs: return float("nan")
    ys = sorted(xs)
    k = max(0, min(len(ys)-1, int(round((len(ys)-1)*p))))
    return ys[k]


def ib_coverage(tokens, k_ctx, mapping, suffix):
    """Report fractions of full hits, suffix-backoff hits, and nulls."""
    full = back = null = 0
    for t in range(len(tokens)):
        if t < k_ctx:
            null += 1
            continue
        ctx = tuple(tokens[t-k_ctx:t])
        if ctx in mapping:
            full += 1
        else:
            hit = False
            for ell in range(k_ctx-1, 0, -1):
                if ctx[-ell:] in suffix.get(ell, {}):
                    back += 1
                    hit = True
                    break
            if not hit:
                null += 1
    denom = max(1, len(tokens) - k_ctx)
    return {
        "full_frac": full/denom,
        "backoff_frac": back/denom,
        "null_frac": null/denom,
    }

def make_lag_labels(train, test, lag, null="L-1"):
    def mk(seq):
        Z=[]
        for i in range(len(seq)):
            if i < lag: Z.append(null)
            else:       Z.append("L"+seq[i-lag])  # one label per alphabet symbol
        return Z
    return mk(train), mk(test)

def make_lagbin_labels(train, test, lag: int, letter: str, null="LB-1"):
    def mk(seq):
        Z=[]
        for i in range(len(seq)):
            if i < lag:
                Z.append(null)
            else:
                Z.append("LB1" if seq[i-lag] == letter else "LB0")
        return Z
    return mk(train), mk(test)

# ----------------------------- analyze-cond ---------------------------
def analyze_cond(argv: Optional[List[str]] = None):
    """
    Conditional codelength analysis on HOLDOUT:
      - Build labels Z on TRAIN (IB, Topic, or Lag/LagBin) and apply to TRAIN/TEST.
      - Compute H_base and H_cond (per args.cond_mode) on TEST.
      - Δ = mean(pointwise_cond - pointwise_base). Δ<0 helps.
      - Hash control: histogram-preserving permutation of TEST labels.
      - IB path reports coverage (full/backoff/null) on TRAIN/TEST.
    """
    import random, pathlib, json
    p = argparse.ArgumentParser(prog="omega.analyze-cond", add_help=True)
    p.add_argument("--train", required=True)
    p.add_argument("--holdout", required=True)
    p.add_argument("--mode", choices=["words"], default="words")
    p.add_argument("--k_ctx", type=int, default=4)

    p.add_argument("--k_grid", type=str, default=None,
                   help="Comma-separated Ks or range like 4:16:2 for IB layer sweeps.")
    p.add_argument("--block_lens", type=str, default=None,
                   help="Comma-separated block lengths for bootstrap stability (e.g., 40,60,80,100,160).")
    p.add_argument("--control", choices=["none","global","block","label"], default="none",
                   help="Apply null transformation: global=global shuffle tokens; block=block shuffle tokens; label=permute labels.")
    p.add_argument("--control_block", type=int, default=50,
                   help="Block length for block control.")
    p.add_argument("--seeds", type=str, default="0",
                   help="Comma-separated RNG seeds for robustness.")
    p.add_argument("--report_json", type=str, default=None,
                   help="Path to write a compact JSON summary (ΔI_pred, ΔSCR, controls).")
    p.add_argument("--ib-clusters", type=int, default=32)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--block-len", type=int, default=128)
    p.add_argument("--out-json", default="reports/alpha_cond.json")
    p.add_argument("--topic-window", type=int, default=100)
    p.add_argument("--topic-stride", type=int, default=50)
    p.add_argument("--topic-k", type=int, default=16)
    p.add_argument("--n", type=int, default=4, help="n-gram order for base/cond coder")
    p.add_argument("--label-mode", choices=["ib","topic","lag","lagbin"], default="ib")
    p.add_argument("--lag", type=int, default=8)
    p.add_argument("--lag-letter", type=str, default="Q")
    p.add_argument("--alpha", type=float, default=1.0, help="Dirichlet smoothing for n-grams")
    p.add_argument("--cond_mode", choices=["gated","mix","mixlw"], default="gated")
    p.add_argument("--lam", type=float, default=0.5, help="λ for 'mix'")
    p.add_argument("--lam-pos", type=float, default=0.7, help="λ for positive label in 'mixlw'")
    p.add_argument("--lam-neg", type=float, default=0.0, help="λ for negative label in 'mixlw'")
    p.add_argument("--ib-max-contexts", type=int, default=4000,
               help="Max contexts kept during IB mapping (↑ for better coverage).")
    args = p.parse_args(argv)

    rng0 = random.Random(0)

    def _load_tokens(path: str) -> List[str]:
        return pathlib.Path(path).read_text(encoding="utf-8").split()

    train = _load_tokens(args.train)
    test  = _load_tokens(args.holdout)

    # ---------------- label-maker branch ----------------
    label_mode = args.label_mode
    mapping = None
    suffix = {}

    if label_mode == "ib":
        # smart cap: avoid quadratic blowups in HAC
        cap = min(args.ib_max_contexts, 1000 + 8 * args.ib_clusters)
        mapping, _, suffix = ib_train_apply(
            train, k_ctx=args.k_ctx, n_clusters=args.ib_clusters,
            max_contexts=cap, stride=args.stride, rng=rng0
        )
        
        
        
#        mapping, _, suffix = ib_train_apply(
#            train, k_ctx=args.k_ctx, n_clusters=args.ib_clusters,
#            max_contexts=args.ib_max_contexts, stride=args.stride, rng=rng0
#        )

        Z_train = ib_apply(train, k_ctx=args.k_ctx, mapping=mapping,
                           null_label="C-1", suffix_majority=suffix, backoff=True)
        Z_test  = ib_apply(test,  k_ctx=args.k_ctx, mapping=mapping,
                           null_label="C-1", suffix_majority=suffix, backoff=True)
        null_label = "C-1"
        n_clusters = args.ib_clusters

    elif label_mode == "topic":
        Z_train, Z_test = make_topic_labels(
            train, test, win=args.topic_window, stride=args.topic_stride, K=args.topic_k, rng=rng0
        )
        null_label = "T-1"
        n_clusters = args.topic_k

    elif label_mode == "lag":
        Z_train, Z_test = make_lag_labels(train, test, lag=args.lag, null="L-1")
        null_label = "L-1"
        # up to 26 letters; count what actually appears (excluding null)
        n_clusters = max(1, len({z for z in Z_test if z != null_label}))

    else:  # lagbin
        Z_train, Z_test = make_lagbin_labels(train, test, lag=args.lag, letter=args.lag_letter, null="LB-1")
        null_label = "LB-1"
        n_clusters = 2

    # ------------- universal alignment seatbelt -------------
    def align_labels_to_tokens(tokens, labels, null_label):
        if labels is None:
            return [null_label] * len(tokens)
        Lx, Lz = len(tokens), len(labels)
        if Lz < Lx:  # pad on the left
            return [null_label] * (Lx - Lz) + list(labels)
        if Lz > Lx:  # trim from front
            return list(labels)[-Lx:]
        return list(labels)

    Z_train = align_labels_to_tokens(train, Z_train, null_label)
    Z_test  = align_labels_to_tokens(test,  Z_test,  null_label)

    # build hash AFTER alignment so lengths match test
    Z_hash  = hash_layer_like(Z_test, n_clusters=n_clusters, rng=random.Random(1))

    # hard sanity
    assert len(Z_train) == len(train), f"train labels {len(Z_train)} != tokens {len(train)}"
    assert len(Z_test)  == len(test),  f"test labels  {len(Z_test)}  != tokens {len(test)}"
    assert len(Z_hash)  == len(test),  f"hash labels  {len(Z_hash)}  != test tokens {len(test)}"

    # ---------------- optional null/control transforms ----------------
    from controls import global_shuffle, block_shuffle, permute_labels, lzma_energy
    control_mode = getattr(args, "control", "none")
    seeds = [int(s) for s in (getattr(args, "seeds","0").split(",")) if s.strip()!=""]
    # Apply control to TEST tokens or labels as requested; run multiple seeds and average later if needed
    test_variants = [(test, Z_test, "true")]
    if control_mode == "global":
        test_variants = [(global_shuffle(test, seed=s), Z_test, f"ctrl-global-{s}") for s in seeds]
    elif control_mode == "block":
        bl = getattr(args,"control_block",50)
        test_variants = [(block_shuffle(test, block_len=bl, seed=s), Z_test, f"ctrl-block-{bl}-{s}") for s in seeds]
    elif control_mode == "label":
        test_variants = [(test, permute_labels(Z_test, seed=s), f"ctrl-label-{s}") for s in seeds]

    # ---------------- choose coder ----------------
    results = []
    N = int(args.n); A = float(args.alpha)
    # Base (unconditional) depends only on train/test tokens
    for test_tok, Zt, tag in test_variants:
        base = compute_conditional_codelength(train, test_tok, None, None, n=N, alpha=A)
        base_lp = pointwise_base_losses(train, test_tok, n=N, alpha=A)

        if args.cond_mode == "gated":
            cond = compute_conditional_codelength(train, test_tok, Z_train, Zt, n=N, alpha=A)
            hsh  = compute_conditional_codelength(train, test_tok, Z_train, Z_hash, n=N, alpha=A)
            cond_lp = pointwise_cond_losses(train, test_tok, Z_train, Zt, n=N, alpha=A)
            hash_lp = pointwise_cond_losses(train, test_tok, Z_train, Z_hash, n=N, alpha=A)
        elif args.cond_mode == "mix":
            uniq = set(Z_train) | set(Zt)
            nullish = {null_label, "__NULL__", "C-1", "T-1", "L-1", "LB-1"} & uniq
            lam_map = {z: float(args.lam) for z in uniq - nullish}
            cond = compute_conditional_codelength_mixture_labelwise(
                train, test_tok, Z_train, Zt, n=N, alpha=A, lam_map=lam_map, default_lam=0.0
            )
            hsh  = compute_conditional_codelength_mixture_labelwise(
                train, test_tok, Z_train, Z_hash, n=N, alpha=A, lam_map=lam_map, default_lam=0.0
            )
            cond_lp = pointwise_cond_losses_mixture_labelwise(train, test_tok, Z_train, Zt, n=N, alpha=A, lam_map=lam_map, default_lam=0.0)
            hash_lp = pointwise_cond_losses_mixture_labelwise(train, test_tok, Z_train, Z_hash, n=N, alpha=A, lam_map=lam_map, default_lam=0.0)
        else:
            raise SystemExit(f"unknown cond_mode: {args.cond_mode}")

        # ΔI_pred surrogate = mean(cond_lp - base_lp)
        import statistics as _stats
        delta = _stats.mean([c - b for c,b in zip(cond_lp, base_lp)])
        delta_hash = _stats.mean([c - b for c,b in zip(hash_lp, base_lp)])

        # SCR using LZMA compressed size as dissipation proxy
        energy = lzma_energy(test_tok)
        scr = (-delta) / max(energy, 1)  # negative delta means gain; SCR higher is better

        results.append({
            "tag": tag,
            "delta_ib": delta,
            "delta_hash": delta_hash,
            "scr_lzma": scr,
            "energy_lzma": energy,
            "n_tokens": len(test_tok),
        })

    N = int(args.n)
    A = float(args.alpha)

    # Base always the same
    base = compute_conditional_codelength(train, test, None, None, n=N, alpha=A)
    base_lp = pointwise_base_losses(train, test, n=N, alpha=A)

    if args.cond_mode == "gated":
        cond = compute_conditional_codelength(train, test, Z_train, Z_test, n=N, alpha=A)
        hsh  = compute_conditional_codelength(train, test, Z_train, Z_hash, n=N, alpha=A)
        cond_lp = pointwise_cond_losses(train, test, Z_train, Z_test, n=N, alpha=A)
        hash_lp = pointwise_cond_losses(train, test, Z_train, Z_hash, n=N, alpha=A)

    elif args.cond_mode == "mix":
        # emulate global-λ mix using label-wise mixture with same λ for all non-null labels
        uniq = set(Z_train) | set(Z_test)
        nullish = {null_label, "__NULL__", "C-1", "T-1", "L-1", "LB-1"} & uniq
        lam_map = {z: float(args.lam) for z in uniq - nullish}
        cond = compute_conditional_codelength_mixture_labelwise(
            train, test, Z_train, Z_test, n=N, alpha=A, lam_map=lam_map, default_lam=0.0
        )
        hsh  = compute_conditional_codelength_mixture_labelwise(
            train, test, Z_train, Z_hash, n=N, alpha=A, lam_map=lam_map, default_lam=0.0
        )
        cond_lp = pointwise_cond_losses_mixture_labelwise(
            train, test, Z_train, Z_test, n=N, alpha=A, lam_map=lam_map, default_lam=0.0
        )
        hash_lp = pointwise_cond_losses_mixture_labelwise(
            train, test, Z_train, Z_hash, n=N, alpha=A, lam_map=lam_map, default_lam=0.0
        )

    else:  # mixlw: label-wise mixture (gate only on chosen labels)
        uniq = set(Z_train) | set(Z_test)
        nullish = {null_label, "__NULL__", "C-1", "T-1", "L-1", "LB-1"} & uniq
        lam_map = {z: 0.0 for z in nullish}
        if label_mode == "lagbin":
            # Prefer canonical names; otherwise pick the rarer non-null as "positive"
            pos = "LB1" if "LB1" in uniq else None
            neg = "LB0" if "LB0" in uniq else None
            if not (pos and neg):
                from collections import Counter
                nonnull = list(uniq - nullish)
                if len(nonnull) == 2:
                    c = Counter(Z_test)
                    nonnull.sort(key=lambda z: c[z])   # rarer first
                    pos, neg = nonnull[0], nonnull[1]
                elif len(nonnull) == 1:
                    pos, neg = nonnull[0], None
            if pos: lam_map[pos] = float(args.lam_pos)
            if neg: lam_map[neg] = float(args.lam_neg)
            print(f"[mixlw] labels gated with λ>0: {[l for l,lam in lam_map.items() if lam>0]}")
        # compute with lam_map
        cond = compute_conditional_codelength_mixture_labelwise(
            train, test, Z_train, Z_test, n=N, alpha=A, lam_map=lam_map, default_lam=0.0
        )
        hsh  = compute_conditional_codelength_mixture_labelwise(
            train, test, Z_train, Z_hash, n=N, alpha=A, lam_map=lam_map, default_lam=0.0
        )
        cond_lp = pointwise_cond_losses_mixture_labelwise(
            train, test, Z_train, Z_test, n=N, alpha=A, lam_map=lam_map, default_lam=0.0
        )
        hash_lp = pointwise_cond_losses_mixture_labelwise(
            train, test, Z_train, Z_hash, n=N, alpha=A, lam_map=lam_map, default_lam=0.0
        )

    # ---------------- summarize ----------------
    #H_base = base["H_base"]; H_cond = cond["H_cond"]; H_hash = hsh["H_cond"]

    import statistics as _stats
    H_base = _stats.mean(base_lp)
    H_cond = _stats.mean(cond_lp)
    H_hash = _stats.mean(hash_lp)

    assert len(base_lp) == len(cond_lp) == len(hash_lp) == len(test), \
        f"lengths mismatch: base={len(base_lp)} cond={len(cond_lp)} hash={len(hash_lp)} test={len(test)}"
    delta_arr      = [cond_lp[i] - base_lp[i] for i in range(len(test))]
    
    if not any(abs(c - b) > 1e-12 for c, b in zip(cond_lp, base_lp)):
        print("[WARN] cond==base everywhere; labels likely degenerate (mostly null). "
              "Try --ib-max-contexts↑ or lower --n.")
 
    delta_hash_arr = [hash_lp[i] - base_lp[i] for i in range(len(test))]

    d_mean      = sum(delta_arr)      / len(delta_arr)
    d_hash_mean = sum(delta_hash_arr) / len(delta_hash_arr)

    boot    = block_bootstrap_mean(delta_arr,      B=args.bootstrap, block_len=args.block_len, rng=random.Random(0))
    boot_h  = block_bootstrap_mean(delta_hash_arr, B=args.bootstrap, block_len=args.block_len, rng=random.Random(1))

    # coverage (IB only)
    if label_mode == "ib" and mapping is not None:
        cov_train = ib_coverage(train, args.k_ctx, mapping, suffix)
        cov_test  = ib_coverage(test,  args.k_ctx, mapping, suffix)
    else:
        cov_train = {"full_frac": None, "backoff_frac": None, "null_frac": None}
        cov_test  = {"full_frac": None, "backoff_frac": None, "null_frac": None}

    if label_mode == "ib" and mapping is not None:
        if cov_train.get("full_frac", 1.0) < 0.04 or cov_test.get("full_frac", 1.0) < 0.04:
            print("[hint] IB coverage is very low; consider --k_ctx 3 or --ib-clusters 16, "
                  "or lower --ib-max-contexts.", file=sys.stderr)


    report = {
        "H_base": H_base,
        "H_ib": H_cond,      # name kept for BC
        "H_hash": H_hash,
        "delta_ib":   {"mean": d_mean,      "lo95": boot["lo95"],   "hi95": boot["hi95"],   "p_le_0": boot["p_le_0"]},
        "delta_hash": {"mean": d_hash_mean, "lo95": boot_h["lo95"], "hi95": boot_h["hi95"], "p_le_0": boot_h["p_le_0"]},
        "k_ctx": args.k_ctx,
        "ib_clusters": args.ib_clusters,
        "stride": args.stride,
        "label_mode": label_mode,
        "null_label": null_label,
        "n_clusters": n_clusters,
        "n_gram_order": N,
        "coverage_train": cov_train,
        "coverage_test":  cov_test,
        "notes": "Bits/token. Δ<0 means conditional helps. Hash control preserves label histogram.",
    }
    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


# ------------------------------- analyze ------------------------------

    # ---------------- report Ω-positivity summary ----------------
    if len(results) > 0:
        import statistics as _stats, json as _json
        # Primary record is the first tag (true or first control)
        primary = results[0]
        print("\n[Ω] summary (first variant):")
        print(f"  tag={primary['tag']}  ΔI_pred≈{primary['delta_ib']:.6f}  Δhash≈{primary['delta_hash']:.6f}  SCR_lzma≈{primary['scr_lzma']:.3e}  energy={primary['energy_lzma']}")
        if getattr(args, "report_json", None):
            _write_text(args.report_json, _json.dumps({"results": results}, indent=2))
            print(f"[report] Ω summary JSON: {args.report_json}")
def cmd_analyze(argv: Optional[List[str]] = None) -> None:
    """
    Core alphabetization analysis (IB stack + BPE) with MDL/prediction/efficiency,
    optional bootstrap over B block-resampled replicates.
    """
    p = argparse.ArgumentParser(prog="omega-analyze", add_help=True)
    p.add_argument("--train", required=True, help="Path to training text (use '-' for STDIN).")
    p.add_argument("--codex", help="Optional codex text (for NCD).")
    p.add_argument("--decoy", help="Optional decoy text (for NCD).")
    p.add_argument("--mode", choices=["words", "chars"], default="words")
    p.add_argument("--k-base", type=int, default=1)
    p.add_argument("--k-ctx", type=int, default=3)
    p.add_argument("--ib-clusters", default="64,32,16,8")
    p.add_argument("--bpe-merges", default="50,100,200")
    p.add_argument("--randomized", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", help="Write metrics report (JSON).")
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--block-len", type=int, default=64)
    p.add_argument("--out-bootstrap", help="Write bootstrap summaries (JSON).")
    p.add_argument("--dump-boot-reps", help="Write per-replicate metric lists (JSON).")
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--delta-target", help="Layer label for paired Δ (e.g., 'IB_K=27').")
    p.add_argument("--delta-vs", help="Baseline label (default 'raw(k=1)')")
    args = p.parse_args(argv)

    cfg = AlphaRunConfig(
        mode=args.mode,
        k_base=args.k_base, k_ctx=args.k_ctx,
        ib_clusters=_parse_int_list(args.ib_clusters, (64, 32, 16, 8)),
        bpe_merges=_parse_int_list(args.bpe_merges, (50, 100, 200)),
        randomized=args.randomized, seed=args.seed,
    )

    train_text = _read_text(args.train)
    codex_text = _read_text(args.codex) if args.codex else None
    decoy_text = _read_text(args.decoy) if args.decoy else None

    rep = run_alphabetization(train_text, codex_text, decoy_text, cfg=cfg)
    print("\n=== Alphabetization Layers (MDL / Prediction / Efficiency) ===")
    print(summarize_layers(rep.layers))
    if rep.ncds:
        print("\nNCDs:", rep.ncds)
    if args.out_json:
        _write_text(args.out_json, as_json(rep))
        print(f"\n[report] JSON: {args.out_json}")

    # Optional: bootstrap on B resamples of the TRAIN stream
    if args.bootstrap and args.bootstrap > 0:
        print(f"\n[bootstrap] running B={args.bootstrap}, block_len={args.block_len} …", file=sys.stderr)
        from alphabet import _bootstrap_collect, _summarize_per_label, _paired_delta, _METRICS  # reuse internal helpers
        by_label = _bootstrap_collect(train_text, cfg, B=args.bootstrap,
                                      block_len=args.block_len, seed=args.seed + 1337,
                                      progress_every=args.progress_every)
        summary = _summarize_per_label(by_label)
        print("\n[bootstrap] per-label summaries (means & 95% CI):")
        for label in sorted(summary.keys()):
            row = summary[label]
            def fmtd(k):
                d = row.get(k, {})
                return f"{d.get('mean', float('nan')):.3g} [{d.get('lo95', float('nan')):.3g},{d.get('hi95', float('nan')):.3g}]"
            print(f"  - {label:>10} | I_pred {fmtd('I_pred')} | eff_lzma {fmtd('eff_lzma')} | eff_brotli {fmtd('eff_brotli')}")
        if args.delta_target:
            base_label = args.delta_vs or "raw(k=1)"
            target_label = args.delta_target
            delta = _paired_delta(by_label, base_label, target_label, _METRICS)
            print(f"\n[Δ-bootstrap] target='{target_label}' minus base='{base_label}' (mean, 95% CI, p(Δ<=0)):")
            for m in _METRICS:
                d = delta[m]
                print(f"  Δ{m:>10}: {d['mean']:.4g}  [{d['lo95']:.4g},{d['hi95']:.4g}]  p<=0={d['p_le0']:.3f}  n={int(d['n'])}")

def main(argv: Optional[List[str]] = None) -> None:
    if argv is None: argv = sys.argv[1:]
    if len(argv) >= 1 and argv[0] == "analyze-cond":
        analyze_cond(argv[1:])
    else:
        cmd_analyze(argv)

if __name__ == "__main__":
    main()
