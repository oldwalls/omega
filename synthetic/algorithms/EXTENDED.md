# EXTENDED – Ω-Scanner: Methods, Parameters, and Execution

This document explains **how the scanner is used in practice**.  
Two goals:
1) expose the **corpus maker + CLI parameters** by example,  
2) document the **stepwise execution** with code/command snippets.

Everything here reflects what we actually ran for the Standard Map result and its controls.


---

## 1) Corpus maker: how corpora are prepared

We keep it simple: each corpus directory contains
- `*_train.txt`  → training stream (prefix)
- `*_tail.txt`   → holdout stream (suffix)
- controls:
  - `*_shuf.txt`     → globally shuffled training stream
  - `*_tail_B9.txt`  → tail block-shuffled with block size 9

### 1.1 Create a block-shuffled tail (control)
Use the included helper:

```bash
python -m block_shuffle   --input corpora/stdmap/stdmap_tail.txt   --block 9   --output corpora/stdmap/stdmap_tail_B9.txt
```

### 1.2 Create a globally shuffled train (control)
You can do this once with a small Python snippet (or your own tool):

```python
# shuf_stdmap.py
import random, sys
random.seed(101)
inp, outp = sys.argv[1], sys.argv[2]
with open(inp, "r", encoding="utf-8") as f:
    toks = f.read().split()
random.shuffle(toks)
with open(outp, "w", encoding="utf-8") as f:
    f.write(" ".join(toks))
```

Run:

```bash
python shuf_stdmap.py corpora/stdmap/stdmap_train.txt corpora/stdmap/stdmap_shuf.txt
```

**Sanity check:** the shuffled corpus should have **much higher** base entropy (H_base) than the structured train (e.g., ~3.36 vs ~1.84 bits/token for Stdmap). That confirms structure destruction.

---

## 2) CLI usage (analyze-cond): parameters by example

All scanner runs use the same entry point:

```
python -m cli analyze-cond   --train <path/to/train.txt>   --holdout <path/to/tail.txt>   --mode words   --label-mode ib   --n <order>   --k_ctx <context>   --ib-clusters <k>   --stride 1   --seed <int>   --bootstrap <B>   --alpha 0.05   [--ib-max-contexts <cap>]   --out-json <path/to/output.json>
```

### Parameter meanings (straightforward)
- `--train` / `--holdout`  
  Paths to the train prefix and tail suffix (tokenized, whitespace-separated).

- `--mode words`  
  Treat tokens as words/symbols (our default).

- `--label-mode ib`  
  Use IB clustering to produce latent labels Z.

- `--n`  
  n-gram order for the **base** coder (H_base) and the conditional coder (H_ib).

- `--k_ctx`  
  Markov context depth (how many past tokens form the context).

- `--ib-clusters`  
  Target number of IB clusters (latent states Z). Practical range 16–64 for our corpora.

- `--ib-max-contexts` (optional, resource guard)  
  Caps the number of distinct contexts passed into clustering.  
  Use when RAM is tight (e.g., 6000–12000). Omitting it uses all contexts.

- `--seed`  
  RNG seed for reproducibility (IB and bootstrap sampling).

- `--bootstrap`  
  Number of bootstrap resamples for ΔI_pred CI. Typical: 1500–4000 (fast), 7000+ (tight).

- `--alpha`  
  Confidence level for intervals (0.05 → 95% CI).

- `--out-json`  
  Where to write the full metrics report.

---

## 3) Step-by-step execution recipes

### 3.1 Standard Map — validated Ω-negative run

```bash
python -m cli analyze-cond ^
  --train corpora/stdmap/stdmap_train.txt ^
  --holdout corpora/stdmap/stdmap_tail.txt ^
  --mode words --label-mode ib ^
  --n 3 --k_ctx 5 --ib-clusters 32 ^
  --stride 1 --seed 101 ^
  --bootstrap 2000 --alpha 0.05 ^
  --ib-max-contexts 6000 ^
  --out-json runs/stdmap_ctx3_k5_cl32_ctx6k_s101.json
```

What to expect in the JSON:
- `H_base ≈ 1.842`, `H_ib ≈ 1.835`, `H_hash ≈ 1.888`
- `delta_ib.mean ≈ -0.00663`
- `delta_ib.lo95 < 0` and `delta_ib.hi95 < 0`
- `p_le_0 = 1.0`
- Coverage (train close to 1.0 full, tail ~0.95 full), minimal backoff.

**Interpretation rule:** Ω-negative if `ΔI_pred (delta_ib.mean) < 0` **and** entire 95% CI below 0.  
Controls must be null (next section) to count as validated.

### 3.2 Standard Map — shuffle control (null)

```bash
python -m cli analyze-cond ^
  --train corpora/stdmap/stdmap_shuf.txt ^
  --holdout corpora/stdmap/stdmap_tail_B9.txt ^
  --mode words --label-mode ib ^
  --n 3 --k_ctx 5 --ib-clusters 32 ^
  --ib-max-contexts 6000 --stride 1 --seed 101 ^
  --bootstrap 800 --alpha 0.05 ^
  --out-json runs/stdmap_control_shuf.json
```

Expected outcome:
- `ΔI_pred ≈ 0.000014` (noise level)
- 95% CI **straddles 0**
- `p_le_0 ≈ 0.2`
- Tail coverage ~0 (almost all backoff) due to block shuffle — that’s fine for a null.

This demonstrates the scanner **does not** fabricate Ω signals on unstructured inputs.

---

## 4) Reading the JSON report

Each run writes a structured JSON with (at least):

- `H_base`, `H_ib`, `H_hash` (bits/token)
- `delta_ib`  
  - `mean` → ΔI_pred  
  - `lo95`, `hi95` → 95% CI  
  - `p_le_0` → fraction of bootstrap samples ≤ 0
- `delta_hash` → control comparison using hash-preserving labels
- `coverage_train` / `coverage_test`  
  - `full_frac`   → exact n-gram context match rate  
  - `backoff_frac`→ backed off to shorter context  
  - `null_frac`   → null paths / unknowns
- `notes` → reminder that Δ<0 means conditional helps

**Quick triage**
- If `ΔI_pred < 0` and CI entirely below 0 → candidate Ω-negative.
- Controls must show Δ ≈ 0 to validate.

---

## 5) Programmatic usage (minimal snippet)

If you prefer to call pieces from Python (e.g., to batch runs), here’s the skeleton:

```python
from cond_coder import NGramCoder
from labelers import ib_labeler
from bootstrap import estimate_delta
from load import load_tokens

# 1) Load tokens
train = load_tokens("corpora/stdmap/stdmap_train.txt")
tail  = load_tokens("corpora/stdmap/stdmap_tail.txt")

# 2) Build base coder
base = NGramCoder(n=3, k_ctx=5)
base.fit(train)

# 3) Build IB labels on train
Z_train = ib_labeler(train, n_clusters=32, max_contexts=6000, seed=101)
# Map/derive labels for tail as needed (same pipeline/contexts)

# 4) Build conditional coder (context + Z)
cond = NGramCoder(n=3, k_ctx=5, labels=Z_train)
cond.fit(train)

# 5) Bootstrap ΔI_pred on tail
report = estimate_delta(base, cond, tail, B=2000, alpha=0.05, seed=101)
print(report)
```

Use the CLI for official runs; programmatic calls are handy for quick sweeps.

---

## 6) Reproducibility and stability notes

- **Seeds.** Always run at least two seeds for any claimed result (e.g., 101, 202).  
- **Bootstrap depth.** Start with B=1500–2000; tighten to B=4000–7000 for final CIs.  
- **Resource guard (`--ib-max-contexts`).** Use a cap (e.g., 6000–12000) if RAM is tight; it shouldn’t flip the sign in healthy regimes.  
- **Determinism.** If you patch HAC/heap to be tie-stable (we did), runs will match across seeds/hosts for a given configuration.

---

## 7) Minimal recipes (we actually use)

**Validated Stdmap run**
```bash
python -m cli analyze-cond ^
  --train corpora/stdmap/stdmap_train.txt ^
  --holdout corpora/stdmap/stdmap_tail.txt ^
  --mode words --label-mode ib ^
  --n 3 --k_ctx 5 --ib-clusters 32 ^
  --stride 1 --seed 101 ^
  --bootstrap 2000 --alpha 0.05 ^
  --ib-max-contexts 6000 ^
  --out-json runs/stdmap_ctx3_k5_cl32_ctx6k_s101.json
```

**Null control**
```bash
python -m cli analyze-cond ^
  --train corpora/stdmap/stdmap_shuf.txt ^
  --holdout corpora/stdmap/stdmap_tail_B9.txt ^
  --mode words --label-mode ib ^
  --n 3 --k_ctx 5 --ib-clusters 32 ^
  --ib-max-contexts 6000 --stride 1 --seed 101 ^
  --bootstrap 800 --alpha 0.05 ^
  --out-json runs/stdmap_control_shuf.json
```

**Block shuffle helper**
```bash
python -m block_shuffle   --input corpora/stdmap/stdmap_tail.txt   --block 9   --output corpora/stdmap/stdmap_tail_B9.txt
```

---

## 8) Decision rule (what “counts”)

A run is accepted as Ω-negative **only if**:
1) ΔI_pred < 0,  
2) the 95% CI is entirely below 0,  
3) controls (global/block shuffles) return Δ ≈ 0.

The Standard Map substrate meets these criteria and is recorded in `results/RESULTS.md`.


