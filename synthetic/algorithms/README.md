# Algorithms – Synthetic Ω Scanner

This folder documents the working implementation of the Ω-Scanner.  

---

## 1. Scanner Workflow

The Ω-Scanner is a **conditional compression test**.  
It asks a simple question:

> Do latent labels (Z) learned by an Information Bottleneck (IB) clustering reduce the codelength of a sequence, compared to the base n-gram coder alone?

If **yes** (ΔI_pred < 0 with statistical significance), then the data contains predictive structure that is better explained by the IB-compressed labels than by chance.  
If **no** (ΔI_pred ≈ 0), then the system is inert with respect to Ω.

---

## 2. Core Components

- **Corpus Preparation [`corpus_maker.py`](./code/corpus_maker.py)**  
  - Tokenizes raw sequences into train / holdout sets.  
  - Supports shuffle and block controls to destroy long-range dependencies.  
  - Uses prefix/suffix splits for stationarity tests.

- **Conditional Coder [`cond_coder.py`](code/cond_coder.py)**  
  - Implements n-gram probability estimation.  
  - Base coder: H(x | context).  
  - Conditional coder: H(x | context, Z).  

- **Labelers [`labelers.py`](code/labelers.py)**  
  - Clusters high-dimensional contexts into latent states Z using Information Bottleneck.  
  - Control modes: shuffled labels, hashed labels.  
  - Null label “C-1” ensures coverage of degenerate cases.

- **Bootstrap Statistics [`bootstrap.py`](code/bootstrap.py)**  
  - Resamples token blocks to estimate confidence intervals.  
  - Produces ΔI_pred mean, CI, and p-value (fraction of samples ≤ 0).  
  - This is the guardrail against spurious signals.

- **Divergence Metric [`js_divergence.py`](code/js_divergence.py)**  
  - Uses Jensen-Shannon divergence to measure cluster separation.  
  - Key step in merging clusters and preventing label collapse.

---

## 3. Output

Each run produces:

- **Entropy values** (H_base, H_ib, H_hash).  
- **Predictive information delta** (ΔI_pred).  
- **95% confidence interval**.  
- **Control Δ values** for shuffled corpora.  

A run is only considered Ω-valid if:

1. ΔI_pred < 0.  
2. The 95% CI is fully below zero.  
3. Controls return Δ ≈ 0.  

---

## 4. Implementation Notes

- Recent patches reduced **quadratic memory growth** in clustering, making long runs feasible on commodity hardware.  
- Progress output was added to track bootstrap sampling.  
- Results are saved in JSON for reproducibility and audit.  

---

## 5. Why This Works

The method is deliberately conservative.  
It cannot “hallucinate” an Ω-signal on unstructured data: shuffled controls consistently collapse ΔI_pred to ~0.  

When a structured system like the **Standard Map** produces a robust Ω-negative result, it means that predictive information is being captured that is not available to the base model alone.  

That is the Ω signature.

---

### 6. Extended Reference Material

For those seeking deeper operational and methodological detail, the **Extended Reference** provides a full technical exposition of the Ω-Scanner framework.
It includes:

* Annotated configuration parameters for `cli` and `corpus_maker`, with example command-line runs.
* Step-wise execution flow of the Information Bottleneck clustering and conditional entropy estimation.
* Code snippets illustrating corpus preparation, bootstrap routines, and ΔI_pred computation.
* Practical notes on performance limits and reproducibility settings.

[**View Extended Documentation →**](EXTENDED.md)

---

