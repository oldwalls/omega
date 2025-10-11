# Announcement

### This repository serves as the central hub for the implementation and validation of the Logos Omega Gradient framework.

---

# DATA RELEASE: Ω-SCANNER MODEL MAPPING PHASE

The complete dataset from the Synthetic Validation Suite (SVS) Phase  is currently being ingested into the dedicated mapping/synthetic repository. This release validates the Ω-Scanner Statistical Methodology for substrate-invariance and structural hierarchy detection across nine canonical dynamical systems.

Each system below was subjected to 192 independent Ω-map runs, including runs against the 
​  Global Shuffle and Block Shuffle.

## Nine Synthetic Systems in Release:

- Lorenz Attractor (lorenz63)

- Standard Map (standard_map)

- Arnold Cat Map (arnold_cat)

- Logistic Map (logistic)

- Hénon Map (henon)

- Hénon-Heiles Hamiltonian (hamiltonian)

- Relativistic Aberration (rel_aberration)

- 1D Ising Model (ising1d)

- 2D Ising Model (ising2d_fixed)

---

Deliverables Per Function:
Raw Data: Raw ΔΩ metric output for all 192 * 9 models (1728 independent Omega Scanner runs)
Summary Data: Consolidated statistical metrics: Mean, Median, σ, and fit parameters  
Distribution Plots: Visual distribution of the ΔΩ effect across the 192 * 9 model runs

Repo Location: [`maps`](maps)

Initial Testing - single runs: [`synthetic`](synthetic) 

---

Thank you for your patience and interest in the scientific validation of LOG.

---

### From Synthetic Omega to Unified Physics

[`Telos`](https://github.com/oldwalls/omega/tree/main/TELOS)

---
<img width="420" alt="image" src="https://github.com/user-attachments/assets/3718558f-f7fb-4e6d-a887-9b87bc2baa82" />


# Logos Omega Gradient (Ω) — Priority Marker

**Abstract: 2025-09-08**

This document records the initial statement of the **Logos Omega Gradient (Ω)** hypothesis and its first computational evidence.

---

### Formal Information-Theoretic Basis (Gemini formulation, abridged)

Let X be a raw token stream and Z = φ(X) a derived alphabetized stream.

* **Predictive Information:**
  I(Y; C) = H(Y) - H(Y | C)
  (The reduction in uncertainty about a future token Y given a context C.)

* **Information Efficiency:**
  η(S) = I(Y; C) / Hμ(S)
  (Ratio of predictive information to entropy rate, i.e. bits per token.)

**Criterion for Ω-positivity:**
A transformation φ is Ω-positive if both of these are true (with 95% confidence):

* Δ I\_pred > 0
* Δ η > 0

---

### Empirical Results (joint synthesis)

* IB-layer runs (K=32 clusters) show simultaneous gains in predictive information and efficiency, with bootstrapped 95% confidence intervals strictly greater than 0.
* Topic-label and shuffled controls do not show this dual gain, confirming that the signal is specific, not an artifact.
* Null runs (random labels) return no Ω-signal, strengthening the conclusion.

Together, these results satisfy the Ω-positive criterion at the symbolic level:
alphabetization itself tilts noisy streams toward **sense-bearing compact codes**.

---

### Statement of Record

* This repo contains the first recorded computational confirmation of the **Logos Omega Gradient**.
* The working codebase will be published once it reaches sufficient quality for independent replication.
* This marker establishes **priority of idea and implementation path**.

---

### Expanded README


[`Expanded`](https://github.com/oldwalls/omega/blob/main/README_EXPANDED.md)

---

