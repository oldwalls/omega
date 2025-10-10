# Ω-Scanner Synthetic Validation Suite  
### Full 9-Core Canonical System Scan · 64-Run Statistical Validation  

This release constitutes the **complete empirical validation** of the Ω-Scanner methodology across the canonical 9 dynamical systems.  
Each system was tested under three regimes — **Native (no control)**, **Global Shuffle**, and **Block-B2 Shuffle** — to verify the Ω-Signal’s statistical and structural integrity.

---

##  Overview

Background: [`Alphabetization`](ALPHABETIZATION.md)

| Tier | Structural Regime | Description |
|:--|:--|:--|
| **No Control (Native)** | Fully ordered | True physical / algorithmic structure preserved. |
| **Global Shuffle** | Completely randomized | Serves as *Ω-Null* baseline (semantic entropy maximal). |
| **Block-B2 Shuffle** | Local perturbation | Intermediate validation tier retaining partial coherence. |

Each tier contains three parallel data layers:

- `/runs_*` — raw JSON outputs of Ω-Scanner analysis  
- `/features_*` — extracted statistical and semantic descriptors  
- `/figures_*` — visualized Δᵢᵦ distributions and sinusoidal residual fits  

---

##  Nine Canonical Model Functions

1. Lorenz Attractor (`lorenz63`)  
2. Standard Map (`standard_map`)  
3. Arnold Cat Map (`arnold_cat`)  
4. Logistic Map (`logistic`)  
5. Hénon Map (`henon`)  
6. Hénon–Heiles Hamiltonian (`hamiltonian`)  
7. Relativistic Aberration (`rel_aberration`)  
8. 1-D Ising Model (`ising1d`)  
9. 2-D Ising Model (`ising2d_fixed`)

Discussion: [`models`](Models.md)

---

##  Purpose of Validation

The **Ω-Scanner** estimates **Δᵢᵦ = H_base − H_IB**,  
measuring semantic predictivity relative to entropy.  

By evaluating Δᵢᵦ across shuffled and unshuffled data, we demonstrate:
- Substrate-invariance of Ω-signal  
- Stability across dynamical regimes  
- Monotonic decay of semantic charge under randomization  
- Empirical verification of the **Semantic Gradient Law**

##   Conclusions

Discussion: [`conclusions`](CONCLUSIONS.md)
