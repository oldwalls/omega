## 1. Lorenz Attractor (`lorenz63`)

**Type:** Continuous 3-dimensional deterministic system
**Equations:**

$$  
\dot{x} = \sigma(y - x), \quad
\dot{y} = x(\rho - z) - y, \quad
\dot{z} = xy - \beta z
$$

**Signature:** Classic *strange attractor* — bounded but non-periodic.
**Entropy / Predictive Structure:**

* Moderate Shannon entropy; high *predictive mutual information* (long memory).
* Alphabetization reveals laminar regions alternating with rapid “wing hops.”
* Ideal for testing whether the Ω-scanner can capture **deterministic chaos with smooth continuity**.
  **Expected ΔIB:** strong positive (>0) at small context windows (k≈3–5).

---

## 2. Standard Map (`standard_map`)

**Type:** Area-preserving discrete map on the torus

$$
p_{t+1} = p_t + K \sin(\theta_t), \quad
\theta_{t+1} = \theta_t + p_{t+1} \pmod{2\pi}
$$

**Signature:** Mixed chaotic–regular phase space; quasi-periodic tori + stochastic sea.
**Entropy / Predictive Structure:**

* When K ≲ 1: ordered resonance islands (negative ΔIB possible).
* When K ≫ 1: fully chaotic, yielding large Ω-positive signals.
* Sensitive to **micro-order within noise** — confirms alphabetization’s ability to extract surviving predictivity.
  **Expected ΔIB:** strongly positive even under partial randomization (confirmed empirically).

---

## 3. Arnold Cat Map (`arnold_cat`)

**Type:** Linear hyperbolic automorphism on the 2-torus

$$
\begin{pmatrix}
x_{t+1} \\
y_{t+1}
\end{pmatrix}
= A
\begin{pmatrix}
x_t \\
y_t
\end{pmatrix}
\bmod 1,
\quad
A =
\begin{pmatrix}
2 & 1 \\
1 & 1
\end{pmatrix}
$$

**Signature:** Fully chaotic but *structurally exact*; preserves area and entropy.
**Entropy / Predictive Structure:**

* Maximal mixing under integer lattice but predictable under modular arithmetic.
* Ideal test of **symbolic chaos with algebraic determinism**.
* Ω-scanner should yield ΔIB > 0 only if alphabet captures modular periodicity; otherwise ≈ 0.

---

## 4. Logistic Map (`logistic`)

**Type:** One-dimensional iterative map

$$
x_{t+1} = r x_t (1 - x_t)
$$

**Signature:** Bifurcation cascade from fixed point → periodic → chaos.
**Entropy / Predictive Structure:**

* Near bifurcation points: moderate predictive gain (deterministic).
* In deep chaos (r ≈ 4): nearly i.i.d., ΔIB ≈ 0.
* Acts as a **control gradient** across entropy regimes.
  **Expected ΔIB:** rises then drops as r increases — non-monotonic Ω-profile.

---

## 5. Hénon Map (`henon`)

**Type:** 2-D dissipative discrete system

$$
x_{t+1} = 1 - a x_t^2 + y_t,\quad y_{t+1} = b x_t
$$

**Signature:** Strange attractor; folding-stretching dynamics.
**Entropy / Predictive Structure:**

* Intermediate between Lorenz and Logistic in complexity.
* Strong local predictability; sensitive to alphabet granularity.
* Benchmarks **low-dimensional chaos with memory depth ≈ 2**.

---

## 6. Hénon–Heiles Hamiltonian (`hamiltonian`)

**Type:** Nonlinear Hamiltonian system with mixed regular and chaotic regions

$$
H = \tfrac12(p_x^2+p_y^2) + \tfrac12(x^2+ y^2) + x^2 y - \tfrac13 y^3
$$

**Signature:** Transition to chaos as total energy increases.
**Entropy / Predictive Structure:**

* Periodic islands embedded in chaotic sea.
* Tests alphabetization under **continuous-to-discrete symbolic projection**.
* ΔIB should correlate with the measure of regular islands (moderate positive values).

---

## 7. Relativistic Aberration (`rel_aberration`)

**Type:** Deterministic transformation of direction cosines under velocity addition

$$
\cos\theta' = \frac{\cos\theta - \beta}{1 - \beta \cos\theta}
$$

**Signature:** Smooth nonlinear mapping, no intrinsic chaos but strong nonlinearity.
**Entropy / Predictive Structure:**

* Fully ordered; monotonic distortion of signal.
* Serves as **smooth-nonchaotic baseline**; expected ΔIB ≈ 0.
* Used to confirm Ω-scanner does not produce false positives on purely geometric transforms.

---

## 8. 1-D Ising Model (`ising1d`)

**Type:** Stochastic spin chain with nearest-neighbor coupling

$$
P({s_i}) \propto e^{\beta \sum_i s_i s_{i+1}}
$$

**Signature:** Markov order 1; no long-range correlations.
**Entropy / Predictive Structure:**

* Simple exponential decay of mutual information.
* Ideal “short-memory null” system.
* ΔIB ≈ 0 except near critical coupling where correlation length → ∞.

---

## 9. 2-D Ising Model (`ising2d_fixed`)

**Type:** Lattice spin model; critical phenomena at T ≈ 2.269 J/kB.
**Signature:** Non-chaotic but **critical** — correlation length diverges.
**Entropy / Predictive Structure:**

* Predictive information peaks sharply at T_c.
* In Synthetic Ω, serves as the **statistical null validation substrate** (global shuffle → Δ≈0; ordered → slight positive).
* Demonstrates that stochastic order without chaos still yields structured predictivity near phase transition.

---

###  Summary Table

| Model           |    Dim.   |  Determinism  | Chaos/Order Type             | Expected ΔIB | Role in Suite               |
| :-------------- | :-------: | :-----------: | :--------------------------- | :----------: | :-------------------------- |
| Lorenz63        |     3     | deterministic | continuous chaotic attractor |      ++      | smooth chaotic benchmark    |
| Standard Map    |     2     | deterministic | mixed chaotic–regular        |      +++     | robust Ω-positive reference |
| Arnold Cat      |     2     | deterministic | linear chaotic               |      +/0     | symbolic mixing control     |
| Logistic        |     1     | deterministic | bifurcating → chaotic        |   variable   | entropy-gradient test       |
| Hénon           |     2     | deterministic | strange attractor            |      ++      | discrete chaos reference    |
| Hénon–Heiles    | 4 (phase) | deterministic | mixed Hamiltonian            |       +      | regular–chaos transition    |
| Rel. Aberration |     1     | deterministic | smooth nonlinear, nonchaotic |       0      | geometric baseline          |
| Ising 1D        |     1     |   stochastic  | Markov chain                 |       0      | simple stochastic null      |
| Ising 2D        |     2     |   stochastic  | critical lattice             |      0/+     | statistical null validation |

---

 **Summary narrative**

Together these nine canonical systems span the full topology of *information organization*:
from smooth deterministic chaos (Lorenz, Hénon) through mixed maps (Standard, Hénon–Heiles, Arnold) to purely stochastic ensembles (Ising).
Each contributes a distinct **entropy–predictivity profile** for calibrating the Ω-scanner:

* **Chaotic deterministic** → ΔIB ≫ 0 (true predictivity in disorder).
* **Regular deterministic** → ΔIB < 0 (over-compressed).
* **Random/stochastic** → ΔIB ≈ 0 (null).

This diversity makes the Synthetic Ω validation set statistically complete across entropy regimes — from *pure order* to *maximal chaos* — ensuring that every Ω-positive result is interpretable as genuine predictive structure, not artefact.
