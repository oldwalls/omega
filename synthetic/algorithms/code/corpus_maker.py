# corpus_maker.py
# Physics-first corpus generator for LOG-GUT Ω-scanner
# Datasets: ising1d, standardmap, rel_aberration
# Outputs: {tag}_train.txt, {tag}_tail.txt, {tag}_shuf.txt,
#          {tag}_tail_B7.txt, {tag}_tail_B9.txt, {tag}_meta.json

import argparse, json, os, math
from pathlib import Path
import numpy as np

# ------------------------- utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def block_shuffle(arr, B, rng):
    if B <= 0:
        return arr.copy()
    n = len(arr)
    nb = n // B
    trimmed = arr[: nb * B].reshape(nb, B)
    order = np.arange(nb)
    rng.shuffle(order)
    shuffled = trimmed[order].reshape(-1)
    tail = arr[nb * B:]
    return np.concatenate([shuffled, tail]) if len(tail) else shuffled

def quantize_stream(x, bins, edges=None):
    x = np.asarray(x, dtype=float)
    if edges is None:
        # equal-frequency quantiles -> stable across long-tailed signals
        qs = np.linspace(0, 1, bins + 1)
        edges = np.quantile(x, qs)
        # guard: merge duplicate edges by nudging
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = np.nextafter(edges[i-1], float('inf'))
    # map to 0..bins-1
    sym = np.searchsorted(edges, x, side="right") - 1
    sym = np.clip(sym, 0, bins - 1)
    return sym.astype(np.int32), edges

def write_symbol_files(tag_dir, tag, syms, bins, train_frac, rng, meta):
    N = len(syms)
    n_train = int(N * train_frac)
    train = syms[:n_train]
    tail  = syms[n_train:]

    # main files
    def dump(name, arr):
        with open(tag_dir / name, "w", encoding="utf-8") as f:
            f.write(" ".join(map(str, arr.tolist())))

    dump(f"{tag}_train.txt", train)
    dump(f"{tag}_tail.txt",  tail)

    # controls
    train_shuf = train.copy()
    rng.shuffle(train_shuf)
    dump(f"{tag}_shuf.txt", train_shuf)

    tail_B7 = block_shuffle(tail, 7, rng)
    tail_B9 = block_shuffle(tail, 9, rng)
    dump(f"{tag}_tail_B7.txt", tail_B7)
    dump(f"{tag}_tail_B9.txt", tail_B9)

    # meta
    with open(tag_dir / f"{tag}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def report_quick_stats(arr, bins_label):
    arr = np.asarray(arr)
    counts = np.bincount(arr, minlength=arr.max()+1)
    return {
        "len": int(arr.size),
        "min_sym": int(arr.min()),
        "max_sym": int(arr.max()),
        "bincount_first10": counts[:10].tolist(),
        "unique_syms": int(np.unique(arr).size),
        "bins_label": bins_label
    }

# ------------------------- generators -------------------------

def gen_ising1d(steps, n, beta_min, beta_max, sweep_period, rng):
    """
    1D Ising, periodic, single-spin Metropolis. Record total magnetization m_t.
    Beta(t) sweeps between [beta_min, beta_max] in a triangle wave with period.
    Output: float sequence m_t in [-1, 1].
    """
    spins = rng.choice([-1, 1], size=n)
    # precompute neighbor indices
    left  = np.arange(n) - 1
    left[0] = n - 1
    right = (np.arange(n) + 1) % n

    def beta_at(t):
        # triangle sweep 0..1..0
        phase = (t % sweep_period) / sweep_period
        if phase <= 0.5:
            a = 2*phase
        else:
            a = 2*(1 - phase)
        return beta_min + (beta_max - beta_min) * a

    out = np.empty(steps, dtype=float)
    for t in range(steps):
        beta = beta_at(t)
        # one Monte Carlo sweep ~ n proposals
        for _ in range(n):
            i = rng.integers(0, n)
            dE = 2 * spins[i] * (spins[left[i]] + spins[right[i]])
            if dE <= 0 or rng.random() < math.exp(-beta * dE):
                spins[i] *= -1
        out[t] = spins.mean()  # total magnetization per spin
    return out

def gen_standardmap(steps, K_schedule, plateau, rng, p0=None, theta0=None):
    """
    Chirikov standard map. Record p_t (wrapped to [-pi, pi]).
    K_schedule: list of K values, each lasting 'plateau' steps.
    """
    if p0 is None:     p = (rng.random() - 0.5) * 2 * math.pi
    else:              p = float(p0)
    if theta0 is None: th = (rng.random()) * 2 * math.pi
    else:              th = float(theta0)

    out = np.empty(steps, dtype=float)
    Ks = []
    for K in K_schedule:
        Ks += [float(K)] * plateau
    Ks = np.array(Ks[:steps])

    for t in range(steps):
        K = Ks[t]
        p = p + K * math.sin(th)
        # wrap p to [-pi, pi]
        p = (p + math.pi) % (2 * math.pi) - math.pi
        th = (th + p) % (2 * math.pi)
        out[t] = p
    return out

def gen_rel_aberration(total_samples, beta_max, beta_plateau, theta_pts):
    """
    Special-relativistic Doppler factor D(θ,β)=gamma*(1-β cosθ)^(-1).
    Build a β ramp 0 -> beta_max in plateaus (beta_plateau samples each),
    for each β sample θ grid of size theta_pts; serialize β-slices.
    Output length ~ n_beta * theta_pts ~= total_samples.
    """
    n_beta = max(1, total_samples // theta_pts // beta_plateau)
    # Construct beta plateaus from 0->beta_max
    betas = []
    for i in range(1, n_beta + 1):
        b = beta_max * i / n_beta
        betas += [b] * beta_plateau
    betas = betas[: max(1, total_samples // theta_pts)]
    thetas = np.linspace(0, 2*math.pi, theta_pts, endpoint=False)

    vals = []
    for b in betas:
        gamma = 1.0 / math.sqrt(1.0 - b*b)
        # serialize θ within each β
        D = gamma / (1.0 - b * np.cos(thetas))
        vals.append(D.astype(float))
    if not vals:
        return np.array([], dtype=float)
    arr = np.concatenate(vals)
    # trim to total_samples exactly
    return arr[:total_samples]

# ------------------------- main flow -------------------------

def main():
    ap = argparse.ArgumentParser(description="Physics-first corpus maker (ising1d, standardmap, rel_aberration)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Common args helper
    def add_common(sp):
        sp.add_argument("--tag", required=True, help="dataset tag (folder/name prefix)")
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--steps", type=int, required=True, help="number of recorded samples")
        sp.add_argument("--bins", type=int, default=8, help="alphabet size (<=16 recommended)")
        sp.add_argument("--train_frac", type=float, default=0.7)
        sp.add_argument("--outdir", default="corpora")

    # ising1d
    sp1 = sub.add_parser("ising1d", help="1D Ising chain (magnetization time series)")
    add_common(sp1)
    sp1.add_argument("--n", type=int, default=128)
    sp1.add_argument("--beta_min", type=float, default=0.35)
    sp1.add_argument("--beta_max", type=float, default=0.47)
    sp1.add_argument("--sweep_period", type=int, default=2000)

    # standard map
    sp2 = sub.add_parser("standardmap", help="Chirikov standard map (quantized p_t)")
    add_common(sp2)
    sp2.add_argument("--K_schedule", type=str, default="0.5,1.0,2.0")
    sp2.add_argument("--plateau", type=int, default=100000)

    # relativistic aberration
    sp3 = sub.add_parser("rel_aberration", help="Relativistic Doppler factor over β–θ")
    add_common(sp3)
    sp3.add_argument("--beta_max", type=float, default=0.9)
    sp3.add_argument("--beta_plateau", type=int, default=500)
    sp3.add_argument("--theta_pts", type=int, default=64)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    tag = args.tag
    base = Path(args.outdir) / tag
    ensure_dir(base)

    # Generate raw float stream
    if args.cmd == "ising1d":
        raw = gen_ising1d(
            steps=args.steps, n=args.n,
            beta_min=args.beta_min, beta_max=args.beta_max,
            sweep_period=args.sweep_period, rng=rng
        )
        kind = "ising1d"
        params = dict(n=args.n, beta_min=args.beta_min, beta_max=args.beta_max,
                      sweep_period=args.sweep_period)

    elif args.cmd == "standardmap":
        K_sched = [float(x) for x in args.K_schedule.split(",") if x.strip()]
        raw = gen_standardmap(
            steps=args.steps, K_schedule=K_sched,
            plateau=args.plateau, rng=rng
        )
        kind = "standardmap"
        params = dict(K_schedule=K_sched, plateau=args.plateau)

    elif args.cmd == "rel_aberration":
        raw = gen_rel_aberration(
            total_samples=args.steps,
            beta_max=args.beta_max,
            beta_plateau=args.beta_plateau,
            theta_pts=args.theta_pts
        )
        kind = "rel_aberration"
        params = dict(beta_max=args.beta_max, beta_plateau=args.beta_plateau, theta_pts=args.theta_pts)

    else:
        raise ValueError("unknown cmd")

    # Quantize with consistent edges
    syms, edges = quantize_stream(raw, bins=args.bins, edges=None)

    meta = {
        "tag": tag,
        "kind": kind,
        "seed": args.seed,
        "steps": args.steps,
        "bins": args.bins,
        "train_frac": args.train_frac,
        "params": params,
        "quant_edges": list(map(float, edges)),
        "quick_stats": report_quick_stats(syms, f"{args.bins}-bin")
    }

    write_symbol_files(base, tag, syms, args.bins, args.train_frac, rng, meta)

    print(f"[ok] wrote corpus at {base}")
    print(json.dumps(meta["quick_stats"], indent=2))

if __name__ == "__main__":
    main()
