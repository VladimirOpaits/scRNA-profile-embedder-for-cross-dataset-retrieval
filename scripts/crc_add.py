import os
import numpy as np
import pandas as pd
from selection import greedy_farthest, greedy_quantile
import crc_enrichment as ce

N_POS = 15
N_NEG = 15
K = 15
SEEDS = 50
QUANT = 0.75


def cohort_curve(pool, D, held, arm, seed):
    rng = np.random.default_rng(seed)
    base = pool[pool.assay == ce.BASE_TECH]
    other = pool[pool.assay != ce.BASE_TECH]

    def pick(sub, n):
        return rng.choice(sub.index.to_numpy(), n, replace=False)

    sp_ = pick(base[base.y == 1], N_POS)
    sn = pick(base[base.y == 0], N_NEG)
    brp = np.setdiff1d(base[base.y == 1].index.to_numpy(), sp_)
    brn = np.setdiff1d(base[base.y == 0].index.to_numpy(), sn)
    rp = other[other.y == 1].index.to_numpy()
    rn = other[other.y == 0].index.to_numpy()

    if arm == "farthest":
        ap = greedy_farthest(D, list(sp_), list(rp), K)
        an = greedy_farthest(D, list(sn), list(rn), K)
    elif arm in ("quantile", "negctrl"):
        ap = greedy_quantile(D, list(sp_), list(rp), K, QUANT)
        an = greedy_quantile(D, list(sn), list(rn), K, QUANT)
    elif arm == "homogeneous":
        ap = list(rng.choice(brp, min(K, len(brp)), replace=False))
        an = list(rng.choice(brn, min(K, len(brn)), replace=False))
    else:
        ap = list(rng.choice(rp, K, replace=False))
        an = list(rng.choice(rn, K, replace=False))

    kmax = min(K, len(ap), len(an))
    rows = []
    for k in range(kmax + 1):
        coh = np.concatenate([sp_, np.array(ap[:k], int),
                              sn, np.array(an[:k], int)])
        Xtr = pool.loc[coh, ce.SCOLS].to_numpy()
        ytr = pool.loc[coh, "y"].to_numpy()
        a = ce.per_study_auc(Xtr, ytr, held, arm == "negctrl", rng)
        for st, v in a.items():
            rows.append((arm, seed, k, len(coh), st, v))
    return rows


def main():
    s = ce.load()
    held, pool = ce.split(s)
    D = ce.l2_distance_matrix(pool[ce.SCOLS].to_numpy())
    ceil = ce.ceilings(held)
    pb = pool[pool.assay == ce.BASE_TECH]
    print(f"ADD experiment | seed {N_POS}+{N_NEG}, add up to {K}/class (cohort {2*N_POS}->{2*N_POS+2*K})")
    print(f"base {ce.BASE_TECH}: {int((pb.y==1).sum())} tumor / {int((pb.y==0).sum())} normal "
          f"(homogeneous-add limited by this)")

    arms = ["quantile", "farthest", "random", "homogeneous", "negctrl"]
    rows = []
    for arm in arms:
        for sd in range(SEEDS):
            rows += cohort_curve(pool, D, held, arm, sd)
    res = pd.DataFrame(rows, columns=["arm", "seed", "k", "n", "study", "auc"])
    res.to_csv("crc_add_results.csv", index=False)

    g = res.groupby(["arm", "study", "k"]).auc.mean().reset_index()
    kmax = int(res.k.max())
    print(f"\n=== ADD: macro AUC vs cohort growth (auc@0 -> auc@{kmax}) ===")
    print(f"{'arm':12s} {'auc@0':>6s} {'auc@'+str(kmax):>7s} {'delta':>7s}  (ceiling macro {np.mean(list(ceil.values())):.2f})")
    for arm in arms:
        a0 = g[(g.arm == arm) & (g.study == "macro") & (g.k == 0)].auc.iloc[0]
        aK = g[(g.arm == arm) & (g.study == "macro") & (g.k == kmax)].auc.iloc[0]
        print(f"{arm:12s} {a0:6.2f} {aK:7.2f} {aK-a0:7.2f}")
    print("\ndiversity-over-volume = diverse delta - homogeneous delta")
    print("wrote crc_add_results.csv")


if __name__ == "__main__":
    main()
