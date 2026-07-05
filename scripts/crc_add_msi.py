import numpy as np
import pandas as pd
from selection import greedy_farthest, greedy_quantile
from crc_enrichment import l2_distance_matrix
import crc_msi as cm

N_POS = 8
N_NEG = 8
K = 8
SEEDS = 50
QUANT = 0.75


def cohort_curve(pool, D, held, arm, seed):
    rng = np.random.default_rng(seed)
    base = pool[pool.assay == cm.BASE_TECH]
    other = pool[pool.assay != cm.BASE_TECH]

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
        coh = np.concatenate([sp_, np.array(ap[:k], int), sn, np.array(an[:k], int)])
        Xtr = pool.loc[coh, cm.SCOLS].to_numpy()
        ytr = pool.loc[coh, "y"].to_numpy()
        a = cm.per_study_auc(Xtr, ytr, held, arm == "negctrl", rng)
        for st, v in a.items():
            rows.append((arm, seed, k, st, v))
    return rows


def main():
    s = cm.load()
    held = s[s.study.isin(cm.HELD)].copy()
    pool = s[~s.study.isin(cm.HELD)].reset_index(drop=True)
    D = l2_distance_matrix(pool[cm.SCOLS].to_numpy())
    ceil = cm.ceilings(held)
    pb = pool[pool.assay == cm.BASE_TECH]
    print(f"MSI ADD | seed {N_POS}+{N_NEG}, add up to {K}/class | base {cm.BASE_TECH}: "
          f"{int((pb.y==1).sum())} MSI / {int((pb.y==0).sum())} MSS | ceiling macro {np.mean(list(ceil.values())):.2f}")

    arms = ["quantile", "farthest", "random", "homogeneous", "negctrl"]
    rows = []
    for arm in arms:
        for sd in range(SEEDS):
            rows += cohort_curve(pool, D, held, arm, sd)
    res = pd.DataFrame(rows, columns=["arm", "seed", "k", "study", "auc"])
    res.to_csv("crc_add_msi_results.csv", index=False)
    km = int(res.k.max())

    g = res[res.study == "macro"].groupby(["arm", "k"]).auc.agg(["mean", "std"]).reset_index()
    print(f"\n=== MSI ADD: macro AUC (mean, sd) vs k ===")
    print("  k  " + "".join(f"{a:>14}" for a in arms))
    for k in range(km + 1):
        cells = []
        for a in arms:
            row = g[(g.arm == a) & (g.k == k)]
            cells.append(f"{row['mean'].iloc[0]:.2f}±{row['std'].iloc[0]:.2f}" if len(row) else "-")
        print(f"  {k:>2} " + "".join(f"{c:>14}" for c in cells))

    def ser(a):
        return res[(res.arm == a) & (res.study == "macro") & (res.k == km)].set_index("seed").auc
    print(f"\n=== paired macro@k={km} (mean, frac>0) ===")
    for a, b in [("quantile", "homogeneous"), ("quantile", "random"), ("farthest", "random")]:
        d = (ser(a) - ser(b)).dropna()
        print(f"  {a:9s} - {b:11s}: {d.mean():+.3f}  frac>0 {(d>0).mean():.0%}")
    print("wrote crc_add_msi_results.csv")


if __name__ == "__main__":
    main()
