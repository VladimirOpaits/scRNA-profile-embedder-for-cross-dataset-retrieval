import numpy as np
import pandas as pd
from selection import greedy_quantile
import crc_enrichment as ce

N = 12
K = 10
SEEDS = 30
QUANT = 0.75
MIN_MINORITY = 10


def pick_base(pool):
    ct = pool.groupby("assay").y.agg(["sum", "count"])
    ct["neg"] = ct["count"] - ct["sum"]
    ct["m"] = ct[["sum", "neg"]].min(1)
    return ct["m"].idxmax()


def endpoint(pool, D, held, base, arm, seeds):
    a0, aK = [], []
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        b = pool[pool.assay == base]
        o = pool[pool.assay != base]
        sp_ = rng.choice(b[b.y == 1].index.to_numpy(), N, replace=False)
        sn = rng.choice(b[b.y == 0].index.to_numpy(), N, replace=False)
        brp = np.setdiff1d(b[b.y == 1].index.to_numpy(), sp_)
        brn = np.setdiff1d(b[b.y == 0].index.to_numpy(), sn)
        rp = o[o.y == 1].index.to_numpy()
        rn = o[o.y == 0].index.to_numpy()
        if arm == "diverse":
            ap = greedy_quantile(D, list(sp_), list(rp), K, QUANT)
            an = greedy_quantile(D, list(sn), list(rn), K, QUANT)
        elif arm == "homogeneous":
            ap = list(rng.choice(brp, min(K, len(brp)), replace=False))
            an = list(rng.choice(brn, min(K, len(brn)), replace=False))
        else:
            ap = greedy_quantile(D, list(sp_), list(rp), K, QUANT)
            an = greedy_quantile(D, list(sn), list(rn), K, QUANT)
        km = min(K, len(ap), len(an))
        for k, store in [(0, a0), (km, aK)]:
            coh = np.concatenate([sp_, np.array(ap[:k], int), sn, np.array(an[:k], int)])
            r = ce.per_study_auc(pool.loc[coh, ce.SCOLS].to_numpy(),
                                 pool.loc[coh, "y"].to_numpy(), held, arm == "negctrl", rng)
            store.append(r["macro"])
    return np.mean(a0), np.mean(aK)


def main():
    s = ce.load()
    pt = pd.crosstab(s.study, s.sample_type)
    pt = pt[(pt.get("tumor", 0) > 0) & (pt.get("normal", 0) > 0)]
    pt["m"] = pt[["tumor", "normal"]].min(1)
    cands = pt[pt.m >= MIN_MINORITY].sort_values("m", ascending=False).index.tolist()
    print(f"leave-one-study-out over {len(cands)} paired studies (minority>={MIN_MINORITY})")
    print(f"{'held-out study':28s} {'ceil':>5s} {'base':>13s} {'base@0':>7s} "
          f"{'div@K':>6s} {'homo@K':>7s} {'neg@K':>6s}  div>homo")
    for S in cands:
        held = s[s.study == S].copy()
        pool = s[s.study != S].reset_index(drop=True)
        if held.y.nunique() < 2:
            continue
        D = ce.l2_distance_matrix(pool[ce.SCOLS].to_numpy())
        ceil = ce.ceilings(held).get(S, np.nan)
        base = pick_base(pool)
        d0, dK = endpoint(pool, D, held, base, "diverse", SEEDS)
        _, hK = endpoint(pool, D, held, base, "homogeneous", SEEDS)
        _, nK = endpoint(pool, D, held, base, "negctrl", SEEDS)
        flag = "YES" if dK - hK > 0.02 else ("~" if dK - hK > -0.02 else "NO")
        print(f"{S:28s} {ceil:5.2f} {base:>13s} {d0:7.2f} {dK:6.2f} {hK:7.2f} {nK:6.2f}   {flag} (+{dK-hK:.2f})")


if __name__ == "__main__":
    main()
