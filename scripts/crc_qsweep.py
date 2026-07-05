import numpy as np
import pandas as pd
from selection import greedy_quantile
import crc_enrichment as ce

QS = [0.50, 0.65, 0.75, 0.90, 1.00]
N = 15
K = 15
SEEDS = 50
SPEED_K = 5


def curve(pool, D, held, q, seed):
    rng = np.random.default_rng(seed)
    b = pool[pool.assay == ce.BASE_TECH]
    o = pool[pool.assay != ce.BASE_TECH]
    sp_ = rng.choice(b[b.y == 1].index.to_numpy(), N, replace=False)
    sn = rng.choice(b[b.y == 0].index.to_numpy(), N, replace=False)
    rp = o[o.y == 1].index.to_numpy()
    rn = o[o.y == 0].index.to_numpy()
    ap = greedy_quantile(D, list(sp_), list(rp), K, q)
    an = greedy_quantile(D, list(sn), list(rn), K, q)
    out = {}
    for k in (0, SPEED_K, K):
        coh = np.concatenate([sp_, np.array(ap[:k], int), sn, np.array(an[:k], int)])
        r = ce.per_study_auc(pool.loc[coh, ce.SCOLS].to_numpy(),
                             pool.loc[coh, "y"].to_numpy(), held, False, rng)
        out[k] = r["macro"]
    return out


def main():
    s = ce.load()
    held = s[s.study.isin(ce.HELD)].copy()
    pool = s[~s.study.isin(ce.HELD)].reset_index(drop=True)
    D = ce.l2_distance_matrix(pool[ce.SCOLS].to_numpy())
    print(f"q-sweep (ADD, base {ce.BASE_TECH}, {SEEDS} seeds). farthest = q=1.00")
    print(f"{'q':>5s} {'auc@0':>6s} {'speed@'+str(SPEED_K):>8s} {'end@'+str(K):>7s} "
          f"{'end_sd':>7s} {'end_min':>8s} {'frac<0.85':>10s}")
    for q in QS:
        e0, ek, eK = [], [], []
        for sd in range(SEEDS):
            r = curve(pool, D, held, q, sd)
            e0.append(r[0]); ek.append(r[SPEED_K]); eK.append(r[K])
        eK = np.array(eK)
        tag = "  <- farthest" if q == 1.0 else ("  <- current default" if q == 0.75 else "")
        print(f"{q:5.2f} {np.mean(e0):6.2f} {np.mean(ek):8.2f} {eK.mean():7.2f} "
              f"{eK.std():7.2f} {eK.min():8.2f} {(eK<0.85).mean():9.0%}{tag}")


if __name__ == "__main__":
    main()
