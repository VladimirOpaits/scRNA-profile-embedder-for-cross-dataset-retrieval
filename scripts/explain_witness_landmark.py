"""Phase 1 validation (power-matched): landmark-basis witness vs RFF-basis witness.
Two readouts:
 (A) FAIR comparison to RFF: evaluate the landmark witness FUNCTION
       f(x) = sum_j v_j * k(x, landmark_j)
     on the SAME 121k held cells, z-score, aggregate by cell type (same procedure the
     RFF witness used) -> removes the few-landmarks-per-type power gap.
 (B) DIRECT weights v_j aggregated by the landmark's own type (the landmark advantage,
     no projection) -- noisier due to L2 shrinkage, shown for reference.
Tight Borras/Scheid seed, confounded k=0 vs de-confounded k=K.
"""
import os
os.environ["CRC_SIG"] = "data/crc/signatures_landmark.parquet"   # before crc_enrichment import

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from selection import greedy_quantile
import crc_enrichment as ce

PCA = "data/crc/pca50.npy"
CELLS = "data/crc/cells.parquet"
CTYPE = "data/crc/cell_types.parquet"
LAND = "data/crc/landmarks.parquet"
CTCOL = "cell_type_coarse_crc_atlas"
TUMOR_STUDY, NORMAL_STUDY = "Borras_2023_Cell_Discov", "Scheid_2023_J_EXP_Med"
N, K, Q, SEEDS = 12, 12, 0.75, 15
CHUNK = 20000


def median_sigma(X, rng, m=5000):
    idx = rng.choice(len(X), min(m, len(X)), replace=False)
    S = X[idx]
    d = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    return float(np.median(d[d > 0]))


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    scb = StandardScaler().fit(poolr[C].to_numpy())
    Draw = ce.l2_distance_matrix(scb.transform(poolr[C].to_numpy()))
    HT = poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    HN = poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()
    OT = poolr[(poolr.study != TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    ON = poolr[(poolr.study != NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()

    def weff(cohort):
        sc = StandardScaler().fit(poolr.loc[cohort, C].to_numpy())
        clf = LogisticRegression(max_iter=2000).fit(
            sc.transform(poolr.loc[cohort, C].to_numpy()), poolr.loc[cohort, "y"].to_numpy())
        return clf.coef_[0] / sc.scale_
    W0, WK = [], []
    for sd in range(SEEDS):
        rng = np.random.default_rng(sd)
        sp = rng.choice(HT, N, replace=False); sn = rng.choice(HN, N, replace=False)
        ap = greedy_quantile(Draw, list(sp), list(np.setdiff1d(OT, sp)), K, Q)
        an = greedy_quantile(Draw, list(sn), list(np.setdiff1d(ON, sn)), K, Q)
        W0.append(weff(np.concatenate([sp, sn])))
        WK.append(weff(np.concatenate([sp, np.array(ap, int), sn, np.array(an, int)])))
    v0 = np.mean(W0, 0); vK = np.mean(WK, 0)

    # landmark coords + kernel sigma (same as build)
    X = np.load(PCA).astype(np.float32)
    land = pd.read_parquet(LAND)
    L = X[land.cell_row.to_numpy()]
    ltype = land[CTCOL].to_numpy()
    sigma = median_sigma(X, np.random.default_rng(0)); denom = 2 * sigma * sigma
    Ln = (L * L).sum(1)

    # (A) landmark witness FUNCTION on held cells
    cells = pd.read_parquet(CELLS); ct = pd.read_parquet(CTYPE)
    hm = cells.study.isin(ce.HELD).to_numpy()
    Xi = X[hm]; types = ct[CTCOL].to_numpy()[hm]

    def witness_on_cells(v):
        out = np.empty(len(Xi), np.float32)
        for a in range(0, len(Xi), CHUNK):
            b = min(a + CHUNK, len(Xi)); xc = Xi[a:b]
            d2 = (xc * xc).sum(1)[:, None] + Ln[None, :] - 2 * xc @ L.T
            out[a:b] = np.exp(-np.maximum(d2, 0) / denom) @ v
        return (out - out.mean()) / out.std()
    z0, zK = witness_on_cells(v0), witness_on_cells(vK)
    A = pd.DataFrame({"type": types, "k0": z0, "kK": zK}).groupby("type").agg(
        n=("k0", "size"), witness_k0=("k0", "mean"), witness_kK=("kK", "mean"))
    A["shift"] = A.witness_kK - A.witness_k0
    A = A.sort_values("witness_kK", ascending=False)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print("(A) LANDMARK witness FUNCTION on 121k held cells (fair, power-matched to RFF):")
    print(A.to_string())
    print("\n    RFF on record:  Cancer +0.13->+0.70(+0.56) Myeloid +0.32->+1.22(+0.90)")
    print("                    Stromal +1.02->-0.33(-1.35) Epith +0.29->-0.64(-0.93)")
    print("                    Schwann +1.27->-1.40(-2.67) B +0.02->-0.90(-0.92)")

    # (B) direct weights by landmark type
    df = pd.DataFrame({"type": ltype,
                       "z0": (v0 - v0.mean()) / v0.std(), "zK": (vK - vK.mean()) / vK.std()})
    B = df.groupby("type").agg(n=("z0", "size"), w_k0=("z0", "mean"), w_kK=("zK", "mean"))
    B["shift"] = B.w_kK - B.w_k0
    print("\n(B) DIRECT landmark weights v_j by type (landmark advantage, shrinkage-noisy):")
    print(B.sort_values("shift", ascending=False).to_string())


if __name__ == "__main__":
    main()
