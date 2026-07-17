"""Sigma-robustness of the leave-one-cell-type-out ablation.

Two fixes prompted by the extreme z-scores:
 (1) report EFFECT SIZES (raw Delta = metric(minusX) - metric(full)), not z. The random-null
     variance is near-degenerate (removing random cells barely moves a mean-over-cells KME, LLN),
     so z is inflated and is NOT an effect size -- it only flags "outside random jitter".
 (2) the RBF bandwidth is chosen by the median heuristic (sigma0=42.34), which is task-agnostic.
     Sweep sigma in {0.5,1,2}xmedian and check whether the qualitative pattern (which types carry
     batch, that Cancer/Epithelial are dilutive, that biology is distributed) is bandwidth-stable.
     Spearman rho of the per-type Delta rankings across sigmas = robustness.

d_bio   = held within-study tumor/normal LOO macro-AUC (batch fixed -> biology).
d_batch = mean between-study eta^2 on the pool (study-driven structure).
No null loop here (we established the jitter ~1e-3); Delta vs the FULL signature is the effect.
"""
import os
os.environ["CRC_SIG"] = "data/crc/signatures_landmark.parquet"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
import crc_enrichment as ce

PCA = "data/crc/pca50.npy"
CELLS = "data/crc/cells.parquet"
CTYPE = "data/crc/cell_types.parquet"
LAND = "data/crc/landmarks.parquet"
SIG = "data/crc/signatures_landmark.parquet"
CTCOL = "cell_type_coarse_crc_atlas"
CHUNK = 20000
FLOOR = 30
MULTS = [0.5, 1.0, 2.0]


def median_sigma(X, rng, m=5000):
    idx = rng.choice(len(X), min(m, len(X)), replace=False); S = X[idx]
    d = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    return float(np.median(d[d > 0]))


def d_bio(emb, meta):
    held = meta[meta.study.isin(ce.HELD)]; aucs = []
    for st, g in held.groupby("study"):
        y = g.y.to_numpy()
        if y.min() == y.max() or len(y) < 4:
            continue
        Xs = emb[g.idx.to_numpy()]; pr = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(Xs):
            sc = StandardScaler().fit(Xs[tr])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xs[tr]), y[tr])
            pr[te] = clf.predict_proba(sc.transform(Xs[te]))[:, 1]
        aucs.append(roc_auc_score(y, pr))
    return float(np.mean(aucs))


def d_batch(emb, meta):
    pool = meta[~meta.study.isin(ce.HELD)]
    E = emb[pool.idx.to_numpy()]; gm = E.mean(0); sst = ((E - gm) ** 2).sum(0) + 1e-12
    ssb = np.zeros(E.shape[1])
    for _, g in pool.groupby("study"):
        Eg = emb[g.idx.to_numpy()]; ssb += len(Eg) * (Eg.mean(0) - gm) ** 2
    return float(np.mean(ssb / sst))


def accumulate(X, L, Ln, denom, srow, trow, S, T, nL):
    sums = np.zeros((S, T, nL)); cnt = np.zeros((S, T))
    for a in range(0, len(X), CHUNK):
        b = min(a + CHUNK, len(X)); xc = X[a:b]
        d2 = (xc * xc).sum(1)[:, None] + Ln[None, :] - 2 * xc @ L.T
        K = np.exp(-np.maximum(d2, 0) / denom)
        np.add.at(sums, (srow[a:b], trow[a:b]), K)
        np.add.at(cnt, (srow[a:b], trow[a:b]), 1)
    return sums, cnt


def main():
    X = np.load(PCA).astype(np.float32)
    cells = pd.read_parquet(CELLS).reset_index(drop=True)
    ct = pd.read_parquet(CTYPE).reset_index(drop=True)
    land = pd.read_parquet(LAND); L = X[land.cell_row.to_numpy()]; Ln = (L * L).sum(1)
    s0 = median_sigma(X, np.random.default_rng(0)); nL = len(L)
    types = np.sort(ct[CTCOL].unique()); tidx = {t: i for i, t in enumerate(types)}; T = len(types)
    trow = ct[CTCOL].map(tidx).to_numpy()
    sid = cells.sample_id.to_numpy(); order = pd.unique(sid)
    srow_of = {s: i for i, s in enumerate(order)}
    srow = np.array([srow_of[s] for s in sid]); S = len(order)
    ref = pd.read_parquet(SIG)[["sample_id", "sample_type", "study"]]
    base = pd.DataFrame({"sample_id": order, "idx": np.arange(S)}).merge(ref, on="sample_id", how="left")
    base = base[base.sample_type.isin(["tumor", "normal"])].reset_index(drop=True)
    base["y"] = (base.sample_type == "tumor").astype(int)
    print(f"median sigma={s0:.2f}; sweeping x{MULTS}\n", flush=True)

    dbio_by_sigma, dbat_by_sigma = {}, {}
    for m in MULTS:
        sigma = m * s0; denom = 2 * sigma * sigma
        sums, cnt = accumulate(X, L, Ln, denom, srow, trow, S, T, nL)
        full_sum = sums.sum(1); full_cnt = cnt.sum(1)
        emb_full = full_sum / full_cnt[:, None]
        bio0, bat0 = d_bio(emb_full, base), d_batch(emb_full, base)
        rows = []
        for t in types:
            Xt = tidx[t]
            num = full_sum - sums[:, Xt, :]; den = full_cnt - cnt[:, Xt]
            emb = num / np.maximum(den, 1)[:, None]
            meta = base[den[base.idx.to_numpy()] >= FLOOR]
            rows.append((t, d_bio(emb, meta) - bio0, d_batch(emb, meta) - bat0))
        df = pd.DataFrame(rows, columns=["type", "dbio", "dbat"]).set_index("type")
        dbio_by_sigma[m], dbat_by_sigma[m] = df.dbio, df.dbat
        print(f"=== sigma x{m} ({sigma:.1f}) | baseline d_bio={bio0:.3f} d_batch={bat0:.4f} ===")
        print(df.sort_values("dbio").to_string(float_format=lambda x: f"{x:+.4f}"), "\n", flush=True)

    B = pd.DataFrame(dbio_by_sigma); K = pd.DataFrame(dbat_by_sigma)
    print("Delta d_bio by sigma:\n", B.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nDelta d_batch by sigma:\n", K.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nSpearman rho of per-type Delta ranking across sigma (robustness):")
    for name, D in [("d_bio", B), ("d_batch", K)]:
        for i, a in enumerate(MULTS):
            for bmm in MULTS[i + 1:]:
                r = spearmanr(D[a], D[bmm]).correlation
                print(f"  {name}: sigma x{a} vs x{bmm}  rho={r:+.2f}")


if __name__ == "__main__":
    main()
