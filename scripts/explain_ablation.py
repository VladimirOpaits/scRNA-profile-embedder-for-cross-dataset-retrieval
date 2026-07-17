"""Causal leave-one-CELL-TYPE-out ablation (Variant B).

Recompute each patient's landmark signature EXCLUDING all cells of one type, and measure
whether the biology axis (d_bio) and/or the batch axis (d_batch) collapses. This turns the
correlational witness ("type X's landmarks co-vary with biology") into a causal necessity test
("remove type X's cells -> does the biological discrimination die?").

emb_j^{-X}(P) = sum_{i in P, type_i != X} K(x_i, l_j) / #{i in P: type_i != X}

Computed in ONE pass: accumulate partial sums over (sample, type, landmark); leave-one-type-out
is then just subtraction (full_sum - sum[:,X]) / (full_cnt - cnt[:,X]). COUNT-MATCHED RANDOM
NULL: shuffle the type labels WITHIN each sample (fake types keep per-sample sizes but random
membership) -> leaving out fake-type-X removes the same number of random cells. R shuffles give
a noise band; the CAUSAL effect of type X = its d_bio/d_batch drop BEYOND that band (else the
drop is merely "fewer cells", not "these cells").

d_bio   = held within-study tumor/normal LOO macro-AUC (batch fixed -> pure biology).
d_batch = mean between-study variance ratio (eta^2) on the pool (how study-driven the signature).
Report per type: real vs null, z-score. Expect a DOUBLE DISSOCIATION -- biology-carriers
(Myeloid/Cancer/Neutrophil) drop d_bio; batch-carriers (Plasma/ILC/NK) drop d_batch.
"""
import os
os.environ["CRC_SIG"] = "data/crc/signatures_landmark.parquet"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
R = 10                       # random-null shuffles
FLOOR = 30                   # min non-X cells for a sample to stay in a given ablation
OUT = "explain_ablation.png"


def median_sigma(X, rng, m=5000):
    idx = rng.choice(len(X), min(m, len(X)), replace=False); S = X[idx]
    d = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    return float(np.median(d[d > 0]))


def d_bio(emb, meta):
    """held within-study tumor/normal LOO macro-AUC on given signatures."""
    held = meta[meta.study.isin(ce.HELD)]
    aucs = []
    for st, g in held.groupby("study"):
        y = g.y.to_numpy()
        if y.min() == y.max() or len(y) < 4:
            continue
        Xs = emb[g.idx.to_numpy()]
        pr = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(Xs):
            sc = StandardScaler().fit(Xs[tr])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xs[tr]), y[tr])
            pr[te] = clf.predict_proba(sc.transform(Xs[te]))[:, 1]
        aucs.append(roc_auc_score(y, pr))
    return float(np.mean(aucs))


def d_batch(emb, meta):
    """mean between-study eta^2 over landmarks on the pool."""
    pool = meta[~meta.study.isin(ce.HELD)]
    E = emb[pool.idx.to_numpy()]; gm = E.mean(0); sst = ((E - gm) ** 2).sum(0) + 1e-12
    ssb = np.zeros(E.shape[1])
    for _, g in pool.groupby("study"):
        Eg = emb[g.idx.to_numpy()]; ssb += len(Eg) * (Eg.mean(0) - gm) ** 2
    return float(np.mean(ssb / sst))


def main():
    X = np.load(PCA).astype(np.float32)
    cells = pd.read_parquet(CELLS).reset_index(drop=True)
    ct = pd.read_parquet(CTYPE).reset_index(drop=True)
    land = pd.read_parquet(LAND); L = X[land.cell_row.to_numpy()]; Ln = (L * L).sum(1)
    sigma = median_sigma(X, np.random.default_rng(0)); denom = 2 * sigma * sigma
    nL = len(L)

    types = np.sort(ct[CTCOL].unique()); tidx = {t: i for i, t in enumerate(types)}
    T = len(types)
    trow = ct[CTCOL].map(tidx).to_numpy()
    sid = cells.sample_id.to_numpy(); order = pd.unique(sid)
    srow_of = {s: i for i, s in enumerate(order)}
    srow = np.array([srow_of[s] for s in sid]); S = len(order)
    print(f"cells={len(X)} samples={S} types={T} landmarks={nL} sigma={sigma:.2f}", flush=True)

    # build R within-sample type shuffles (count-matched random nulls)
    rng = np.random.default_rng(0)
    trow_r = np.tile(trow, (R, 1))
    for _, pos in pd.Series(np.arange(len(sid))).groupby(srow):
        p = pos.to_numpy()
        for r in range(R):
            trow_r[r, p] = rng.permutation(trow[p])

    # ONE pass: partial sums (R+1, S, T, L); cnt (S, T) identical across shuffles
    sums = np.zeros((R + 1, S, T, nL))
    cnt = np.zeros((S, T))
    for a in range(0, len(X), CHUNK):
        b = min(a + CHUNK, len(X)); xc = X[a:b]
        d2 = (xc * xc).sum(1)[:, None] + Ln[None, :] - 2 * xc @ L.T
        K = np.exp(-np.maximum(d2, 0) / denom)
        sr = srow[a:b]
        np.add.at(sums[0], (sr, trow[a:b]), K)
        np.add.at(cnt, (sr, trow[a:b]), 1)
        for r in range(R):
            np.add.at(sums[r + 1], (sr, trow_r[r, a:b]), K)
    full_sum = sums.sum(2)                    # (R+1, S, L)
    full_cnt = cnt.sum(1)                      # (S,)

    ref = pd.read_parquet(SIG)[["sample_id", "sample_type", "study"]]
    base = pd.DataFrame({"sample_id": order, "idx": np.arange(S)}).merge(ref, on="sample_id", how="left")
    base = base[base.sample_type.isin(["tumor", "normal"])].reset_index(drop=True)
    base["y"] = (base.sample_type == "tumor").astype(int)   # idx = row into emb (order position)

    def emb_leaveout(r, X_type):
        num = full_sum[r] - sums[r, :, X_type, :]
        den = full_cnt - cnt[:, X_type]
        return num, den

    # baseline (full signature)
    emb_full = full_sum[0] / full_cnt[:, None]
    bio0, bat0 = d_bio(emb_full, base), d_batch(emb_full, base)
    print(f"\nBASELINE (full):  d_bio={bio0:.3f}  d_batch={bat0:.4f}\n", flush=True)

    rows = []
    for t in types:
        Xt = tidx[t]
        # real ablation
        num, den = emb_leaveout(0, Xt)
        emb = num / np.maximum(den, 1)[:, None]
        meta = base[den[base.idx.to_numpy()] >= FLOOR]
        bio_real, bat_real = d_bio(emb, meta), d_batch(emb, meta)
        # null band
        bion, batn = [], []
        for r in range(1, R + 1):
            numr, denr = emb_leaveout(r, Xt)
            embr = numr / np.maximum(denr, 1)[:, None]
            mr = base[denr[base.idx.to_numpy()] >= FLOOR]
            bion.append(d_bio(embr, mr)); batn.append(d_batch(embr, mr))
        bion, batn = np.array(bion), np.array(batn)
        zb = (bio_real - bion.mean()) / (bion.std() + 1e-9)
        zk = (bat_real - batn.mean()) / (batn.std() + 1e-9)
        removed = int(cnt[:, Xt].sum())
        rows.append(dict(type=t, removed=removed,
                         d_bio=bio_real, bio_null=bion.mean(), z_bio=zb,
                         d_batch=bat_real, batch_null=batn.mean(), z_batch=zk))
        print(f"drop {t:15s} cells={removed:7d} | d_bio {bio_real:.3f} (null {bion.mean():.3f}"
              f" z={zb:+.1f}) | d_batch {bat_real:.4f} (null {batn.mean():.4f} z={zk:+.1f})",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("explain_ablation.csv", index=False)
    print("\nBIOLOGY-CARRIERS (most negative z_bio = removing them causally kills d_bio):")
    print(df.sort_values("z_bio")[["type", "removed", "z_bio", "z_batch"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.axhline(0, color="0.7", lw=1); ax.axvline(0, color="0.7", lw=1)
    ax.scatter(df.z_batch, df.z_bio, s=40, color="#2b6cb0")
    for _, r in df.iterrows():
        ax.annotate(r.type, (r.z_batch, r.z_bio), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("z(d_batch)  <- removing type reduces batch axis")
    ax.set_ylabel("z(d_bio)  <- removing type reduces biology axis")
    ax.set_title("Causal leave-one-cell-type-out (Variant B) vs count-matched random null\n"
                 "lower-left = specifically kills that axis beyond chance")
    fig.tight_layout(); fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT} and explain_ablation.csv")


if __name__ == "__main__":
    main()
