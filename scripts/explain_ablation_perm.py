"""Rigorous leave-one-cell-type-out ablation: effect size (Delta) + non-parametric
permutation p (no Gaussian-z, no distributional assumption).

For type X:  Delta_real = metric(drop type X) - metric(full signature).
Null = R count-matched random removals (shuffle type labels WITHIN each sample -> same per-sample
counts, same FLOOR mask, random membership). Delta_null^r = metric(drop fake-type X) - metric(full).
permutation p = (1 + #{r : |Delta_null^r| >= |Delta_real|}) / (R+1)   [two-sided, add-one].

Memory-safe: kernel K (N x L) precomputed ONCE (float32); each null draw only re-accumulates
per-(sample,type) sums against the stored K (no kernel recompute, no (R,S,T,L) tensor).

d_bio   = held within-study tumor/normal LOO macro-AUC (batch fixed -> biology).
d_batch = mean between-study eta^2 on the pool.
"""
import os
os.environ["CRC_SIG"] = "data/crc/signatures_landmark.parquet"

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import crc_enrichment as ce

PCA = "data/crc/pca50.npy"; CELLS = "data/crc/cells.parquet"; CTYPE = "data/crc/cell_types.parquet"
LAND = "data/crc/landmarks.parquet"; SIG = "data/crc/signatures_landmark.parquet"
CTCOL = "cell_type_coarse_crc_atlas"
CHUNK = 20000; R = 100; FLOOR = 30; NFOLD = 5; OUT = "explain_ablation_perm.png"


def median_sigma(X, rng, m=5000):
    idx = rng.choice(len(X), min(m, len(X)), replace=False); S = X[idx]
    d = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)); return float(np.median(d[d > 0]))


def d_bio(emb, meta):
    """held within-study tumor/normal, pooled StratifiedKFold-CV macro-AUC (5 folds instead of
    LOO -> ~10x fewer fits; same construct: out-of-fold biological separability)."""
    held = meta[meta.study.isin(ce.HELD)]; aucs = []
    for st, g in held.groupby("study"):
        y = g.y.to_numpy()
        if y.min() == y.max() or min((y == 0).sum(), (y == 1).sum()) < NFOLD:
            continue
        Xs = emb[g.idx.to_numpy()]; pr = np.zeros(len(y))
        for tr, te in StratifiedKFold(NFOLD, shuffle=True, random_state=0).split(Xs, y):
            sc = StandardScaler().fit(Xs[tr])
            clf = LogisticRegression(max_iter=500).fit(sc.transform(Xs[tr]), y[tr])
            pr[te] = clf.predict_proba(sc.transform(Xs[te]))[:, 1]
        aucs.append(roc_auc_score(y, pr))
    return float(np.mean(aucs))


def d_batch(emb, meta):
    pool = meta[~meta.study.isin(ce.HELD)]; E = emb[pool.idx.to_numpy()]
    gm = E.mean(0); sst = ((E - gm) ** 2).sum(0) + 1e-12; ssb = np.zeros(E.shape[1])
    for _, g in pool.groupby("study"):
        Eg = emb[g.idx.to_numpy()]; ssb += len(Eg) * (Eg.mean(0) - gm) ** 2
    return float(np.mean(ssb / sst))


def main():
    X = np.load(PCA).astype(np.float32)
    cells = pd.read_parquet(CELLS).reset_index(drop=True)
    ct = pd.read_parquet(CTYPE).reset_index(drop=True)
    land = pd.read_parquet(LAND); L = X[land.cell_row.to_numpy()]; Ln = (L * L).sum(1)
    sigma = median_sigma(X, np.random.default_rng(0)); denom = 2 * sigma * sigma; nL = len(L)
    types = np.sort(ct[CTCOL].unique()); tidx = {t: i for i, t in enumerate(types)}; T = len(types)
    trow = ct[CTCOL].map(tidx).to_numpy().astype(np.int64)
    sid = cells.sample_id.to_numpy(); order = pd.unique(sid)
    srow = np.array([{s: i for i, s in enumerate(order)}[s] for s in sid]); S = len(order)
    N = len(X)
    print(f"cells={N} samples={S} types={T} landmarks={nL} sigma={sigma:.2f} R={R}", flush=True)

    # precompute kernel K (N x L) once
    Kall = np.empty((N, nL), np.float32)
    for a in range(0, N, CHUNK):
        b = min(a + CHUNK, N); xc = X[a:b]
        d2 = (xc * xc).sum(1)[:, None] + Ln[None, :] - 2 * xc @ L.T
        Kall[a:b] = np.exp(-np.maximum(d2, 0) / denom)
    print("kernel precomputed", flush=True)

    cols = np.arange(N)
    ones = np.ones(N, np.float32)

    def sums_by_type(labels):
        # (S*T x N) one-hot @ (N x L) = per (sample,type) kernel sum, C-speed sparse matmul
        A = csr_matrix((ones, (srow * T + labels, cols)), shape=(S * T, N))
        return np.asarray(A @ Kall).reshape(S, T, nL)
    cnt = np.bincount(srow * T + trow, minlength=S * T).reshape(S, T).astype(float)
    sumT = sums_by_type(trow)
    full_sum = sumT.sum(1); full_cnt = cnt.sum(1)

    ref = pd.read_parquet(SIG)[["sample_id", "sample_type", "study"]]
    base = pd.DataFrame({"sample_id": order, "idx": np.arange(S)}).merge(ref, on="sample_id", how="left")
    base = base[base.sample_type.isin(["tumor", "normal"])].reset_index(drop=True)
    base["y"] = (base.sample_type == "tumor").astype(int)
    bidx = base.idx.to_numpy()

    def metrics_leaveout(sumT_src, Xt):
        num = full_sum - sumT_src[:, Xt, :]; den = full_cnt - cnt[:, Xt]
        emb = num / np.maximum(den, 1)[:, None]; meta = base[den[bidx] >= FLOOR]
        return d_bio(emb, meta), d_batch(emb, meta)

    bio0, bat0 = d_bio(full_sum / full_cnt[:, None], base), d_batch(full_sum / full_cnt[:, None], base)
    print(f"BASELINE d_bio={bio0:.3f} d_batch={bat0:.4f}\n", flush=True)

    real = {t: metrics_leaveout(sumT, tidx[t]) for t in types}

    # per-sample cell positions for within-sample shuffling
    posper = [np.where(srow == p)[0] for p in range(S)]
    rng = np.random.default_rng(0)
    null_bio = {t: [] for t in types}; null_bat = {t: [] for t in types}
    for r in range(R):
        lab = trow.copy()
        for p in posper:
            lab[p] = rng.permutation(trow[p])
        sr = sums_by_type(lab)
        for t in types:
            b, k = metrics_leaveout(sr, tidx[t]); null_bio[t].append(b); null_bat[t].append(k)
        if (r + 1) % 20 == 0:
            print(f"  null {r+1}/{R}", flush=True)

    rows = []
    for t in types:
        db, dk = real[t][0] - bio0, real[t][1] - bat0
        nb = np.array(null_bio[t]) - bio0; nk = np.array(null_bat[t]) - bat0
        p_bio = (1 + int((np.abs(nb) >= abs(db)).sum())) / (R + 1)
        p_bat = (1 + int((np.abs(nk) >= abs(dk)).sum())) / (R + 1)
        rows.append(dict(type=t, removed=int(cnt[:, tidx[t]].sum()),
                         d_bio=db, p_bio=p_bio, d_batch=dk, p_batch=p_bat,
                         null_bio_sd=float(nb.std()), null_bat_sd=float(nk.std())))
    df = pd.DataFrame(rows).sort_values("d_batch")
    df.to_csv("explain_ablation_perm.csv", index=False)
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}")
    print(f"\n=== effect size (Delta vs full) + permutation p (R={R}) ===")
    print(df[["type", "removed", "d_bio", "p_bio", "d_batch", "p_batch",
              "null_bio_sd", "null_bat_sd"]].to_string(index=False))
    print("\n(null_*_sd = spread of random-removal jitter; tiny -> why z blew up)")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.axhline(0, color="0.7", lw=1); ax.axvline(0, color="0.7", lw=1)
    for _, r in df.iterrows():
        sig = (r.p_bio < 0.05) or (r.p_batch < 0.05)
        ax.scatter(r.d_batch, r.d_bio, s=70 if sig else 35,
                   color="#c0392b" if sig else "#95a5a6", zorder=3)
        ax.annotate(f"{r.type}\n(p_b={r.p_bio:.3f},p_k={r.p_batch:.3f})",
                    (r.d_batch, r.d_bio), fontsize=7, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Delta d_batch  (<0 = removing type cuts the batch axis)")
    ax.set_ylabel("Delta d_bio  (>0 = removing type IMPROVES biology = dilutive)")
    ax.set_title(f"Causal leave-one-cell-type-out, effect size + permutation p (R={R})\n"
                 "red = significant on either axis")
    fig.tight_layout(); fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT} and explain_ablation_perm.csv")


if __name__ == "__main__":
    main()
