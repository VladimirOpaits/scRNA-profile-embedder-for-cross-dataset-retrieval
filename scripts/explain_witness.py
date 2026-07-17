"""Biological layer: the MMD witness function per cell type.
Patient score = w . mean_cells phi(cell); so each cell contributes w_eff . phi(cell),
w_eff = w_std / scaler.scale_ (fold standardization into raw RFF space). We evaluate
the witness on held-out cells and aggregate by atlas cell type, for the CONFOUNDED
classifier (k=0, tight Borras/Scheid seed) vs the DE-CONFOUNDED one (k=K far adds).
Question: does the classifier shift its weight onto Cancer/Epithelial (real tumor
biology) as diversity de-rotates it? RFF params reproduced exactly from crc_signatures.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from selection import greedy_quantile
import crc_enrichment as ce

PCA = "data/crc/pca50.npy"
CELLS = "data/crc/cells.parquet"
CTYPE = "data/crc/cell_types.parquet"
TUMOR_STUDY, NORMAL_STUDY = "Borras_2023_Cell_Discov", "Scheid_2023_J_EXP_Med"
CTCOL = "cell_type_coarse_crc_atlas"
N, K, Q, SEEDS, D, RSEED = 12, 12, 0.75, 10, 1024, 0
CHUNK = 20000


def rff_params(X):
    rng = np.random.default_rng(RSEED)
    idx = rng.choice(len(X), min(5000, len(X)), replace=False)
    S = X[idx]
    sigma = float(np.median((lambda d: d[d > 0])(
        np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)))))
    W = (rng.standard_normal((X.shape[1], D)) / sigma).astype(np.float32)
    b = (rng.uniform(0, 2 * np.pi, D)).astype(np.float32)
    return W, b


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    Draw = ce.l2_distance_matrix(
        StandardScaler().fit_transform(poolr[C].to_numpy()))
    HT = poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    HN = poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()
    OT = poolr[(poolr.study != TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    ON = poolr[(poolr.study != NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()

    # average w_eff (raw RFF space) over seeds, at k=0 and k=K
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
    w0 = np.mean(W0, 0); wK = np.mean(WK, 0)

    # held-out cells: pca + coarse type
    X = np.load(PCA).astype(np.float32)
    cells = pd.read_parquet(CELLS); ct = pd.read_parquet(CTYPE)
    heldmask = cells.study.isin(ce.HELD).to_numpy()
    Xi = X[heldmask]; types = ct[CTCOL].to_numpy()[heldmask]
    print(f"held cells: {len(Xi)} | RFF witness per cell (chunked)")
    Wm, bm = rff_params(X)

    def witness(wv):
        out = np.empty(len(Xi), np.float32)
        for a in range(0, len(Xi), CHUNK):
            b_ = min(a + CHUNK, len(Xi))
            phi = np.sqrt(2.0 / D) * np.cos(Xi[a:b_] @ Wm + bm)
            out[a:b_] = phi @ wv
        return (out - out.mean()) / out.std()      # z-score across held cells (compare shape, not scale)

    z0, zK = witness(w0), witness(wK)
    df = pd.DataFrame({"type": types, "k0": z0, "kK": zK})
    g = df.groupby("type").agg(n=("k0", "size"), witness_k0=("k0", "mean"),
                               witness_kK=("kK", "mean"))
    g["shift"] = g.witness_kK - g.witness_k0
    g = g.sort_values("witness_kK", ascending=False)
    pd.set_option("display.float_format", lambda v: f"{v:+.3f}")
    print("\nmean z-witness per coarse cell type (+ = pushes toward TUMOR):")
    print(g.to_string())
    print("\n(positive witness_kK = de-confounded classifier keys on this type for tumor call)")


if __name__ == "__main__":
    main()
