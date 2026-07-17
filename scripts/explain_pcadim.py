"""Should we compress the 1024-dim signatures before the classifier?
(A) how fast do PCA components of the signature space capture the BATCH axis vs the
    BIOLOGY axis? (if batch is captured by the top few PCs, hard truncation keeps batch
    and discards biology -> backfires on de-confounding.)
(B) run the tight-seed ADD enrichment with the CLASSIFIER input PCA-truncated to r dims
    (retrieval stays in full space), report held-out AUC@0 and AUC@K vs r.
StandardScaler->PCA(full)->L2-LR == the current pipeline (L2 is rotation-invariant),
so r=1024 reproduces the baseline; r<1024 = drop the low-variance tail.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from selection import greedy_quantile
import crc_enrichment as ce

TUMOR_STUDY, NORMAL_STUDY = "Borras_2023_Cell_Discov", "Scheid_2023_J_EXP_Med"
N, K, Q, SEEDS = 12, 12, 0.75, 15
RGRID = [5, 10, 20, 50, 100, 200, 1024]


def unit(v):
    return v / np.linalg.norm(v)


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    sc = StandardScaler().fit(poolr[C].to_numpy())
    Zp = sc.transform(poolr[C].to_numpy())
    Zh_raw = {st: sc.transform(g[C].to_numpy()) for st, g in held.groupby("study") if g.y.nunique() > 1}
    yh = {st: g.y.to_numpy() for st, g in held.groupby("study") if g.y.nunique() > 1}

    # axes in standardized signature space
    bio = []
    for st, g in poolr.groupby("study"):
        if g.sample_type.nunique() < 2:
            continue
        i1 = g[g.sample_type == "tumor"].index; i0 = g[g.sample_type == "normal"].index
        bio.append(Zp[poolr.index.get_indexer(i1)].mean(0) - Zp[poolr.index.get_indexer(i0)].mean(0))
    d_bio = unit(np.mean(bio, 0))
    bt = []
    for tp, g in poolr.groupby("sample_type"):
        a = poolr.index.get_indexer(g[g.assay == "10x 3' v3"].index)
        b = poolr.index.get_indexer(g[g.assay == "10x 5' v1"].index)
        if len(a) < 3 or len(b) < 3:
            continue
        bt.append(Zp[a].mean(0) - Zp[b].mean(0))
    d_batch = unit(np.mean(bt, 0))

    pca = PCA(n_components=min(len(Zp), len(C))).fit(Zp)
    comp = pca.components_                      # (ncomp, 1024)
    evr = pca.explained_variance_ratio_
    cb = np.cumsum((comp @ d_batch) ** 2)       # fraction of d_batch captured by top-k PCs
    co = np.cumsum((comp @ d_bio) ** 2)
    print("(A) how many PCs to capture each axis, and variance carried:")
    print(f"{'#PCs':>5s} {'cumVar':>7s} {'d_batch cap':>12s} {'d_bio cap':>10s}")
    for r in [5, 10, 20, 50, 100, 200]:
        print(f"{r:5d} {evr[:r].sum():7.2f} {cb[r-1]:12.2f} {co[r-1]:10.2f}")

    # (B) enrichment with truncated classifier input
    Dfull = ce.l2_distance_matrix(Zp)           # retrieval stays full-space
    scores_pool = pca.transform(Zp)             # (npool, ncomp)
    scores_held = {st: pca.transform(Zh_raw[st]) for st in Zh_raw}
    HT = poolr.index.get_indexer(poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index)
    HN = poolr.index.get_indexer(poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index)
    OT = poolr.index.get_indexer(poolr[(poolr.study != TUMOR_STUDY) & (poolr.y == 1)].index)
    ON = poolr.index.get_indexer(poolr[(poolr.study != NORMAL_STUDY) & (poolr.y == 0)].index)
    y = poolr.y.to_numpy()

    def auc_at(coh, r):
        clf = LogisticRegression(max_iter=2000).fit(scores_pool[coh, :r], y[coh])
        return np.mean([roc_auc_score(yh[st], clf.predict_proba(scores_held[st][:, :r])[:, 1]) for st in yh])

    print(f"\n(B) tight-seed ADD, classifier input PCA-truncated to r dims ({SEEDS} seeds):")
    print(f"{'r':>5s} {'AUC@0':>6s} {'AUC@K':>6s} {'delta':>6s}")
    for r in RGRID:
        a0, aK = [], []
        for sd in range(SEEDS):
            rng = np.random.default_rng(sd)
            sp = rng.choice(HT, N, replace=False); sn = rng.choice(HN, N, replace=False)
            ap = greedy_quantile(Dfull, list(sp), list(np.setdiff1d(OT, sp)), K, Q)
            an = greedy_quantile(Dfull, list(sn), list(np.setdiff1d(ON, sn)), K, Q)
            a0.append(auc_at(np.concatenate([sp, sn]), r))
            aK.append(auc_at(np.concatenate([sp, np.array(ap, int), sn, np.array(an, int)]), r))
        tag = "  <- current (=full)" if r == 1024 else ""
        print(f"{r:5d} {np.mean(a0):6.2f} {np.mean(aK):6.2f} {np.mean(aK)-np.mean(a0):+6.2f}{tag}")


if __name__ == "__main__":
    main()
