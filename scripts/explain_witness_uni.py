"""Marginal (per-landmark) witness -- the scLKME-style fix for the joint-L2 collapse.
Instead of one regularized classifier over all 280 landmarks (weights share a penalty
budget -> collective shrinkage -> collapse), test EACH landmark INDEPENDENTLY. Each test
uses the full sample on its own, so p>>n never arises (Yi&Stanley use Wilcoxon per
landmark; n as low as 20).

De-confounding version, per landmark j (embedding_j = similarity-to-landmark_j):
  a_cohort_j = tumor-vs-normal AUC of embedding_j in the CONFOUNDED seed (Borras tumor /
               Scheid normal) -> mixes biology + batch (Borras vs Scheid differ).
  a_held_j   = mean within-study tumor-vs-normal AUC of embedding_j across held studies
               (batch held constant -> genuine biological discrimination).
  inflation_j = a_cohort_j - a_held_j -> how much the confound INFLATES this landmark's
               apparent discrimination via batch. High inflation = batch-carrying state.
Aggregate by landmark cell type. Robust (each landmark independent, full n, no joint reg).
"""
import os
os.environ["CRC_SIG"] = "data/crc/signatures_landmark.parquet"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import crc_enrichment as ce

LAND = "data/crc/landmarks.parquet"
CTCOL = "cell_type_coarse_crc_atlas"
TUMOR_STUDY, NORMAL_STUDY = "Borras_2023_Cell_Discov", "Scheid_2023_J_EXP_Med"
N, SEEDS = 12, 15


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    land = pd.read_parquet(LAND); ltype = land[CTCOL].to_numpy()
    E = poolr[C].to_numpy()                          # pool landmark embeddings (n_pool x 280)
    HT = poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    HN = poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()

    nL = len(C)
    # a_held_j: within-study tumor/normal AUC per landmark, averaged over held studies
    aheld = np.zeros(nL)
    nst = 0
    for st, g in held.groupby("study"):
        y = g.y.to_numpy()
        if y.min() == y.max():
            continue
        Eg = g[C].to_numpy()
        aheld += np.array([roc_auc_score(y, Eg[:, j]) for j in range(nL)])
        nst += 1
    aheld /= nst

    # a_cohort_j: confounded seed tumor/normal AUC per landmark, averaged over seeds
    acoh = np.zeros(nL)
    for sd in range(SEEDS):
        rng = np.random.default_rng(sd)
        sp = rng.choice(HT, N, replace=False); sn = rng.choice(HN, N, replace=False)
        idx = np.concatenate([sp, sn]); y = poolr.loc[idx, "y"].to_numpy()
        Ec = E[poolr.index.get_indexer(idx)]
        acoh += np.array([roc_auc_score(y, Ec[:, j]) for j in range(nL)])
    acoh /= SEEDS

    df = pd.DataFrame({"type": ltype, "a_cohort": acoh, "a_held": aheld})
    # fold to discrimination strength around 0.5 (direction-agnostic), then inflation
    df["disc_cohort"] = (df.a_cohort - 0.5).abs()
    df["disc_held"] = (df.a_held - 0.5).abs()
    df["inflation"] = df.disc_cohort - df.disc_held
    g = df.groupby("type").agg(n=("type", "size"),
                               disc_held=("disc_held", "mean"),
                               disc_cohort=("disc_cohort", "mean"),
                               inflation=("inflation", "mean"))
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("per-landmark MARGINAL witness by cell type (robust, no joint regularization):")
    print("  disc_held   = genuine tumor/normal signal (batch-controlled, within held study)")
    print("  disc_cohort = apparent signal in the CONFOUNDED seed (biology + batch)")
    print("  inflation   = confound-driven over-statement (high = batch-carrying type)\n")
    print(g.sort_values("disc_held", ascending=False).to_string())
    print("\nsorted by INFLATION (which types the confound spuriously amplifies):")
    print(g.sort_values("inflation", ascending=False)[["n", "inflation", "disc_held"]].to_string())


if __name__ == "__main__":
    main()
