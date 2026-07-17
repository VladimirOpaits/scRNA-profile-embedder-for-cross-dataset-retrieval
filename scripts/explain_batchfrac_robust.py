"""Robustness of the near->biology / far->batch signal: is it an artifact of hand-picking
40 extreme landmarks, or real?

The retrieval pairs (P,Q) do NOT depend on the landmark labeling, so compute them once, then
recompute batch_frac under three groupings:
  (1) REAL   : 40 lowest / 40 highest by marginal score (disc_held - inflation)  [as in fig]
  (2) NULL   : N_NULL random 40/40 splits of all 280 landmarks. If the q-ordering is an
               artifact of geometry (far pairs simply differ more), random splits reproduce it.
               If it is genuinely about batch vs biology, random splits are FLAT ~0.5.
  (3) INDEP  : seed-free labeling -- batch = 40 landmarks with the highest between-STUDY
               variance ratio (eta^2) on the pool; biology = 40 with the highest within-study
               tumor/normal disc on held. Uses neither the Borras/Scheid seed nor inflation,
               so a surviving signal here rules out circularity.
Statistic per grouping: contrast = median batch_frac(q=0.90) - median batch_frac(q=0.10)
(far minus near, method-A picks). Report REAL & INDEP vs the NULL distribution.
"""
import os
os.environ["CRC_SIG"] = "data/crc/signatures_landmark.parquet"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import crc_enrichment as ce

TUMOR_STUDY, NORMAL_STUDY = "Borras_2023_Cell_Discov", "Scheid_2023_J_EXP_Med"
N, SEEDS = 12, 15
QS = [0.10, 0.50, 0.75, 0.90]
MTAIL, N_NULL = 40, 500


def marginal_score(poolr, C, held, HT, HN):
    nL = len(C)
    aheld = np.zeros(nL); nst = 0
    for st, g in held.groupby("study"):
        y = g.y.to_numpy()
        if y.min() == y.max():
            continue
        aheld += np.array([roc_auc_score(y, g[C].to_numpy()[:, j]) for j in range(nL)]); nst += 1
    aheld /= nst
    E = poolr[C].to_numpy(); acoh = np.zeros(nL)
    for sd in range(SEEDS):
        rng = np.random.default_rng(sd)
        idx = np.concatenate([rng.choice(HT, N, replace=False), rng.choice(HN, N, replace=False)])
        y = poolr.loc[idx, "y"].to_numpy()
        acoh += np.array([roc_auc_score(y, E[poolr.index.get_indexer(idx)][:, j]) for j in range(nL)])
    acoh /= SEEDS
    disc_held = np.abs(aheld - 0.5)
    return disc_held - (np.abs(acoh - 0.5) - disc_held), disc_held


def study_eta2(poolr, C):
    """Between-study variance ratio per landmark (seed-free batch score)."""
    E = poolr[C].to_numpy(); gm = E.mean(0)
    sst = ((E - gm) ** 2).sum(0)
    ssb = np.zeros(len(C))
    for _, g in poolr.groupby("study"):
        Eg = g[C].to_numpy()
        ssb += len(Eg) * (Eg.mean(0) - gm) ** 2
    return ssb / sst


def frac(Z, Pi, Qi, bidx, gidx):
    d = (Z[Qi] - Z[Pi]) ** 2
    b = d[:, bidx].sum(1); g = d[:, gidx].sum(1)
    return b / (b + g)


def contrast(Z, pairs, bidx, gidx):
    m = {q: np.median(frac(Z, pairs[q][0], pairs[q][1], bidx, gidx)) for q in QS}
    return m, m[0.90] - m[0.10]


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    emb = poolr[C].to_numpy(); Z = (emb - emb.mean(0)) / emb.std(0)
    HT = poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    HN = poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()
    OT = poolr[(poolr.study != TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    ON = poolr[(poolr.study != NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()

    # pairs (P,Q), labeling-independent
    pairs = {q: ([], []) for q in QS}
    for seeds, cand in [(HT, OT), (HN, ON)]:
        for P in seeds:
            dP = np.linalg.norm(emb[cand] - emb[P], axis=1)
            for q in QS:
                pick = cand[int(np.argmin(np.abs(dP - np.quantile(dP, q))))]
                pairs[q][0].append(P); pairs[q][1].append(pick)
    pairs = {q: (np.array(a), np.array(b)) for q, (a, b) in pairs.items()}

    # (1) REAL
    score, disc_held = marginal_score(poolr, C, held, HT, HN)
    order = np.argsort(score)
    real_m, real_c = contrast(Z, pairs, order[:MTAIL], order[-MTAIL:])

    # (2) NULL: random 40/40 splits of all 280
    rng = np.random.default_rng(0)
    null_c = np.empty(N_NULL)
    for r in range(N_NULL):
        perm = rng.permutation(len(C))
        null_c[r] = contrast(Z, pairs, perm[:MTAIL], perm[MTAIL:2 * MTAIL])[1]

    # (3) INDEP: seed-free labeling
    eta = study_eta2(poolr, C)
    bidx_i = np.argsort(eta)[-MTAIL:]                 # most study-driven = batch
    gidx_i = np.argsort(disc_held)[-MTAIL:]           # most tumor/normal = biology
    gidx_i = np.setdiff1d(gidx_i, bidx_i)[:MTAIL]     # drop any overlap
    indep_m, indep_c = contrast(Z, pairs, bidx_i, gidx_i)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("median batch_frac by q (method-A picks):")
    print(f"{'grouping':10s} " + " ".join(f"q{q:.2f}" for q in QS) + "   far-near")
    print(f"{'REAL':10s} " + " ".join(f"{real_m[q]:5.3f}" for q in QS) + f"   {real_c:+.3f}")
    print(f"{'INDEP':10s} " + " ".join(f"{indep_m[q]:5.3f}" for q in QS) + f"   {indep_c:+.3f}")
    print(f"{'NULL mean':10s} " + " " * 24 + f"   {null_c.mean():+.3f}  (random 40/40)")
    print(f"\nfar-near contrast vs NULL (random-split) distribution:")
    print(f"  NULL: mean {null_c.mean():+.3f}  sd {null_c.std():.3f}  "
          f"95% CI [{np.quantile(null_c,.025):+.3f}, {np.quantile(null_c,.975):+.3f}]")
    for name, c in [("REAL", real_c), ("INDEP", indep_c)]:
        z = (c - null_c.mean()) / null_c.std()
        p = (np.abs(null_c - null_c.mean()) >= abs(c - null_c.mean())).mean()
        print(f"  {name}: {c:+.3f}  z={z:+.1f}  two-sided p={p:.3f} vs null")
    print(f"\n  overlap of REAL batch-set with INDEP batch-set: "
          f"{len(np.intersect1d(order[:MTAIL], bidx_i))}/{MTAIL} landmarks")


if __name__ == "__main__":
    main()
