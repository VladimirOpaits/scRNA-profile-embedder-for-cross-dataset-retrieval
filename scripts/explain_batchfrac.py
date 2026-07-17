"""Visual A/B: does distance-based retrieval reach for pairs whose difference is more (or
less) batch-loaded than a random same-diagnosis pair?

Per pair (P,Q):  batch_frac = sum_{j in batch} z_j^2 / (sum_{j in batch} + sum_{j in bio}) z_j^2
z_j = (emb_j(P)-emb_j(Q)) / std_j  (columns z-scored on pool: batch landmarks carry ~all the
raw variance -> raw batch_frac saturates at ~0.95 and cannot separate A/B; whitening + a
BALANCED landmark set makes 0.5 = genuine batch/biology parity).
Landmark labels are SEED-FREE (no circularity; verified vs a random-split null in
explain_batchfrac_robust.py -- the inflation-based labeling was ~null there, this one is z=+4.9):
  batch = M landmarks with the highest between-STUDY variance ratio (eta^2) on the pool,
  bio   = M landmarks with the highest within-study tumor/normal disc on held studies.
SELECTION distance stays raw-emb L2 (the actual MMD retrieval uses); only the readout metric
is whitened.

Method A (RFF-MMD): for seed P and same-diagnosis candidate pool, pick candidate at quantile
                    q of the distance-from-P. One Q per (P,q).
Method B (random) : RANDOM_DRAWS random candidates per P (many draws so the baseline's own
                    spread is measured, not a single noisy pick). q-independent grey reference.
Seeds = tight confounded home studies (Borras tumor / Scheid normal); pools = other-study
same-y patients. Figure: 4 panels (one per q), overlaid density histograms A(blue) vs B(grey),
red line at 0.5 (batch/biology parity of the difference vector).
"""
import os
os.environ["CRC_SIG"] = "data/crc/signatures_landmark.parquet"   # before crc_enrichment import

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import crc_enrichment as ce

LAND = "data/crc/landmarks.parquet"
CTCOL = "cell_type_coarse_crc_atlas"
TUMOR_STUDY, NORMAL_STUDY = "Borras_2023_Cell_Discov", "Scheid_2023_J_EXP_Med"
N, SEEDS = 12, 15
QS = [0.10, 0.50, 0.75, 0.90]
RANDOM_DRAWS = 60
MTAIL = 40                    # balanced set: M most-biology vs M most-batch landmarks
OUT = "explain_batchfrac.png"


def seedfree_labels(poolr, C, held, M):
    """batch = top-M between-study eta^2 (pool); bio = top-M within-study tumor/normal disc
    (held). No Borras/Scheid seed, no inflation -> non-circular."""
    nL = len(C)
    aheld = np.zeros(nL); nst = 0
    for st, g in held.groupby("study"):
        y = g.y.to_numpy()
        if y.min() == y.max():
            continue
        aheld += np.array([roc_auc_score(y, g[C].to_numpy()[:, j]) for j in range(nL)]); nst += 1
    disc_held = np.abs(aheld / nst - 0.5)
    E = poolr[C].to_numpy(); gm = E.mean(0); sst = ((E - gm) ** 2).sum(0)
    ssb = np.zeros(nL)
    for _, g in poolr.groupby("study"):
        Eg = g[C].to_numpy(); ssb += len(Eg) * (Eg.mean(0) - gm) ** 2
    eta = ssb / sst
    bidx = np.argsort(eta)[-M:]
    gidx = np.setdiff1d(np.argsort(disc_held)[-M:], bidx)[:M]
    return bidx, gidx


def batch_frac(Z, P, Q, bidx, gidx):
    d = (Z[Q] - Z[P]) ** 2                          # (m, nL) if Q array; z-scored space
    b = d[..., bidx].sum(-1); g = d[..., gidx].sum(-1)
    return b / (b + g)


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    emb = poolr[C].to_numpy()
    HT = poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    HN = poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()
    OT = poolr[(poolr.study != TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    ON = poolr[(poolr.study != NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()
    bidx, gidx = seedfree_labels(poolr, C, held, MTAIL)
    Z = (emb - emb.mean(0)) / emb.std(0)            # whiten columns (readout only)
    print(f"seeds: {len(HT)} tumor(Borras) + {len(HN)} normal(Scheid) | "
          f"pools: {len(OT)} tumor + {len(ON)} normal cand | "
          f"balanced set: {MTAIL} batch vs {MTAIL} biology landmarks")

    rng = np.random.default_rng(0)
    A = {q: [] for q in QS}; B = []
    for seeds, cand in [(HT, OT), (HN, ON)]:
        for P in seeds:
            dP = np.linalg.norm(emb[cand] - emb[P], axis=1)   # raw-emb MMD selection
            for q in QS:                            # method A: quantile pick
                pick = cand[int(np.argmin(np.abs(dP - np.quantile(dP, q))))]
                A[q].append(float(batch_frac(Z, P, pick, bidx, gidx)))
            draws = rng.choice(cand, min(RANDOM_DRAWS, len(cand)), replace=False)  # method B
            B.extend(batch_frac(Z, P, draws, bidx, gidx).tolist())
    B = np.array(B)

    print(f"\n{'q':>5s} {'medianA':>8s} {'medianB':>8s} {'A<0.5':>6s} {'B<0.5':>6s}")
    for q in QS:
        a = np.array(A[q])
        print(f"{q:5.2f} {np.median(a):8.3f} {np.median(B):8.3f} "
              f"{(a<0.5).mean():6.1%} {(B<0.5).mean():6.1%}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, q in zip(axes, QS):
        ax.hist(B, bins=bins, density=True, color="0.6", alpha=0.55, label="random")
        ax.hist(A[q], bins=bins, density=True, histtype="stepfilled",
                color="#2b6cb0", alpha=0.55, edgecolor="#1a4a80", lw=1.3,
                label=f"RFF-MMD q={q:.2f}")
        ax.axvline(0.5, color="crimson", lw=1.6)
        ax.set_title(f"q = {q:.2f}"); ax.set_xlabel("batch_frac of the P->Q difference")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("density")
    fig.suptitle("Batch-loading of the retrieved difference vector: distance-pick vs random "
                 "(same-diagnosis pool)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
