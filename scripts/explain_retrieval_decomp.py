"""What does retrieval actually reach for? Decompose the SELECTION DISTANCE by landmark.

Landmark signatures make L2 distance separable per landmark:
    d(P,Q)^2 = sum_j (z_j(P) - z_j(Q))^2 ,  z = standardized emb_j = sim-to-landmark_j.
Each landmark j is a real cell state; label it batch- or biology-carrying with the SAME
marginal test as explain_witness_uni.py:
    disc_held_j  = |within-held-study tumor/normal AUC - 0.5|  (genuine, batch-controlled)
    inflation_j  = disc_cohort_j - disc_held_j                 (confound-driven overstatement)
    -> batch-carrying  if inflation_j >  disc_held_j  (apparent signal is mostly batch)
       biology-carrying otherwise.

Then, on the tight confounded seed (Borras tumor / Scheid normal), run q=0.75 retrieval and
attribute the squared distance each retrieved patient adds (to its nearest cohort member at
selection time) to batch- vs biology-landmarks. Contrast with the seed's OWN internal spread.
If retrieval de-confounds, the retrieved distance sits far more on batch-landmarks than the
seed's internal spread does -> "retrieval buys batch-diversity, leaves biology near-untouched".
"""
import os
os.environ["CRC_SIG"] = "data/crc/signatures_landmark.parquet"   # before crc_enrichment import

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import crc_enrichment as ce

LAND = "data/crc/landmarks.parquet"
CTCOL = "cell_type_coarse_crc_atlas"
TUMOR_STUDY, NORMAL_STUDY = "Borras_2023_Cell_Discov", "Scheid_2023_J_EXP_Med"
N, K, Q, SEEDS = 12, 12, 0.75, 15


def marginal_labels(poolr, C, held, HT, HN):
    """Per-landmark disc_held / inflation -> boolean batch-carrying mask (column order)."""
    nL = len(C)
    aheld = np.zeros(nL); nst = 0
    for st, g in held.groupby("study"):
        y = g.y.to_numpy()
        if y.min() == y.max():
            continue
        Eg = g[C].to_numpy()
        aheld += np.array([roc_auc_score(y, Eg[:, j]) for j in range(nL)])
        nst += 1
    aheld /= nst
    E = poolr[C].to_numpy()
    acoh = np.zeros(nL)
    for sd in range(SEEDS):
        rng = np.random.default_rng(sd)
        sp = rng.choice(HT, N, replace=False); sn = rng.choice(HN, N, replace=False)
        idx = np.concatenate([sp, sn]); y = poolr.loc[idx, "y"].to_numpy()
        Ec = E[poolr.index.get_indexer(idx)]
        acoh += np.array([roc_auc_score(y, Ec[:, j]) for j in range(nL)])
    acoh /= SEEDS
    disc_held = np.abs(aheld - 0.5)
    inflation = np.abs(acoh - 0.5) - disc_held
    batch_mask = inflation > disc_held
    return disc_held, inflation, batch_mask


def greedy_quantile_pairs(Draw, Z, cohort, pool, k, q):
    """q-quantile greedy retrieval; also return, per pick, the squared per-landmark diff to
    the nearest cohort member that DEFINED its selection distance."""
    cohort = list(cohort); pool = list(pool)
    deltas = []
    for _ in range(min(k, len(pool))):
        sub = Draw[np.ix_(pool, cohort)]
        d = sub.min(axis=1)
        target = np.quantile(d, q)
        best = int(np.argmin(np.abs(d - target)))
        chosen = pool[best]
        near = cohort[int(np.argmin(sub[best]))]
        deltas.append((Z[chosen] - Z[near]) ** 2)
        cohort.append(chosen); pool.pop(best)
    return np.array(deltas)                       # (k, nL)


def within_seed_deltas(Z, seed):
    """Each seed member -> nearest OTHER seed member: squared per-landmark diff."""
    S = Z[seed]
    g = (S * S).sum(1)
    D2 = g[:, None] + g[None, :] - 2 * S @ S.T
    np.fill_diagonal(D2, np.inf)
    near = D2.argmin(1)
    return (S - S[near]) ** 2                      # (len(seed), nL)


def frac(delta_rows, mask):
    """delta_rows: (m, nL). Return (frac on batch-landmarks, per-landmark-normalized batch:bio)."""
    tot = delta_rows.sum()
    b = delta_rows[:, mask].sum(); o = delta_rows[:, ~mask].sum()
    per_b = b / mask.sum(); per_o = o / (~mask).sum()
    return b / tot, per_b / per_o


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    HT = poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    HN = poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()
    OT = poolr[(poolr.study != TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    ON = poolr[(poolr.study != NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()
    land = pd.read_parquet(LAND); ltype = land[CTCOL].to_numpy()

    disc_held, inflation, batch_mask = marginal_labels(poolr, C, held, HT, HN)
    print(f"landmarks: {len(C)} total | batch-carrying {batch_mask.sum()} | "
          f"biology-carrying {(~batch_mask).sum()}")

    scb = StandardScaler().fit(poolr[C].to_numpy())
    Z = scb.transform(poolr[C].to_numpy())
    Draw = ce.l2_distance_matrix(Z)

    seed_rows, ret_rows = [], []
    for sd in range(SEEDS):
        rng = np.random.default_rng(sd)
        sp = rng.choice(HT, N, replace=False); sn = rng.choice(HN, N, replace=False)
        seed = np.concatenate([sp, sn])
        seed_rows.append(within_seed_deltas(Z, seed))
        ret_rows.append(greedy_quantile_pairs(Draw, Z, list(sp), list(np.setdiff1d(OT, sp)), K, Q))
        ret_rows.append(greedy_quantile_pairs(Draw, Z, list(sn), list(np.setdiff1d(ON, sn)), K, Q))
    seed_d = np.vstack(seed_rows); ret_d = np.vstack(ret_rows)

    fb_seed, ratio_seed = frac(seed_d, batch_mask)
    fb_ret, ratio_ret = frac(ret_d, batch_mask)
    print("\n=== squared-distance decomposition: batch- vs biology-landmarks ===")
    print(f"{'source':16s} {'mean d^2':>9s} {'%batch':>7s} {'perLM batch:bio':>16s}")
    print(f"{'within-seed':16s} {seed_d.sum(1).mean():9.2f} {fb_seed:7.1%} {ratio_seed:15.2f}x")
    print(f"{'retrieved(q.75)':16s} {ret_d.sum(1).mean():9.2f} {fb_ret:7.1%} {ratio_ret:15.2f}x")
    print(f"  -> retrieval shifts distance onto batch-landmarks by "
          f"{fb_ret - fb_seed:+.1%} of budget ({ratio_ret/ratio_seed:.2f}x per-landmark).")

    # named readout: which cell states accumulate retrieval distance
    by = pd.DataFrame({"type": ltype,
                       "disc_held": disc_held, "inflation": inflation,
                       "batch": batch_mask,
                       "ret": ret_d.sum(0), "seed": seed_d.sum(0)})
    agg = by.groupby("type").agg(n=("type", "size"),
                                 disc_held=("disc_held", "mean"),
                                 inflation=("inflation", "mean"),
                                 ret_share=("ret", "sum"), seed_share=("seed", "sum"))
    agg["ret_share"] /= agg.ret_share.sum(); agg["seed_share"] /= agg.seed_share.sum()
    agg["ret_vs_seed"] = agg.ret_share - agg.seed_share
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== retrieval distance by cell state (share of total squared distance) ===")
    print("  ret_vs_seed > 0 : retrieval loads MORE distance here than the seed's own spread")
    print(agg.sort_values("ret_vs_seed", ascending=False).to_string())


if __name__ == "__main__":
    main()
