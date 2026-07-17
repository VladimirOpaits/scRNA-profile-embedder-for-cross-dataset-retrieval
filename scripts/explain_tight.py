"""Fair cows-on-beach: TIGHT single-study seed clusters.
Previous seed drew tumors from a whole TECHNOLOGY (many studies) -> already spread
across space, so near/far retrieval had no meaningful reference. Here the seed is
tumors from ONE study + normals from ONE (different) study = two compact blobs, a
study-level confound. Now 'near' picks (studies like the seed) vs 'far' picks
(distant studies) have a real reference. Re-test whether distance matters, paired.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from selection import greedy_quantile
import crc_enrichment as ce

TUMOR_STUDY = "Borras_2023_Cell_Discov"     # tight tumor cluster (spread 13.5)
NORMAL_STUDY = "Scheid_2023_J_EXP_Med"      # normal-only study
N, K, SEEDS = 12, 12, 40
QN, QF = 0.2, 0.9


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    sc = StandardScaler().fit(poolr[C].to_numpy())
    Zp = pd.DataFrame(sc.transform(poolr[C].to_numpy()), index=poolr.index)
    D = ce.l2_distance_matrix(Zp.to_numpy())
    bio = []
    for st, g in poolr.groupby("study"):
        if g.sample_type.nunique() < 2:
            continue
        bio.append(Zp.loc[g[g.sample_type == "tumor"].index].mean(0).to_numpy()
                   - Zp.loc[g[g.sample_type == "normal"].index].mean(0).to_numpy())
    dbio = np.mean(bio, 0); dbio /= np.linalg.norm(dbio)
    Zh = {st: sc.transform(g[C].to_numpy()) for st, g in held.groupby("study") if g.y.nunique() > 1}
    yh = {st: g.y.to_numpy() for st, g in held.groupby("study") if g.y.nunique() > 1}

    st_t = poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    sn_n = poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()
    ot = poolr[(poolr.study != TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    on = poolr[(poolr.study != NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()

    def spread(idx):
        idx = list(idx)
        return np.median(D[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)])
    print(f"tumor seed = {TUMOR_STUDY} (n={len(st_t)}, spread {spread(st_t):.1f})")
    print(f"normal seed = {NORMAL_STUDY} (n={len(sn_n)}, spread {spread(sn_n):.1f})")
    print(f"diverse-add pool: tumor {len(ot)}, normal {len(on)}\n")

    def evalc(coh):
        clf = LogisticRegression(max_iter=2000).fit(Zp.loc[coh].to_numpy(), poolr.loc[coh, "y"].to_numpy())
        au = np.mean([roc_auc_score(yh[st], clf.predict_proba(Zh[st])[:, 1]) for st in Zh])
        w = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
        return au, w @ dbio

    def run(q_or_rand, rng, sp, sn):
        if q_or_rand == "random":
            ap = list(rng.choice(ot, K, replace=False)); an = list(rng.choice(on, K, replace=False))
        else:
            ap = greedy_quantile(D, list(sp), list(ot), K, q_or_rand)
            an = greedy_quantile(D, list(sn), list(on), K, q_or_rand)
        pd_ = np.mean([D[np.ix_([p], list(sp))].min() for p in ap])
        out = {}
        for k in [0, 3, 6, 12]:
            out[k] = evalc(np.concatenate([sp, np.array(ap[:k], int), sn, np.array(an[:k], int)]))
        return out, pd_

    arms = {"near": QN, "far": QF, "random": "random"}
    agg = {a: {k: [] for k in [0, 3, 6, 12]} for a in arms}
    aggc = {a: [] for a in arms}; pdist = {a: [] for a in arms}
    paired = {k: [] for k in [3, 6, 12]}; pairedc = []
    for sd in range(SEEDS):
        rng = np.random.default_rng(sd)
        sp = rng.choice(st_t, N, replace=False); sn = rng.choice(sn_n, N, replace=False)
        res = {}
        for a, q in arms.items():
            o, pdd = run(q, np.random.default_rng(1000 + sd), sp, sn)
            res[a] = o; pdist[a].append(pdd)
            for k in [0, 3, 6, 12]:
                agg[a][k].append(o[k][0])
            aggc[a].append(o[12][1])
        for k in [3, 6, 12]:
            paired[k].append(res["far"][k][0] - res["near"][k][0])
        pairedc.append(res["far"][12][1] - res["near"][12][1])

    print(f"{'arm':>7s} {'pickDist':>9s} {'AUC@0':>6s} {'AUC@3':>6s} {'AUC@6':>6s} {'AUC@12':>7s} {'cosbio@12':>10s}")
    for a in arms:
        print(f"{a:>7s} {np.mean(pdist[a]):9.1f} {np.mean(agg[a][0]):6.2f} {np.mean(agg[a][3]):6.2f} "
              f"{np.mean(agg[a][6]):6.2f} {np.mean(agg[a][12]):7.2f} {np.mean(aggc[a]):10.2f}")
    print(f"\npaired far-near ({SEEDS} seeds, tight seed):")
    for k in [3, 6, 12]:
        d = np.array(paired[k])
        print(f"  AUC@{k:<2d}: mean {d.mean():+.3f}  sd {d.std():.3f}  frac(far>near) {(d>0).mean():.0%}  "
              f"[p10 {np.percentile(d,10):+.2f}, p90 {np.percentile(d,90):+.2f}]")
    c = np.array(pairedc)
    print(f"  cosbio@12: mean {c.mean():+.3f}  sd {c.std():.3f}  frac>0 {(c>0).mean():.0%}")


if __name__ == "__main__":
    main()
