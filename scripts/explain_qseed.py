"""Reconcile the q=0.90-vs-0.75 discrepancy: sweep q on BOTH seed types.
  'tech'  seed = base technology (10x 5' v1), both classes -> SPREAD seed (old crc_qsweep style)
  'tight' seed = Borras tumors / Scheid normals -> TIGHT single-study seed (fair cows-on-beach)
Same held-out, same ADD design, paired over seeds. Shows where the optimal q sits per seed type.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from selection import greedy_quantile
import crc_enrichment as ce

BASE_TECH = "10x 5' v1"
TUMOR_STUDY, NORMAL_STUDY = "Borras_2023_Cell_Discov", "Scheid_2023_J_EXP_Med"
N, K, SEEDS = 12, 12, 40
QS = [0.5, 0.65, 0.75, 0.9, 1.0]


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    sc = StandardScaler().fit(poolr[C].to_numpy())
    Zp = pd.DataFrame(sc.transform(poolr[C].to_numpy()), index=poolr.index)
    D = ce.l2_distance_matrix(Zp.to_numpy())
    Zh = {st: sc.transform(g[C].to_numpy()) for st, g in held.groupby("study") if g.y.nunique() > 1}
    yh = {st: g.y.to_numpy() for st, g in held.groupby("study") if g.y.nunique() > 1}

    def auc(coh):
        clf = LogisticRegression(max_iter=2000).fit(Zp.loc[coh].to_numpy(), poolr.loc[coh, "y"].to_numpy())
        return np.mean([roc_auc_score(yh[st], clf.predict_proba(Zh[st])[:, 1]) for st in Zh])

    # seed sources
    src = {
        "tech": dict(
            ht=poolr[(poolr.assay == BASE_TECH) & (poolr.y == 1)].index.to_numpy(),
            hn=poolr[(poolr.assay == BASE_TECH) & (poolr.y == 0)].index.to_numpy(),
            ot=poolr[(poolr.assay != BASE_TECH) & (poolr.y == 1)].index.to_numpy(),
            on=poolr[(poolr.assay != BASE_TECH) & (poolr.y == 0)].index.to_numpy()),
        "tight": dict(
            ht=poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy(),
            hn=poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy(),
            ot=poolr[(poolr.study != TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy(),
            on=poolr[(poolr.study != NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()),
    }

    def spread(idx):
        idx = list(idx[:min(len(idx), 40)])
        return np.median(D[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)])

    for mode, S in src.items():
        print(f"\n===== seed={mode}  (tumor-seed spread={spread(S['ht']):.0f}, "
              f"normal-seed spread={spread(S['hn']):.0f}) =====")
        # per q: arrays of AUC@3 and AUC@12 over seeds
        a3 = {q: [] for q in QS}; a12 = {q: [] for q in QS}; pdist = {q: [] for q in QS}
        for sd in range(SEEDS):
            rng = np.random.default_rng(sd)
            sp = rng.choice(S["ht"], N, replace=False); sn = rng.choice(S["hn"], N, replace=False)
            apool = np.setdiff1d(S["ot"], sp); anpool = np.setdiff1d(S["on"], sn)
            for q in QS:
                ap = greedy_quantile(D, list(sp), list(apool), K, q)
                an = greedy_quantile(D, list(sn), list(anpool), K, q)
                pdist[q].append(np.mean([D[np.ix_([p], list(sp))].min() for p in ap]))
                a3[q].append(auc(np.concatenate([sp, np.array(ap[:3], int), sn, np.array(an[:3], int)])))
                a12[q].append(auc(np.concatenate([sp, np.array(ap, int), sn, np.array(an, int)])))
        print(f"{'q':>5s} {'pickDist':>9s} {'AUC@3':>6s} {'AUC@12':>7s}")
        for q in QS:
            tag = "  <- max@12" if np.mean(a12[q]) == max(np.mean(a12[qq]) for qq in QS) else ""
            print(f"{q:5.2f} {np.mean(pdist[q]):9.0f} {np.mean(a3[q]):6.2f} {np.mean(a12[q]):7.2f}{tag}")
        # paired 0.90 vs 0.75
        d3 = np.array(a3[0.9]) - np.array(a3[0.75]); d12 = np.array(a12[0.9]) - np.array(a12[0.75])
        print(f"  paired 0.90-0.75 @3 : mean {d3.mean():+.3f} frac>0 {(d3>0).mean():.0%}")
        print(f"  paired 0.90-0.75 @12: mean {d12.mean():+.3f} frac>0 {(d12>0).mean():.0%}")


if __name__ == "__main__":
    main()
