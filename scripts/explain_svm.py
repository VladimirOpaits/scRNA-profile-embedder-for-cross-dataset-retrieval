"""Linear SVM (hinge, like CKME) vs L2 logistic (log-loss, ours) — paired.
Same KME/RFF features, same L2, same interpretability; only the loss differs. Does the
loss choice change held-out AUC on our tight-seed enrichment? (AUC is rank-based, so
SVM decision_function is fine.) rho=1.0 tight Borras/Scheid seed, q=0.75.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from selection import greedy_quantile
import crc_enrichment as ce

TUMOR_STUDY, NORMAL_STUDY = "Borras_2023_Cell_Discov", "Scheid_2023_J_EXP_Med"
N, K, Q, SEEDS = 12, 12, 0.75, 30


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    scb = StandardScaler().fit(poolr[C].to_numpy())
    Draw = ce.l2_distance_matrix(scb.transform(poolr[C].to_numpy()))
    Zh = {st: g[C].to_numpy() for st, g in held.groupby("study") if g.y.nunique() > 1}
    yh = {st: g.y.to_numpy() for st, g in held.groupby("study") if g.y.nunique() > 1}
    HT = poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    HN = poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()
    OT = poolr[(poolr.study != TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    ON = poolr[(poolr.study != NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()

    def macro(clf, sc, cohort):
        clf.fit(sc.transform(poolr.loc[cohort, C].to_numpy()), poolr.loc[cohort, "y"].to_numpy())
        sco = (lambda Xt: clf.predict_proba(Xt)[:, 1] if hasattr(clf, "predict_proba")
               else clf.decision_function(Xt))
        return np.mean([roc_auc_score(yh[st], sco(sc.transform(Zh[st]))) for st in yh])

    def run(cohort):
        sc = StandardScaler().fit(poolr.loc[cohort, C].to_numpy())
        lr = macro(LogisticRegression(max_iter=2000), sc, cohort)
        sv = macro(LinearSVC(C=1.0, max_iter=5000), sc, cohort)
        return lr, sv

    res = {0: [], K: []}
    for sd in range(SEEDS):
        rng = np.random.default_rng(sd)
        sp = rng.choice(HT, N, replace=False); sn = rng.choice(HN, N, replace=False)
        ap = greedy_quantile(Draw, list(sp), list(np.setdiff1d(OT, sp)), K, Q)
        an = greedy_quantile(Draw, list(sn), list(np.setdiff1d(ON, sn)), K, Q)
        res[0].append(run(np.concatenate([sp, sn])))
        res[K].append(run(np.concatenate([sp, np.array(ap, int), sn, np.array(an, int)])))

    print(f"tight seed, q={Q}, {SEEDS} seeds. macro held-out AUC:")
    print(f"{'k':>3s} {'logistic':>9s} {'linSVM':>7s} {'SVM-LR':>7s} {'frac(SVM>LR)':>13s}")
    for k in (0, K):
        a = np.array(res[k]); d = a[:, 1] - a[:, 0]
        print(f"{k:3d} {a[:,0].mean():9.3f} {a[:,1].mean():7.3f} {d.mean():+7.3f} {(d>0).mean():12.0%}")


if __name__ == "__main__":
    main()
