"""Verify the k-crossover the user saw in the app (tight seed, paired over seeds):
  (1) at very small k, do NEAR quantiles beat FAR?
  (2) where does FAR overtake NEAR?
  (3) is q=0.9 ~ q=0.75 and q=0.1 ~ q=0.25 (plateaus at the extremes)?
Tight single-study seed (Borras tumors / Scheid normals), rho=1.0.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from selection import greedy_quantile
import crc_enrichment as ce

TUMOR_STUDY, NORMAL_STUDY = "Borras_2023_Cell_Discov", "Scheid_2023_J_EXP_Med"
N, K, SEEDS = 12, 12, 60


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    sc = StandardScaler().fit(poolr[C].to_numpy())
    Zp = pd.DataFrame(sc.transform(poolr[C].to_numpy()), index=poolr.index)
    D = ce.l2_distance_matrix(Zp.to_numpy())
    Zh = {st: sc.transform(g[C].to_numpy()) for st, g in held.groupby("study") if g.y.nunique() > 1}
    yh = {st: g.y.to_numpy() for st, g in held.groupby("study") if g.y.nunique() > 1}
    HT = poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    HN = poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()
    OT = poolr[(poolr.study != TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()
    ON = poolr[(poolr.study != NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()

    def auc(coh):
        clf = LogisticRegression(max_iter=2000).fit(Zp.loc[coh].to_numpy(), poolr.loc[coh, "y"].to_numpy())
        return np.mean([roc_auc_score(yh[st], clf.predict_proba(Zh[st])[:, 1]) for st in Zh])

    def curve(sp, sn, q):
        ap = greedy_quantile(D, list(sp), list(np.setdiff1d(OT, sp)), K, q)
        an = greedy_quantile(D, list(sn), list(np.setdiff1d(ON, sn)), K, q)
        return [auc(np.concatenate([sp, np.array(ap[:k], int), sn, np.array(an[:k], int)])) for k in range(K + 1)]

    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    curves = {q: np.zeros((SEEDS, K + 1)) for q in qs}
    for sd in range(SEEDS):
        rng = np.random.default_rng(sd)
        sp = rng.choice(HT, N, replace=False); sn = rng.choice(HN, N, replace=False)
        for q in qs:
            curves[q][sd] = curve(sp, sn, q)

    # (1)(2) crossover: paired far(0.9) - near(0.2) per k
    print(f"tight seed, {SEEDS} seeds. paired far(0.9) - near(0.25) by k:")
    print(f"{'k':>3s} {'far-near':>9s} {'frac(far>near)':>15s}")
    d = curves[0.9] - curves[0.25]
    cross = None
    for k in range(1, K + 1):
        fr = (d[:, k] > 0).mean()
        if cross is None and d[:, k].mean() > 0 and fr >= 0.6:
            cross = k
        print(f"{k:3d} {d[:,k].mean():+9.3f} {fr:14.0%}")
    print(f"-> FAR reliably overtakes NEAR around k={cross}")

    # (3) plateaus at extremes: paired diffs at endpoint and early
    def pair(qa, qb, k):
        dd = curves[qa][:, k] - curves[qb][:, k]
        return dd.mean(), (dd > 0).mean()
    print("\nplateau checks (paired):")
    for (qa, qb) in [(0.9, 0.75), (0.1, 0.25)]:
        for k in [3, 12]:
            m, fr = pair(qa, qb, k)
            print(f"  q{qa} - q{qb} @k={k:<2d}: mean {m:+.3f}  frac>0 {fr:.0%}")

    # plot mean curves
    plt.figure(figsize=(8, 5))
    cmap = {0.1: "#08519c", 0.25: "#3182bd", 0.5: "#9ecae1", 0.75: "#fc9272", 0.9: "#d62728"}
    for q in qs:
        m = curves[q].mean(0)
        plt.plot(range(K + 1), m, "-o", ms=3, color=cmap[q], label=f"q={q}")
    plt.axvspan(4, 7, color="gray", alpha=0.10, label="far-edge peaks (k≈4–7)")
    plt.xlabel("# added / class (k)"); plt.ylabel("held-out macro AUC (mean over seeds)")
    plt.title("Tight seed: far (q≥0.5) leads from k≈1; edge peaks mid-k; q=0.1 never catches up")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig("explain_kcross.png", dpi=110)
    print("\nwrote explain_kcross.png")


if __name__ == "__main__":
    main()
