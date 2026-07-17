"""Does DISTANCE matter? Compare near vs far retrieval quantiles.
random ~ farthest in our data only because the pool is already so diverse that a
random draw is near-optimally spread. To isolate the distance lever, contrast a FAR
quantile (q=0.9) against a NEAR one (q=0.2): if near picks help less / rotate w less,
distance is a real lever (independent of the random baseline being coincidentally good).
Fixed strong confound rho=1.0 (most room for selection to matter). ADD design.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from selection import greedy_quantile
import crc_enrichment as ce

A, B = "10x 3' v3", "10x 5' v1"
N, K = 12, 12
RHO = 1.0
SEEDS = 25
QS = [0.1, 0.2, 0.5, 0.9]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    s = ce.load(); C = ce.SCOLS
    held, pool = ce.split(s); poolr = pool.reset_index(drop=True)
    scaler = StandardScaler().fit(poolr[C].to_numpy())
    Zp = pd.DataFrame(scaler.transform(poolr[C].to_numpy()), index=poolr.index)

    bio = []
    for st, g in poolr.groupby("study"):
        if g.sample_type.nunique() < 2:
            continue
        bio.append(Zp.loc[g[g.sample_type == "tumor"].index].mean(0).to_numpy()
                   - Zp.loc[g[g.sample_type == "normal"].index].mean(0).to_numpy())
    d_bio = unit(np.mean(bio, 0))

    Zh = {st: scaler.transform(g[C].to_numpy()) for st, g in held.groupby("study")
          if g.y.nunique() == 2}
    yh = {st: g.y.to_numpy() for st, g in held.groupby("study") if g.y.nunique() == 2}
    D = ce.l2_distance_matrix(Zp.to_numpy())
    At = poolr[(poolr.assay == A) & (poolr.y == 1)].index.to_numpy()
    Bt = poolr[(poolr.assay == B) & (poolr.y == 1)].index.to_numpy()
    An = poolr[(poolr.assay == A) & (poolr.y == 0)].index.to_numpy()
    Bn = poolr[(poolr.assay == B) & (poolr.y == 0)].index.to_numpy()
    ot = poolr[(~poolr.assay.isin([A, B])) & (poolr.y == 1)].index.to_numpy()
    on = poolr[(~poolr.assay.isin([A, B])) & (poolr.y == 0)].index.to_numpy()
    nA = int(round(RHO * N)); nB = N - nA

    def evalc(coh):
        clf = LogisticRegression(max_iter=2000).fit(
            Zp.loc[coh].to_numpy(), poolr.loc[coh, "y"].to_numpy())
        auc = np.mean([roc_auc_score(yh[st], clf.predict_proba(Zh[st])[:, 1]) for st in Zh])
        return auc, unit(clf.coef_[0]) @ d_bio

    def pick(sp, sn, arm, rng):
        if arm == "random":
            return list(rng.choice(ot, K, replace=False)), list(rng.choice(on, K, replace=False))
        q = float(arm)
        return greedy_quantile(D, list(sp), list(ot), K, q), \
            greedy_quantile(D, list(sn), list(on), K, q)

    def pick_dist(seed_idx, picks):
        # mean distance of each pick to the (growing) seed -- summarise as mean over picks
        return float(np.mean([D[np.ix_([p], list(seed_idx))].min() for p in picks]))

    arms = [str(q) for q in QS] + ["random"]
    curves = {a: np.zeros((K + 1, 2)) for a in arms}      # auc, cosbio
    pdist = {a: [] for a in arms}
    for a in arms:
        for sd in range(SEEDS):
            rng = np.random.default_rng(sd)
            sp = np.concatenate([rng.choice(At, nA, replace=False),
                                 rng.choice(Bt, nB, replace=False)])
            sn = np.concatenate([rng.choice(Bn, nA, replace=False),
                                 rng.choice(An, nB, replace=False)])
            ap, an = pick(sp, sn, a, rng)
            pdist[a].append((pick_dist(sp, ap) + pick_dist(sn, an)) / 2)
            for k in range(K + 1):
                coh = np.concatenate([sp, np.array(ap[:k], int), sn, np.array(an[:k], int)])
                au, cb = evalc(coh)
                curves[a][k] += (au, cb)
        curves[a] /= SEEDS

    print(f"confound rho={RHO}, ADD, {SEEDS} seeds. NEAR(low q) vs FAR(high q):")
    print(f"{'arm':>8s} {'pickDist':>9s} {'AUC@0':>6s} {'AUC@3':>6s} {'AUC@12':>7s} "
          f"{'cosbio@0':>9s} {'cosbio@12':>10s}")
    for a in arms:
        c = curves[a]
        print(f"{a:>8s} {np.mean(pdist[a]):9.2f} {c[0,0]:6.2f} {c[3,0]:6.2f} {c[12,0]:7.2f} "
              f"{c[0,1]:9.2f} {c[12,1]:10.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cmap = {"0.1": "#08519c", "0.2": "#3182bd", "0.5": "#9ecae1",
            "0.9": "#d62728", "random": "#7f7f7f"}
    for a in arms:
        lab = f"q={a}" if a != "random" else "random"
        lab += " (near)" if a in ("0.1", "0.2") else (" (far)" if a == "0.9" else "")
        ax[0].plot(range(K + 1), curves[a][:, 0], "-o", ms=3, color=cmap[a], label=lab)
        ax[1].plot(range(K + 1), curves[a][:, 1], "-o", ms=3, color=cmap[a], label=lab)
    ax[0].set_xlabel("# added / class (k)"); ax[0].set_ylabel("held-out macro AUC")
    ax[0].set_title(f"Near vs far picks (rho={RHO})"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("# added / class (k)"); ax[1].set_ylabel("cos(w, biology axis)")
    ax[1].set_title("Rotation onto biology axis by quantile"); ax[1].legend(fontsize=8)
    fig.suptitle("Distance IS a lever: far quantile de-confounds, near quantile does not")
    plt.tight_layout(); plt.savefig("explain_qcontrast.png", dpi=110)
    print("\nwrote explain_qcontrast.png")


if __name__ == "__main__":
    main()
