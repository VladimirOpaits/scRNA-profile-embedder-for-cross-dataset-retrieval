"""Sweep the confound level rho and ask: is REPLACE>ADD only at extreme confound?
rho = fraction of tumors drawn from tech A (and normals from tech B). The rest of
each class comes from the OTHER tech. rho=0.5 -> no confound (both classes 50/50
across A,B). rho=1.0 -> perfect confound (label<->batch). For each rho, build the
seed, then ADD vs REPLACE diverse same-label samples, measure held-out AUC and the
biology-axis alignment at k=0 and k=K.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from selection import greedy_quantile
import crc_enrichment as ce

A = "10x 3' v3"
B = "10x 5' v1"
N = 12
K = 12
Q = 0.90
SEEDS = 25
RHOS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    s = ce.load()
    C = ce.SCOLS
    held, pool = ce.split(s)
    poolr = pool.reset_index(drop=True)
    scaler = StandardScaler().fit(poolr[C].to_numpy())
    Zp = pd.DataFrame(scaler.transform(poolr[C].to_numpy()), index=poolr.index)

    # biology axis (for rotation readout)
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

    def evalc(coh):
        clf = LogisticRegression(max_iter=2000).fit(
            Zp.loc[coh].to_numpy(), poolr.loc[coh, "y"].to_numpy())
        auc = np.mean([roc_auc_score(yh[st], clf.predict_proba(Zh[st])[:, 1]) for st in Zh])
        cb = unit(clf.coef_[0]) @ d_bio
        return auc, cb

    print(f"seed N={N}/class, K={K}, {SEEDS} seeds, quantile-{Q}. "
          f"rho = frac(tumor from {A}) = frac(normal from {B})")
    print(f"{'rho':>4s} {'base@0':>7s} {'add@K':>6s} {'rep@K':>6s} {'rep-add':>8s}  "
          f"{'cosbio@0':>9s} {'cosbio addK':>11s} {'cosbio repK':>11s}")
    table = []
    for rho in RHOS:
        nA = int(round(rho * N)); nB = N - nA
        acc = {k: [] for k in ["b0", "aK", "rK", "cb0", "cbA", "cbR"]}
        for sd in range(SEEDS):
            rng = np.random.default_rng(sd)
            sp = np.concatenate([rng.choice(At, nA, replace=False),
                                 rng.choice(Bt, nB, replace=False)])
            sn = np.concatenate([rng.choice(Bn, nA, replace=False),
                                 rng.choice(An, nB, replace=False)])
            rng.shuffle(sp); rng.shuffle(sn)   # unbiased REPLACE removal order
            ap = greedy_quantile(D, list(sp), list(ot), K, Q)
            an = greedy_quantile(D, list(sn), list(on), K, Q)
            a0, cb0 = evalc(np.concatenate([sp, sn]))
            aK, cbA = evalc(np.concatenate([sp, np.array(ap, int), sn, np.array(an, int)]))
            rK, cbR = evalc(np.concatenate([sp[K:], np.array(ap, int),
                                            sn[K:], np.array(an, int)]))
            acc["b0"].append(a0); acc["aK"].append(aK); acc["rK"].append(rK)
            acc["cb0"].append(cb0); acc["cbA"].append(cbA); acc["cbR"].append(cbR)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        table.append((rho, m))
        print(f"{rho:4.1f} {m['b0']:7.2f} {m['aK']:6.2f} {m['rK']:6.2f} "
              f"{m['rK']-m['aK']:+8.2f}  {m['cb0']:9.2f} {m['cbA']:11.2f} {m['cbR']:11.2f}")

    rhos = [t[0] for t in table]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(rhos, [t[1]["b0"] for t in table], "-o", c="gray", label="baseline (k=0)")
    ax[0].plot(rhos, [t[1]["aK"] for t in table], "-s", c="#1f77b4", label="ADD @K")
    ax[0].plot(rhos, [t[1]["rK"] for t in table], "-^", c="#d62728", label="REPLACE @K")
    ax[0].set_xlabel("confound level rho"); ax[0].set_ylabel("held-out macro AUC")
    ax[0].set_title("Recovery vs confound severity"); ax[0].legend()
    ax[1].plot(rhos, [t[1]["rK"] - t[1]["aK"] for t in table], "-o", c="k")
    ax[1].axhline(0, ls=":", c="gray")
    ax[1].set_xlabel("confound level rho"); ax[1].set_ylabel("REPLACE - ADD  (AUC @K)")
    ax[1].set_title("REPLACE's edge grows with confound")
    fig.suptitle("Does REPLACE beat ADD only under strong confound?")
    plt.tight_layout(); plt.savefig("explain_confound_sweep.png", dpi=110)
    print("\nwrote explain_confound_sweep.png")


if __name__ == "__main__":
    main()
