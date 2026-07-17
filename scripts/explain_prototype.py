"""Prototype for the geometry-of-diversity figure.
Canvas = orthonormal plane (d_batch, d_bio). Watch the LR weight vector w
rotate from the batch axis toward the biology axis as diverse samples are added.
Static: k=0 vs k=K, one seed, quantile-0.90. Reports ||proj||/||w||.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from selection import greedy_quantile
import crc_enrichment as ce

SEED = 0
N = 15          # seed cohort per class
K = 15          # added per class
Q = 0.90


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    s = ce.load()                      # sets ce.SCOLS
    C = ce.SCOLS
    held, pool = ce.split(s)           # HELD5 vs rest

    # --- fixed coordinate frame: standardize on pool ---
    scaler = StandardScaler().fit(pool[C].to_numpy())
    Zall = pd.DataFrame(scaler.transform(s[C].to_numpy()), index=s.index)
    Zpool = scaler.transform(pool[C].to_numpy())

    def Z(df_idx):
        return Zall.loc[df_idx].to_numpy()

    # --- biology axis: tumor-normal within each study (batch fixed), averaged ---
    bio = []
    for st, g in s.groupby("study"):
        if g.sample_type.nunique() < 2:
            continue
        t = Z(g[g.sample_type == "tumor"].index).mean(0)
        n = Z(g[g.sample_type == "normal"].index).mean(0)
        bio.append(t - n)
    d_bio = unit(np.mean(bio, 0))

    # --- batch axis: base-tech vs rest within each sample_type (biology fixed) ---
    bat = []
    for tp, g in s.groupby("sample_type"):
        b = g[g.assay == ce.BASE_TECH].index
        o = g[g.assay != ce.BASE_TECH].index
        if len(b) < 3 or len(o) < 3:
            continue
        bat.append(Z(b).mean(0) - Z(o).mean(0))
    d_batch = unit(np.mean(bat, 0))

    # orthonormal canvas
    e_batch = d_batch
    e_bio = unit(d_bio - (d_bio @ e_batch) * e_batch)
    axis_angle = np.degrees(np.arccos(np.clip(d_bio @ d_batch, -1, 1)))
    print(f"angle between d_bio and d_batch: {axis_angle:.1f} deg "
          f"(90=orthogonal, ideal; small=axes collinear, premise weak)")

    def proj(v):
        return np.array([v @ e_batch, v @ e_bio])

    # --- build ADD cohort (quantile-0.90), one seed ---
    rng = np.random.default_rng(SEED)
    D = ce.l2_distance_matrix(Zpool)   # distances in fixed frame
    poolr = pool.reset_index(drop=True)
    base = poolr[poolr.assay == ce.BASE_TECH]
    other = poolr[poolr.assay != ce.BASE_TECH]
    sp = rng.choice(base[base.y == 1].index.to_numpy(), N, replace=False)
    sn = rng.choice(base[base.y == 0].index.to_numpy(), N, replace=False)
    rp = other[other.y == 1].index.to_numpy()
    rn = other[other.y == 0].index.to_numpy()
    ap = greedy_quantile(D, list(sp), list(rp), K, Q)
    an = greedy_quantile(D, list(sn), list(rn), K, Q)

    Zpool_df = pd.DataFrame(Zpool, index=poolr.index)

    def fit_w(k):
        coh = np.concatenate([sp, np.array(ap[:k], int), sn, np.array(an[:k], int)])
        X = Zpool_df.loc[coh].to_numpy()
        y = poolr.loc[coh, "y"].to_numpy()
        clf = LogisticRegression(max_iter=2000).fit(X, y)
        return unit(clf.coef_[0]), coh

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, k in zip(axes, [0, K]):
        w, coh = fit_w(k)
        pw = proj(w)
        frac = np.linalg.norm(pw)  # w is unit, so ||proj|| = in-plane fraction
        cb, cbio = w @ d_batch, w @ d_bio

        # background cloud (all samples, faint), colored by type
        for tp, col in [("tumor", "#d62728"), ("normal", "#1f77b4")]:
            m = s.sample_type == tp
            P = np.array([proj(v) for v in Zall[m.values].to_numpy()])
            ax.scatter(P[:, 0], P[:, 1], s=8, c=col, alpha=0.12, linewidths=0)

        # added points colored by assay
        added = np.array(list(ap[:k]) + list(an[:k]), int)
        if len(added):
            Pa = np.array([proj(v) for v in Zpool_df.loc[added].to_numpy()])
            asy = poolr.loc[added, "assay"].to_numpy()
            for a in np.unique(asy):
                mm = asy == a
                ax.scatter(Pa[mm, 0], Pa[mm, 1], s=70, marker="*",
                           edgecolors="k", linewidths=0.5, label=a)

        # w arrow (direction only), scaled to canvas
        sc = 2.5
        ax.annotate("", xy=(pw[0] / frac * sc, pw[1] / frac * sc), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color="k"))
        ax.axhline(0, ls=":", c="gray", lw=0.7)
        ax.axvline(0, ls=":", c="gray", lw=0.7)
        ax.set_xlabel("batch axis  (d_batch)")
        if k == 0:
            ax.set_ylabel("biology axis  (d_bio)")
        ax.set_title(f"k={k}   in-plane ||proj||/||w||={frac:.2f}\n"
                     f"cos(w,batch)={cb:+.2f}  cos(w,bio)={cbio:+.2f}", fontsize=10)
        ax.legend(fontsize=6, loc="upper left", framealpha=0.9)
        print(f"k={k:2d}: cos(w,batch)={cb:+.3f}  cos(w,bio)={cbio:+.3f}  "
              f"in-plane frac={frac:.3f}")

    fig.suptitle("Does w rotate off the batch axis toward biology? (quantile-0.90, seed 0)")
    plt.tight_layout()
    out = "explain_prototype.png"
    plt.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
