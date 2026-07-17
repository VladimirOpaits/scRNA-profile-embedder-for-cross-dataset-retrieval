"""Cows-on-beach in its purest form.
Seed cohort is perfectly confounded: all tumors from tech A, all normals from tech B.
=> in training, batch predicts label perfectly, so w sits on the BATCH axis and
held-out AUC collapses (<0.5). Then ADD diverse same-label samples from OTHER techs,
breaking the label<->batch coupling, and watch w rotate onto the BIOLOGY axis while
held-out AUC recovers. One fixed-frame model gives BOTH geometry and performance.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from selection import greedy_quantile
import crc_enrichment as ce

A = "10x 3' v3"      # tumors drawn from here
B = "10x 5' v1"      # normals drawn from here
N = 12               # seed per class
K = int(os.environ.get("K", 12))         # added per class
Q = 0.90
SEEDS = int(os.environ.get("SEEDS", 20))
FRAME_SEED = 0
DO_FRAMES = os.environ.get("FRAMES", "1") == "1"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    s = ce.load()
    C = ce.SCOLS
    held, pool = ce.split(s)
    poolr = pool.reset_index(drop=True)

    scaler = StandardScaler().fit(poolr[C].to_numpy())
    Zpool = pd.DataFrame(scaler.transform(poolr[C].to_numpy()), index=poolr.index)
    Zall = pd.DataFrame(scaler.transform(s[C].to_numpy()), index=s.index)

    # --- axes (pool only) ---
    bio = []
    for st, g in poolr.groupby("study"):
        if g.sample_type.nunique() < 2:
            continue
        bio.append(Zpool.loc[g[g.sample_type == "tumor"].index].mean(0).to_numpy()
                   - Zpool.loc[g[g.sample_type == "normal"].index].mean(0).to_numpy())
    d_bio = unit(np.mean(bio, 0))
    bat = []
    for tp, g in poolr.groupby("sample_type"):
        a = g[g.assay == A].index
        b = g[g.assay == B].index
        if len(a) < 3 or len(b) < 3:
            continue
        bat.append(Zpool.loc[a].mean(0).to_numpy() - Zpool.loc[b].mean(0).to_numpy())
    d_batch = unit(np.mean(bat, 0))
    e_batch = d_batch
    e_bio = unit(d_bio - (d_bio @ e_batch) * e_batch)
    ang = np.degrees(np.arccos(np.clip(d_bio @ d_batch, -1, 1)))
    print(f"seed confound: tumors={A} / normals={B}")
    print(f"angle(d_bio,d_batch)={ang:.1f} deg\n")

    # candidate index pools
    at = poolr[(poolr.assay == A) & (poolr.y == 1)].index.to_numpy()   # tumor-A seed
    bn = poolr[(poolr.assay == B) & (poolr.y == 0)].index.to_numpy()   # normal-B seed
    ot = poolr[(~poolr.assay.isin([A, B])) & (poolr.y == 1)].index.to_numpy()  # diverse tumor
    on = poolr[(~poolr.assay.isin([A, B])) & (poolr.y == 0)].index.to_numpy()  # diverse normal
    print(f"seed pools: tumor-A={len(at)} normal-B={len(bn)} | "
          f"diverse-add tumor={len(ot)} normal={len(on)}")

    D = ce.l2_distance_matrix(Zpool.to_numpy())
    # held rows via scaler directly (held was reset_index'd, so it does NOT share
    # labels with Zall/s.index -- do not .loc into Zall here)
    Zheld = {st: scaler.transform(g[C].to_numpy()) for st, g in held.groupby("study")
             if g.y.nunique() == 2}
    yheld = {st: g.y.to_numpy() for st, g in held.groupby("study")
             if g.y.nunique() == 2}

    def run(arm, seed, want_frames=False):
        rng = np.random.default_rng(seed)
        sp = rng.choice(at, N, replace=False)
        sn = rng.choice(bn, N, replace=False)
        if arm in ("quantile", "replace"):
            ap = greedy_quantile(D, list(sp), list(ot), K, Q)
            an = greedy_quantile(D, list(sn), list(on), K, Q)
        else:
            ap = list(rng.choice(ot, K, replace=False))
            an = list(rng.choice(on, K, replace=False))
        rows, frames = [], {}
        for k in range(K + 1):
            if arm == "replace":   # remove k confounded anchors while adding k diverse
                coh = np.concatenate([sp[k:], np.array(ap[:k], int),
                                      sn[k:], np.array(an[:k], int)])
            else:                  # ADD: confounded anchors stay
                coh = np.concatenate([sp, np.array(ap[:k], int),
                                      sn, np.array(an[:k], int)])
            X = Zpool.loc[coh].to_numpy()
            y = poolr.loc[coh, "y"].to_numpy()
            clf = LogisticRegression(max_iter=2000).fit(X, y)
            w = unit(clf.coef_[0])
            aucs = [roc_auc_score(yheld[st],
                    clf.predict_proba(Zheld[st])[:, 1]) for st in Zheld]
            rows.append((k, w @ d_batch, w @ d_bio,
                         float(np.linalg.norm([w @ e_batch, w @ e_bio])),
                         float(np.mean(aucs))))
            if want_frames and k in (0, K):
                frames[k] = (w, coh, list(ap[:k]) + list(an[:k]))
        return np.array(rows), frames

    # --- averaged curves ---
    curves = {}
    for arm in ["quantile", "random", "replace"]:
        acc = np.stack([run(arm, sd)[0] for sd in range(SEEDS)])
        curves[arm] = acc.mean(0)
        m = curves[arm]
        print(f"\n[{arm}]  k: cos_batch  cos_bio  inplane  heldAUC")
        for r in m[::max(1, K // 6)]:
            print(f"   k={int(r[0]):2d}: {r[1]:+.2f}     {r[2]:+.2f}    "
                  f"{r[3]:.2f}     {r[4]:.2f}")

    # --- plot curves ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for arm, ls in [("quantile", "-"), ("random", "--"), ("replace", "-.")]:
        m = curves[arm]
        ax[0].plot(m[:, 0], m[:, 1], ls, color="#d62728", label=f"{arm}: cos(w,batch)")
        ax[0].plot(m[:, 0], m[:, 2], ls, color="#1f77b4", label=f"{arm}: cos(w,bio)")
        ax[1].plot(m[:, 0], m[:, 4], ls, color="k", label=f"{arm}: held-out AUC")
    ax[0].axhline(0, c="gray", lw=0.6, ls=":")
    ax[0].set_xlabel("# diverse samples added / class (k)")
    ax[0].set_ylabel("cosine of w with axis")
    ax[0].set_title("w rotates: OFF batch (red), ONTO biology (blue)")
    ax[0].legend(fontsize=7)
    ax[1].axhline(0.5, c="gray", lw=0.6, ls=":")
    ax[1].set_xlabel("# diverse samples added / class (k)")
    ax[1].set_ylabel("held-out macro AUC")
    ax[1].set_title("...and held-out AUC recovers from <0.5")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Confounded seed (tumors={A}, normals={B}): diversity de-rotates the classifier")
    plt.tight_layout()
    plt.savefig("explain_confound_curves.png", dpi=110)
    print("\nwrote explain_confound_curves.png")

    # --- geometry frames (one seed, quantile) ---
    if not DO_FRAMES:
        return
    _, frames = run("quantile", FRAME_SEED, want_frames=True)

    def proj(M):
        return np.c_[M @ e_batch, M @ e_bio]

    fig, axf = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax_, k in zip(axf, [0, K]):
        w, coh, added = frames[k]
        for tp, col in [("tumor", "#d62728"), ("normal", "#1f77b4")]:
            P = proj(Zall.loc[s[s.sample_type == tp].index].to_numpy())
            ax_.scatter(P[:, 0], P[:, 1], s=8, c=col, alpha=0.10, linewidths=0)
        if added:
            Pa = proj(Zpool.loc[added].to_numpy())
            asy = poolr.loc[added, "assay"].to_numpy()
            for a in np.unique(asy):
                mm = asy == a
                ax_.scatter(Pa[mm, 0], Pa[mm, 1], s=70, marker="*",
                            edgecolors="k", linewidths=0.5, label=a)
        pw = np.array([w @ e_batch, w @ e_bio]); pw = pw / np.linalg.norm(pw) * 2.5
        ax_.annotate("", xy=(pw[0], pw[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", lw=2.5, color="k"))
        ax_.axhline(0, ls=":", c="gray", lw=0.7); ax_.axvline(0, ls=":", c="gray", lw=0.7)
        ax_.set_xlabel("batch axis (d_batch)")
        if k == 0:
            ax_.set_ylabel("biology axis (d_bio)")
        ax_.set_title(f"k={k}   cos(w,batch)={w@d_batch:+.2f}  cos(w,bio)={w@d_bio:+.2f}")
        ax_.legend(fontsize=6, loc="upper left", framealpha=0.9)
    fig.suptitle(f"Confounded seed: w starts on BATCH, rotates to BIOLOGY (quantile, seed {FRAME_SEED})")
    plt.tight_layout()
    plt.savefig("explain_confound_frames.png", dpi=110)
    print("wrote explain_confound_frames.png")


if __name__ == "__main__":
    main()
