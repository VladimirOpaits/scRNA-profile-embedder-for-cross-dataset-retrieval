import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from selection import greedy_farthest, greedy_quantile

SIG = os.environ.get("CRC_SIG", "data/crc/signatures.parquet")
HELD = ["Chen_2024_Cancer_Cell", "Lee_2020_Nat_Genet", "Uhlitz_2021_EMBO_Mol_Med",
        "MUI_Innsbruck", "Zhang_2020_Cell"]
BASE_TECH = "10x 5' v1"
QUANT = 0.90
N_POS = 25
N_NEG = 25
K = 20
SEEDS = 50
OUT = "crc_enrichment_results.csv"


def l2_distance_matrix(M):
    g = (M * M).sum(1)
    d2 = g[:, None] + g[None, :] - 2 * M @ M.T
    return np.sqrt(np.maximum(d2, 0))


def load():
    s = pd.read_parquet(SIG)
    global SCOLS
    SCOLS = [c for c in s.columns if c.startswith("s") and c[1:].isdigit()]
    s = s[s.sample_type.isin(["tumor", "normal"])].reset_index(drop=True)
    s["y"] = (s.sample_type == "tumor").astype(int)
    return s


def split(s):
    held = s[s.study.isin(HELD)].copy()
    pool = s[~s.study.isin(HELD)].copy()
    return held.reset_index(drop=True), pool.reset_index(drop=True)


def ceilings(held):
    out = {}
    for st, g in held.groupby("study"):
        X, y = g[SCOLS].to_numpy(), g.y.to_numpy()
        if y.min() == y.max():
            continue
        pr = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            pr[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        out[st] = roc_auc_score(y, pr)
    return out


def per_study_auc(Xtr, ytr, held, shuffle, rng):
    y = ytr.copy()
    if shuffle:
        rng.shuffle(y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), y)
    out = {}
    for st, g in held.groupby("study"):
        yte = g.y.to_numpy()
        if yte.min() == yte.max():
            continue
        p = clf.predict_proba(sc.transform(g[SCOLS].to_numpy()))[:, 1]
        out[st] = roc_auc_score(yte, p)
    out["macro"] = float(np.mean(list(out.values())))
    return out


def cohort_curve(pool, D, held, arm, seed):
    rng = np.random.default_rng(seed)
    base = pool[pool.assay == BASE_TECH]
    other = pool[pool.assay != BASE_TECH]

    def pick(sub, n):
        return rng.choice(sub.index.to_numpy(), n, replace=False)

    sp_ = pick(base[base.y == 1], N_POS)
    sn = pick(base[base.y == 0], N_NEG)
    rp = other[other.y == 1].index.to_numpy()
    rn = other[other.y == 0].index.to_numpy()

    if arm == "farthest":
        ap = greedy_farthest(D, list(sp_), list(rp), K)
        an = greedy_farthest(D, list(sn), list(rn), K)
    elif arm == "quantile":
        ap = greedy_quantile(D, list(sp_), list(rp), K, QUANT)
        an = greedy_quantile(D, list(sn), list(rn), K, QUANT)
    else:
        ap = list(rng.choice(rp, K, replace=False))
        an = list(rng.choice(rn, K, replace=False))

    rows = []
    for k in range(K + 1):
        coh = np.concatenate([sp_[k:], np.array(ap[:k], int),
                              sn[k:], np.array(an[:k], int)])
        Xtr = pool.loc[coh, SCOLS].to_numpy()
        ytr = pool.loc[coh, "y"].to_numpy()
        nt = pool.loc[coh, "assay"].nunique()
        a = per_study_auc(Xtr, ytr, held, arm == "negctrl", rng)
        for st, v in a.items():
            rows.append((arm, seed, k, nt, st, v))
    return rows


def main():
    s = load()
    held, pool = split(s)
    D = l2_distance_matrix(pool[SCOLS].to_numpy())
    ceil = ceilings(held)
    for st, g in held.groupby("study"):
        print(f"held {st}: {dict(g.sample_type.value_counts())} | ceiling={ceil.get(st,float('nan')):.3f}")
    pb = pool[pool.assay == BASE_TECH]
    print(f"pool: {len(pool)} ({int(pool.y.sum())} tumor) | base {BASE_TECH}: "
          f"{int((pb.y==1).sum())} tumor / {int((pb.y==0).sum())} normal | assays={pool.assay.nunique()}")

    arms = ["quantile", "farthest", "random", "negctrl"]
    rows = []
    for arm in arms:
        for sd in range(SEEDS):
            rows += cohort_curve(pool, D, held, arm, sd)
    res = pd.DataFrame(rows, columns=["arm", "seed", "k", "n_techs", "study", "auc"])
    res.to_csv(OUT, index=False)

    g = res.groupby(["arm", "study", "k"]).auc.mean().reset_index()
    print("\n=== GAP TO CEILING (auc@0 -> auc@20, closed) ===")
    print(f"{'study':16s} {'ceil':>5s} {'arm':>9s} {'auc@0':>6s} {'auc@20':>7s} {'closed':>7s}")
    for st in HELD + ["macro"]:
        c = ceil.get(st, np.nan) if st != "macro" else np.mean(list(ceil.values()))
        for arm in ["quantile", "farthest", "random", "negctrl"]:
            a0 = g[(g.arm == arm) & (g.study == st) & (g.k == 0)].auc.iloc[0]
            a20 = g[(g.arm == arm) & (g.study == st) & (g.k == 20)].auc.iloc[0]
            closed = (c - a0) - (c - a20) if np.isfinite(c) else np.nan
            print(f"{st:16s} {c:5.2f} {arm:>9s} {a0:6.2f} {a20:7.2f} {closed:7.2f}")
        print()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
