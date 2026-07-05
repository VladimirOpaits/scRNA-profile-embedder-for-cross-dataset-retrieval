import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from selection import greedy_farthest, greedy_quantile
from crc_enrichment import l2_distance_matrix

SIG = os.environ.get("CRC_SIG", "data/crc/signatures.parquet")
LAB = "data/crc/msi_labels.parquet"
HELD = ["Joanito_2022_Nat_Genet", "Chen_2024_Cancer_Cell", "Borras_2023_Cell_Discov"]
BASE_TECH = "10x 3' v3"
QUANT = 0.75
N_POS = 10
N_NEG = 10
K = 15
SEEDS = 50


def load():
    s = pd.read_parquet(SIG)
    global SCOLS
    SCOLS = [c for c in s.columns if c.startswith("s") and c[1:].isdigit()]
    lab = pd.read_parquet(LAB)
    s = s.merge(lab, on="sample_id")
    s = s[(s.sample_type == "tumor") & (s.msi.isin(["MSI", "MSS"]))].reset_index(drop=True)
    s["y"] = (s.msi == "MSI").astype(int)
    return s


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
        a = per_study_auc(Xtr, ytr, held, arm == "negctrl", rng)
        for st, v in a.items():
            rows.append((arm, seed, k, st, v))
    return rows


def main():
    s = load()
    held = s[s.study.isin(HELD)].copy()
    pool = s[~s.study.isin(HELD)].reset_index(drop=True)
    D = l2_distance_matrix(pool[SCOLS].to_numpy())
    ceil = ceilings(held)
    for st, g in held.groupby("study"):
        print(f"held {st}: {dict(g.msi.value_counts())} | ceiling={ceil.get(st,float('nan')):.3f}")
    pb = pool[pool.assay == BASE_TECH]
    print(f"pool: {len(pool)} ({int(pool.y.sum())} MSI) | base {BASE_TECH}: "
          f"{int((pb.y==1).sum())} MSI / {int((pb.y==0).sum())} MSS | "
          f"other-tech MSI={int(((pool.assay!=BASE_TECH)&(pool.y==1)).sum())}")

    rows = []
    for arm in ["quantile", "farthest", "random", "negctrl"]:
        for sd in range(SEEDS):
            rows += cohort_curve(pool, D, held, arm, sd)
    res = pd.DataFrame(rows, columns=["arm", "seed", "k", "study", "auc"])
    res.to_csv("crc_msi_results.csv", index=False)

    g = res.groupby(["arm", "study", "k"]).auc.mean().reset_index()
    print("\n=== MSI-vs-MSS  GAP TO CEILING (auc@0 -> auc@K, closed) ===")
    print(f"{'study':16s} {'ceil':>5s} {'arm':>9s} {'auc@0':>6s} {'auc@K':>6s} {'closed':>7s}")
    for st in HELD + ["macro"]:
        c = ceil.get(st, np.nan) if st != "macro" else np.mean(list(ceil.values()))
        for arm in ["quantile", "farthest", "random", "negctrl"]:
            a0 = g[(g.arm == arm) & (g.study == st) & (g.k == 0)].auc.iloc[0]
            aK = g[(g.arm == arm) & (g.study == st) & (g.k == K)].auc.iloc[0]
            closed = (c - a0) - (c - aK) if np.isfinite(c) else np.nan
            print(f"{st:16s} {c:5.2f} {arm:>9s} {a0:6.2f} {aK:6.2f} {closed:7.2f}")
        print()


if __name__ == "__main__":
    main()
