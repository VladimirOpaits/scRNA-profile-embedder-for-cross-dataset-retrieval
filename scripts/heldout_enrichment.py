import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from selection import cosine_distance_matrix, greedy_farthest, greedy_quantile

EMB = "data/scset/patient_emb.parquet"
META = "data/scvi_corpus/meta.csv"
HELD = ["576f193c", "9f222629", "b617ee1b"]
QUANT = 0.75
BASE_TECH = "10x 3' v2"
MALIG = {"lung adenocarcinoma", "squamous cell lung carcinoma",
         "non-small cell lung carcinoma", "small cell lung carcinoma",
         "lung cancer", "lung large cell carcinoma", "pleomorphic carcinoma"}
CLASSES = MALIG | {"normal"}
N_AD = 25
N_NO = 25
K = 20
SEEDS = 50
OUT = "enrichment_results.csv"
ECOLS = [f"e{j}" for j in range(256)]


def load():
    e = pd.read_parquet(EMB)
    m = pd.read_csv(META)[["pid", "dataset_id", "donor_id", "assay"]]
    df = e.merge(m, on="pid", how="left")
    df = df[df["disease"].isin(CLASSES)].reset_index(drop=True)
    df["y"] = df["disease"].isin(MALIG).astype(int)
    return df


def is_held(s):
    return s.str.startswith(tuple(HELD))


def study_of(s):
    for h in HELD:
        if s.startswith(h):
            return h
    return "?"


def split(df):
    held = df[is_held(df["dataset_id"])].copy()
    held["study"] = held["dataset_id"].map(study_of)
    bad = set(zip(held["donor_id"], held["assay"]))
    pool = df[~is_held(df["dataset_id"])].copy()
    mask = np.array([(d, a) not in bad for d, a in zip(pool["donor_id"], pool["assay"])])
    pool = pool[mask]
    return held.reset_index(drop=True), pool.reset_index(drop=True)


def ceilings(held):
    out = {}
    for st, g in held.groupby("study"):
        X = g[ECOLS].to_numpy()
        y = g["y"].to_numpy()
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
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(Xtr), y)
    out = {}
    for st, g in held.groupby("study"):
        yte = g["y"].to_numpy()
        if yte.min() == yte.max():
            continue
        p = clf.predict_proba(sc.transform(g[ECOLS].to_numpy()))[:, 1]
        out[st] = roc_auc_score(yte, p)
    out["macro"] = float(np.mean(list(out.values())))
    return out


def cohort_curve(pool, D, held, arm, seed):
    rng = np.random.default_rng(seed)

    base = pool[pool["assay"] == BASE_TECH]
    other = pool[pool["assay"] != BASE_TECH]

    def pick(sub, n):
        idx = sub.index.to_numpy()
        return rng.choice(idx, size=n, replace=False)

    seed_ad = pick(base[base["y"] == 1], N_AD)
    seed_no = pick(base[base["y"] == 0], N_NO)

    r_ad = other[other["y"] == 1].index.to_numpy()
    r_no = other[other["y"] == 0].index.to_numpy()

    if arm == "farthest":
        add_ad = greedy_farthest(D, list(seed_ad), list(r_ad), K)
        add_no = greedy_farthest(D, list(seed_no), list(r_no), K)
    elif arm == "quantile":
        add_ad = greedy_quantile(D, list(seed_ad), list(r_ad), K, QUANT)
        add_no = greedy_quantile(D, list(seed_no), list(r_no), K, QUANT)
    else:
        add_ad = list(rng.choice(r_ad, size=K, replace=False))
        add_no = list(rng.choice(r_no, size=K, replace=False))

    rows = []
    for k in range(0, K + 1):
        coh = np.concatenate([
            seed_ad[k:], np.array(add_ad[:k], dtype=int),
            seed_no[k:], np.array(add_no[:k], dtype=int)])
        Xtr = pool.loc[coh, ECOLS].to_numpy()
        ytr = pool.loc[coh, "y"].to_numpy()
        n_tech = pool.loc[coh, "assay"].nunique()
        a = per_study_auc(Xtr, ytr, held, arm == "negctrl", rng)
        for st, v in a.items():
            rows.append((arm, seed, k, n_tech, st, v))
    return rows


def main():
    df = load()
    held, pool = split(df)
    Dfull = cosine_distance_matrix(pool[ECOLS].to_numpy())

    ceil = ceilings(held)
    for h, g in held.groupby("study"):
        print(f"held {h}: {dict(g['disease'].value_counts())} | "
              f"within-study ceiling={ceil.get(h, float('nan')):.3f}")
    print(f"pool: {len(pool)} ({int(pool['y'].sum())} adeno) techs={pool['assay'].nunique()}")

    arms = ["quantile", "farthest", "random", "negctrl"]
    all_rows = []
    for arm in arms:
        for s in range(SEEDS):
            all_rows += cohort_curve(pool, Dfull, held, arm, s)

    res = pd.DataFrame(all_rows, columns=["arm", "seed", "k", "n_techs", "study", "auc"])
    res.to_csv(OUT, index=False)

    studies = HELD + ["macro"]
    g = res.groupby(["arm", "study", "k"]).agg(
        m=("auc", "mean"), sd=("auc", "std"),
        lo=("auc", lambda x: x.quantile(0.1)),
        frac_bad=("auc", lambda x: float((x < 0.5).mean()))).reset_index()
    for arm in ["quantile", "negctrl"]:
        print(f"\n=== {arm} : per-study mean [10th pct] (frac AUC<0.5) ===")
        print("  k  " + "".join(f"{s:>26s}" for s in studies))
        for k in [0, 5, 10, 15, 20]:
            cells = []
            for s in studies:
                r = g[(g.arm == arm) & (g.study == s) & (g.k == k)]
                if len(r):
                    r = r.iloc[0]
                    cells.append(f"{r.m:.2f} [{r.lo:.2f}] ({r.frac_bad:.0%})")
                else:
                    cells.append("-")
            print(f" {k:2d}  " + "".join(f"{c:>26s}" for c in cells))
    print("\n=== GAP TO WITHIN-STUDY CEILING (does enrichment close it?) ===")
    print(f"{'study':10s} {'ceil':>5s} {'arm':>9s} {'auc@0':>6s} {'auc@20':>7s}"
          f" {'gap@0':>6s} {'gap@20':>7s} {'closed':>7s}")
    for st in HELD:
        c = ceil.get(st)
        if c is None:
            continue
        for arm in ["quantile", "farthest", "random"]:
            a0 = g[(g.arm == arm) & (g.study == st) & (g.k == 0)]["m"].iloc[0]
            a20 = g[(g.arm == arm) & (g.study == st) & (g.k == 20)]["m"].iloc[0]
            g0, g20 = c - a0, c - a20
            print(f"{st:10s} {c:5.2f} {arm:>9s} {a0:6.2f} {a20:7.2f}"
                  f" {g0:6.2f} {g20:7.2f} {g0 - g20:7.2f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
