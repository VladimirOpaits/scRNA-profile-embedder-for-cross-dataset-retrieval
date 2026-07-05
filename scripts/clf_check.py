import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from heldout_enrichment import (load, split, ceilings, BASE_TECH, ECOLS,
                                N_AD, N_NO, K, QUANT, HELD)
from selection import cosine_distance_matrix, greedy_quantile

SEEDS = 20


def clf(name):
    if name == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "poly2":
        return make_pipeline(StandardScaler(), PCA(20),
                             PolynomialFeatures(2, include_bias=False),
                             LogisticRegression(max_iter=5000, C=0.5))
    if name == "rbf_svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0,
                             probability=False))
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=1500, alpha=1e-2))
    if name == "rf":
        return make_pipeline(RandomForestClassifier(n_estimators=300,
                             max_depth=4, random_state=0))


def score(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    return model.decision_function(X)


def cross_auc(name, Xtr, ytr, held):
    model = clf(name).fit(Xtr, ytr)
    out = {}
    for st, gg in held.groupby("study"):
        yte = gg["y"].to_numpy()
        if yte.min() == yte.max():
            continue
        out[st] = roc_auc_score(yte, score(model, gg[ECOLS].to_numpy()))
    return out


def main():
    df = load()
    held, pool = split(df)
    D = cosine_distance_matrix(pool[ECOLS].to_numpy())
    ceil = ceilings(held)
    base = pool[pool["assay"] == BASE_TECH]
    other = pool[pool["assay"] != BASE_TECH]

    names = ["linear", "poly2", "rbf_svm", "mlp", "rf"]
    acc = {n: {st: {0: [], 20: []} for st in HELD} for n in names}

    for s in range(SEEDS):
        rng = np.random.default_rng(s)
        sa = rng.choice(base[base.y == 1].index.to_numpy(), N_AD, replace=False)
        sn = rng.choice(base[base.y == 0].index.to_numpy(), N_NO, replace=False)
        ra = other[other.y == 1].index.to_numpy()
        rn = other[other.y == 0].index.to_numpy()
        aa = greedy_quantile(D, list(sa), list(ra), K, QUANT)
        an = greedy_quantile(D, list(sn), list(rn), K, QUANT)
        for k in (0, 20):
            coh = np.concatenate([sa[k:], np.array(aa[:k], int),
                                  sn[k:], np.array(an[:k], int)])
            X = pool.loc[coh, ECOLS].to_numpy()
            y = pool.loc[coh, "y"].to_numpy()
            for n in names:
                for st, v in cross_auc(n, X, y, held).items():
                    acc[n][st][k].append(v)

    print("ceilings:", {k: round(v, 2) for k, v in ceil.items()})
    print(f"\n{'clf':8s} {'study':10s} {'auc@0':>6s} {'auc@20':>7s} "
          f"{'ceil':>5s} {'closed':>7s}")
    for n in names:
        for st in HELD:
            a0 = np.mean(acc[n][st][0])
            a20 = np.mean(acc[n][st][20])
            c = ceil[st]
            print(f"{n:8s} {st:10s} {a0:6.2f} {a20:7.2f} {c:5.2f} "
                  f"{(c - a0) - (c - a20):7.2f}")
        print()


if __name__ == "__main__":
    main()
