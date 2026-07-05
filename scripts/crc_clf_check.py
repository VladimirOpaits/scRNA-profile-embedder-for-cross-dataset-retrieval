import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import crc_enrichment as ce
from selection import greedy_quantile

SEEDS = 20


def clf(name):
    if name == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "rbf_svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
    if name == "rf":
        return RandomForestClassifier(n_estimators=400, max_depth=5, random_state=0)


def score(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        return model.decision_function(X)


def cross_auc(name, Xtr, ytr, held, SC):
    m = clf(name).fit(Xtr, ytr)
    out = {}
    for st, g in held.groupby("study"):
        yte = g.y.to_numpy()
        if yte.min() == yte.max():
            continue
        out[st] = roc_auc_score(yte, score(m, g[SC].to_numpy()))
    return out


def main():
    s = ce.load()
    SC = ce.SCOLS
    held, pool = ce.split(s)
    D = ce.l2_distance_matrix(pool[SC].to_numpy())
    ceil = ce.ceilings(held)
    base = pool[pool.assay == ce.BASE_TECH]
    other = pool[pool.assay != ce.BASE_TECH]

    names = ["linear", "rbf_svm", "rf"]
    acc = {n: {st: {0: [], 20: []} for st in ce.HELD} for n in names}

    for sd in range(SEEDS):
        rng = np.random.default_rng(sd)
        sp_ = rng.choice(base[base.y == 1].index.to_numpy(), ce.N_POS, replace=False)
        sn = rng.choice(base[base.y == 0].index.to_numpy(), ce.N_NEG, replace=False)
        rp = other[other.y == 1].index.to_numpy()
        rn = other[other.y == 0].index.to_numpy()
        ap = greedy_quantile(D, list(sp_), list(rp), ce.K, ce.QUANT)
        an = greedy_quantile(D, list(sn), list(rn), ce.K, ce.QUANT)
        for k in (0, 20):
            coh = np.concatenate([sp_[k:], np.array(ap[:k], int),
                                  sn[k:], np.array(an[:k], int)])
            X = pool.loc[coh, SC].to_numpy()
            y = pool.loc[coh, "y"].to_numpy()
            for n in names:
                for st, v in cross_auc(n, X, y, held, SC).items():
                    acc[n][st][k].append(v)

    print(f"{'clf':8s} {'study':16s} {'ceil':>5s} {'auc@0':>6s} {'auc@20':>7s} {'closed':>7s}")
    for n in names:
        macro0, macro20, cs = [], [], []
        for st in ce.HELD:
            a0, a20, c = np.mean(acc[n][st][0]), np.mean(acc[n][st][20]), ceil[st]
            macro0.append(a0); macro20.append(a20); cs.append(c)
            print(f"{n:8s} {st:16s} {c:5.2f} {a0:6.2f} {a20:7.2f} {(a20-a0):7.2f}")
        m0, m20, mc = np.mean(macro0), np.mean(macro20), np.mean(cs)
        print(f"{n:8s} {'MACRO':16s} {mc:5.2f} {m0:6.2f} {m20:7.2f} {(m20-m0):7.2f}\n")


if __name__ == "__main__":
    main()
