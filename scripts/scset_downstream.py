import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from selection import cosine_distance_matrix

EMB = "data/scset/patient_emb.parquet"
ADENO = "lung adenocarcinoma"
NORMAL = "normal"


def load():
    df = pd.read_parquet(EMB)
    df = df[df.disease.isin([ADENO, NORMAL])].reset_index(drop=True)
    ecols = [c for c in df.columns if c.startswith("e")]
    V = df[ecols].to_numpy(np.float32)
    return df, V


def mean_pair(D, labels, same):
    n = len(labels)
    v = [D[i, j] for i in range(n) for j in range(i + 1, n)
         if (labels[i] == labels[j]) == same]
    return float(np.mean(v))


def dominance(V, tech, dis):
    D = cosine_distance_matrix(V)
    st, dt = mean_pair(D, tech, True), mean_pair(D, tech, False)
    sd, dd = mean_pair(D, dis, True), mean_pair(D, dis, False)
    print(f"technology gap: {dt-st:+.4f} (same {st:.3f} / diff {dt:.3f})")
    print(f"diagnosis  gap: {dd-sd:+.4f} (same {sd:.3f} / diff {dd:.3f})")
    print(f"tech-gap / bio-gap = {(dt-st)/(dd-sd):.2f}x "
          f"(>1 => technology dominates)")


def cv_auc(V, y, groups=None, seed=0):
    Xs = StandardScaler().fit_transform(V)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def loto(V, y, tech):
    Xs = StandardScaler().fit_transform(V)
    aucs = {}
    for t in sorted(set(tech)):
        tr, te = tech != t, tech == t
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            aucs[t] = np.nan
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[tr], y[tr])
        aucs[t] = roc_auc_score(y[te], clf.predict_proba(Xs[te])[:, 1])
    return aucs


def main():
    df, V = load()
    tech = df.technology.to_numpy()
    dis = df.disease.to_numpy()
    y = (dis == ADENO).astype(int)
    print(f"patients: {len(df)} | adeno {y.sum()} / normal {(1-y).sum()}")
    print("techs:", dict(pd.Series(tech).value_counts()))

    print("\n=== structure (cosine on scSet patient vectors) ===")
    dominance(V, tech, dis)

    print("\n=== diagnosis predictability (5-fold CV logreg AUC) ===")
    print(f"adeno-vs-normal AUC: {cv_auc(V, y):.3f}")
    ty = pd.factorize(tech)[0]
    if len(set(ty)) == 2:
        print(f"technology AUC (2-tech): "
              f"{cv_auc(V, (ty == ty[0]).astype(int)):.3f}")

    print("\n=== leave-one-technology-out diagnosis AUC ===")
    for t, a in loto(V, y, tech).items():
        n = (tech == t).sum()
        print(f"  hold {t:12} (n={n}): AUC {a:.3f}")


if __name__ == "__main__":
    main()
