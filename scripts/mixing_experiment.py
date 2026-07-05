import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import hlca_experiment as H
import heldout_classify as C
from selection import cosine_distance_matrix, greedy_quantile

A = "10x 3' v2"      # home of normal
B = "Smart-seq2"     # home of adeno
TEST = "10x 3' v3"   # held-out technology
N = 8
Q = 0.75
MS = [1.0, 0.75, 0.5]
SEEDS = range(5)
CSV = "mixing_results.csv"


def build_cohort(m, seed, meta, samples, D, idx):
    home_ad = sorted(meta[(meta.disease == H.ADENO) & (meta.technology == B)].index)
    cross_ad = sorted(meta[(meta.disease == H.ADENO) & (meta.technology == A)].index)
    home_no = sorted(meta[(meta.disease == H.NORMAL) & (meta.technology == A)].index)
    cross_no = sorted(meta[(meta.disease == H.NORMAL) & (meta.technology == B)].index)
    h = int(round(m * N))
    c = N - h
    rng = np.random.default_rng(seed)
    ad_home = list(rng.choice(home_ad, size=min(h, len(home_ad)), replace=False))
    no_home = list(rng.choice(home_no, size=min(h, len(home_no)), replace=False))

    def qsel(home, pool, k):
        if k <= 0:
            return []
        o = greedy_quantile(D, [idx[s] for s in home], [idx[s] for s in pool], k, Q)
        return [samples[i] for i in o]

    return ad_home + qsel(ad_home, cross_ad, c) + no_home + qsel(no_home, cross_no, c)


def probing(cohort):
    a = H.build_raw(cohort)
    Z, tech, dis, ct = H.harmony_pca(a)
    bio = H._knn_f1(Z, dis)
    bp = H._knn_f1(Z, tech)
    return bio, (1.0 - bp if bp == bp else np.nan)


def heldout(cohort, test, cellvecs, pb, meta):
    yt = np.array([1 if meta.loc[s, "disease"] == H.ADENO else 0 for s in test])
    pk = C.eval_cellknn(cohort, test, cellvecs, meta)
    pl = C.eval_pblog(cohort, test, pb, meta)
    return (roc_auc_score(yt, [pk[s] for s in test]),
            roc_auc_score(yt, [pl[s] for s in test]))


def run():
    meta, samples, M, cellvecs, pb = C.load()
    D = cosine_distance_matrix(M)
    idx = {s: i for i, s in enumerate(samples)}
    test = sorted(meta[meta.technology == TEST].index)
    print("held-out test:", TEST, "| n =", len(test),
          "|", dict(meta.loc[test].disease.value_counts()))

    rows = []
    for m in MS:
        for seed in SEEDS:
            coh = build_cohort(m, seed, meta, samples, D, idx)
            bio, bmix = probing(coh)
            auc_k, auc_l = heldout(coh, test, cellvecs, pb, meta)
            rows.append({"m": m, "seed": seed, "BioPred": bio, "BatchMix": bmix,
                         "heldout_AUC_knn": auc_k, "heldout_AUC_log": auc_l})
            print(f"m={m} seed={seed} | Bio={bio:.3f} BatchMix={bmix:.3f} "
                  f"| heldAUC knn={auc_k:.3f} log={auc_l:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    print("\n=== dose-response (mean over seeds) ===")
    g = df.groupby("m").agg(["mean", "std"]).round(3)
    print(g[["BioPred", "BatchMix", "heldout_AUC_knn", "heldout_AUC_log"]].to_string())
    return df


if __name__ == "__main__":
    run()
