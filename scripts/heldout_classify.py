import numpy as np
import pandas as pd
import lancedb
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

from selection import cosine_distance_matrix, greedy_farthest, greedy_quantile

ADENO = "lung adenocarcinoma"
NORMAL = "normal"
N_PER_CLASS = 6
K = 4
REMOVE_SEED = 0
RANDOM_SEEDS = range(5)
PCA_CELL = 50
PCA_PB = 10
KNN_K = 25
LOGREG_C = 0.1


def load():
    t = lancedb.connect("vectors").open_table("cells")
    df = t.search().limit(t.count_rows()).select(
        ["sample_id", "technology", "disease", "vector"]).to_pandas()
    V = np.stack(df["vector"].to_numpy())
    meta = df.drop_duplicates("sample_id").set_index("sample_id")[
        ["technology", "disease"]]
    cellvecs, pb = {}, {}
    for sid, g in df.groupby("sample_id"):
        cellvecs[sid] = V[g.index.to_numpy()]
        pb[sid] = cellvecs[sid].mean(axis=0)
    samples = sorted(pb)
    M = np.vstack([pb[s] for s in samples])
    return meta, samples, M, cellvecs, pb


def roles(meta, held, techs):
    train_t = [t for t in techs if t != held]
    ad = {t: meta[(meta.disease == ADENO) & (meta.technology == t)] for t in train_t}
    adeno_dom = max(train_t, key=lambda t: (len(ad[t]), t))
    normal_dom = [t for t in train_t if t != adeno_dom][0]
    return adeno_dom, normal_dom


def sets_for(meta, adeno_dom, normal_dom):
    def sel(dis, tech):
        return sorted(meta[(meta.disease == dis) & (meta.technology == tech)].index)
    cohort_ad = sel(ADENO, adeno_dom)[:N_PER_CLASS]
    cohort_no = sel(NORMAL, normal_dom)[:N_PER_CLASS]
    pool_ad = sel(ADENO, normal_dom)          # adeno from the other train tech
    pool_no = sel(NORMAL, adeno_dom)          # normal from the other train tech
    return cohort_ad, cohort_no, pool_ad, pool_no


def plan(arm, q, seed, sets, samples, D):
    cohort_ad, cohort_no, pool_ad, pool_no = sets
    idx = {s: i for i, s in enumerate(samples)}
    rng = np.random.default_rng(REMOVE_SEED)
    rm_ad = set(rng.choice(cohort_ad, size=K, replace=False))
    rng2 = np.random.default_rng(REMOVE_SEED + 1)
    rm_no = set(rng2.choice(cohort_no, size=K, replace=False))
    keep_ad = [s for s in cohort_ad if s not in rm_ad]
    keep_no = [s for s in cohort_no if s not in rm_no]

    if arm == "PURE":
        return cohort_ad + cohort_no

    def pick(keep, pool):
        ck = [idx[s] for s in keep]
        pk = [idx[s] for s in pool]
        if arm == "RETRIEVAL":
            o = greedy_farthest(D, ck, pk, K)
        elif arm.startswith("QUANT"):
            o = greedy_quantile(D, ck, pk, K, q)
        else:
            r = np.random.default_rng(1000 + seed + (hash(pool[0]) % 97))
            o = [idx[s] for s in r.choice(pool, size=K, replace=False)]
        return [samples[i] for i in o]

    return keep_ad + pick(keep_ad, pool_ad) + keep_no + pick(keep_no, pool_no)


def y_of(sids, meta):
    return np.array([1 if meta.loc[s, "disease"] == ADENO else 0 for s in sids])


def eval_cellknn(train, test, cellvecs, meta):
    Xtr = np.vstack([cellvecs[s] for s in train])
    ytr = np.concatenate([[1 if meta.loc[s, "disease"] == ADENO else 0]
                          * len(cellvecs[s]) for s in train])
    pca = PCA(PCA_CELL).fit(Xtr)
    knn = KNeighborsClassifier(n_neighbors=KNN_K).fit(pca.transform(Xtr), ytr)
    return {s: knn.predict_proba(pca.transform(cellvecs[s]))[:, 1].mean() for s in test}


def eval_pblog(train, test, pb, meta):
    Xtr = np.vstack([pb[s] for s in train])
    ytr = y_of(train, meta)
    nc = min(PCA_PB, len(train) - 1)
    pca = PCA(nc).fit(Xtr)
    clf = LogisticRegression(C=LOGREG_C, max_iter=2000).fit(pca.transform(Xtr), ytr)
    return {s: clf.predict_proba(pca.transform(pb[s][None, :]))[0, 1] for s in test}


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if len(np.unique(y[b])) < 2:
            continue
        a.append(roc_auc_score(y[b], p[b]))
    return np.percentile(a, [2.5, 97.5])


def run_arm(arm, q, seed, meta, samples, D, sets_by_fold, cellvecs, pb, techs):
    preds, truth, fold_auc = {}, {}, {}
    for held in techs:
        sets = sets_by_fold[held]
        train = plan(arm, q, seed, sets, samples, D)
        test = sorted(meta[meta.technology == held].index)
        pk = eval_cellknn(train, test, cellvecs, meta)
        pl = eval_pblog(train, test, pb, meta)
        yt = y_of(test, meta)
        fold_auc[held] = (roc_auc_score(yt, [pk[s] for s in test]),
                          roc_auc_score(yt, [pl[s] for s in test]))
        for s in test:
            truth[s] = 1 if meta.loc[s, "disease"] == ADENO else 0
            preds[s] = (pk[s], pl[s])
    sids = list(preds)
    y = np.array([truth[s] for s in sids])
    pk = np.array([preds[s][0] for s in sids])
    pl = np.array([preds[s][1] for s in sids])
    return y, pk, pl, fold_auc


def run():
    meta, samples, M, cellvecs, pb = load()
    D = cosine_distance_matrix(M)
    techs = sorted(meta.technology.unique())
    sets_by_fold = {}
    print("=== folds / confound roles ===")
    for held in techs:
        ad, no = roles(meta, held, techs)
        sets_by_fold[held] = sets_for(meta, ad, no)
        print(f"held-out {held:12} | train adeno<-{ad:12} normal<-{no:12} "
              f"| test n={len(meta[meta.technology==held])}")

    arms = [("PURE", 1.0), ("RETRIEVAL", 1.0), ("QUANT0.5", 0.5),
            ("QUANT0.75", 0.75), ("RANDOM", 1.0)]
    rows = []
    for arm, q in arms:
        if arm == "RANDOM":
            aucs_k, aucs_l = [], []
            for s in RANDOM_SEEDS:
                y, pk, pl, _ = run_arm(arm, q, s, meta, samples, D,
                                       sets_by_fold, cellvecs, pb, techs)
                aucs_k.append(roc_auc_score(y, pk))
                aucs_l.append(roc_auc_score(y, pl))
            rows.append({"arm": arm, "knn_AUC": np.mean(aucs_k),
                         "knn_sd": np.std(aucs_k), "log_AUC": np.mean(aucs_l),
                         "log_sd": np.std(aucs_l)})
            print(f"\n{arm}: kNN AUC {np.mean(aucs_k):.3f}±{np.std(aucs_k):.3f} "
                  f"| logreg AUC {np.mean(aucs_l):.3f}±{np.std(aucs_l):.3f} (5 seeds)")
        else:
            y, pk, pl, fa = run_arm(arm, q, 0, meta, samples, D,
                                    sets_by_fold, cellvecs, pb, techs)
            ak, al = roc_auc_score(y, pk), roc_auc_score(y, pl)
            ck, cl = boot_ci(y, pk), boot_ci(y, pl)
            f1k = f1_score(y, (pk > 0.5).astype(int), average="macro")
            f1l = f1_score(y, (pl > 0.5).astype(int), average="macro")
            rows.append({"arm": arm, "knn_AUC": ak, "knn_CI": ck,
                         "log_AUC": al, "log_CI": cl, "knn_F1": f1k, "log_F1": f1l})
            print(f"\n{arm}: kNN AUC {ak:.3f} [{ck[0]:.2f},{ck[1]:.2f}] F1={f1k:.3f}"
                  f" | logreg AUC {al:.3f} [{cl[0]:.2f},{cl[1]:.2f}] F1={f1l:.3f}")
            print("  per-fold (kNN,log):",
                  {h: (round(a, 2), round(b, 2)) for h, (a, b) in fa.items()})
    pd.DataFrame(rows).to_csv("heldout_results.csv", index=False)
    return rows


if __name__ == "__main__":
    run()
