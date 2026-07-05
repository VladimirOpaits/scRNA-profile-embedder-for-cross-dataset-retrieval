import os
import re
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import harmonypy as hm
import lancedb
import scib_metrics as sm
from scib_metrics.nearest_neighbors import NeighborsResults
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score

from selection import cosine_distance_matrix, greedy_farthest, greedy_quantile

H5DIR = "data/geneformer_input/hlca"
ADENO = "lung adenocarcinoma"
NORMAL = "normal"
T1 = "10x 3' v2"     # normal-dominant (cohort healthy)
T2 = "Smart-seq2"    # adeno-dominant (cohort sick)
T3 = "10x 3' v3"     # neutral third

KS = (2, 5)
RANDOM_SEEDS = range(5)
REMOVE_SEED = 0
CSV = "hlca_results.csv"


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def load_gf():
    t = lancedb.connect("vectors").open_table("cells")
    df = t.search().limit(t.count_rows()).select(
        ["sample_id", "technology", "disease", "vector"]).to_pandas()
    meta = df.drop_duplicates("sample_id").set_index("sample_id")[
        ["technology", "disease"]]
    V = np.stack(df["vector"].to_numpy())
    pb = {}
    for sid, g in df.groupby("sample_id"):
        pb[sid] = V[g.index.to_numpy()].mean(axis=0)
    samples = list(pb.keys())
    M = np.vstack([pb[s] for s in samples])
    return meta, samples, M


def define_sets(meta):
    def sel(dis, tech):
        return sorted(meta[(meta.disease == dis) & (meta.technology == tech)].index)
    cohort_ad = sel(ADENO, T2)
    cohort_no = sel(NORMAL, T1)
    pool_ad = sorted(meta[(meta.disease == ADENO)
                          & (~meta.index.isin(cohort_ad))].index)
    pool_no = sorted(meta[(meta.disease == NORMAL)
                          & (~meta.index.isin(cohort_no))].index)
    return cohort_ad, cohort_no, pool_ad, pool_no


def plan(mode, k, seed, sets, samples, D, q=1.0):
    cohort_ad, cohort_no, pool_ad, pool_no = sets
    idx = {s: i for i, s in enumerate(samples)}
    rng = np.random.default_rng(REMOVE_SEED)
    rm_ad = set(rng.choice(cohort_ad, size=k, replace=False))
    rng2 = np.random.default_rng(REMOVE_SEED + 1)
    rm_no = set(rng2.choice(cohort_no, size=k, replace=False))
    keep_ad = [s for s in cohort_ad if s not in rm_ad]
    keep_no = [s for s in cohort_no if s not in rm_no]

    if mode == "PURE":
        return cohort_ad + cohort_no
    if mode in ("RETRIEVAL", "QUANTILE"):
        def pick(keep, pool):
            ck = [idx[s] for s in keep]
            pk = [idx[s] for s in pool]
            o = (greedy_farthest(D, ck, pk, k) if mode == "RETRIEVAL"
                 else greedy_quantile(D, ck, pk, k, q))
            return [samples[i] for i in o]
        add_ad = pick(keep_ad, pool_ad)
        add_no = pick(keep_no, pool_no)
    else:  # RANDOM
        r = np.random.default_rng(1000 + seed)
        add_ad = list(r.choice(pool_ad, size=k, replace=False))
        add_no = list(r.choice(pool_no, size=k, replace=False))
    return keep_ad + add_ad + keep_no + add_no


_RAW = {}


def load_raw(sid):
    if sid not in _RAW:
        a = sc.read_h5ad(os.path.join(H5DIR, safe(sid) + ".h5ad"))
        a.obs["sample_id"] = sid
        _RAW[sid] = a
    return _RAW[sid]


def build_raw(sample_ids):
    parts = [load_raw(s) for s in sample_ids]
    a = ad.concat(parts, join="inner", index_unique="-")
    a.obs_names_make_unique()
    return a


def harmony_pca(a, n_pca=50, batch_key="technology"):
    a = a.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=n_pca)
    ho = hm.run_harmony(a.obsm["X_pca"], a.obs, [batch_key])
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != a.n_obs:
        Z = Z.T
    return (Z, a.obs["technology"].to_numpy(), a.obs["disease"].to_numpy(),
            a.obs["cell_type"].to_numpy())


def _nn(X, k=90):
    k = min(k, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, idx = nn.kneighbors(X)
    return NeighborsResults(indices=idx, distances=dist)


def _balance(labels, seed=0, cap=1500, minc=2):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    per = {}
    for c in np.unique(labels):
        per[c] = np.where(labels == c)[0]
    n = min(cap, min(len(v) for v in per.values()))
    if n < minc:
        return None
    keep = np.concatenate([rng.choice(v, size=n, replace=False) for v in per.values()])
    return np.sort(keep)


def _knn_f1(X, labels, seed=0, k=15, cv=3):
    idx = _balance(labels, seed)
    if idx is None or len(np.unique(labels[idx])) < 2:
        return np.nan
    Xc, yc = X[idx], np.asarray(labels)[idx]
    kk = min(k, np.min(np.bincount(pd.factorize(yc)[0])) - 1)
    if kk < 1:
        return np.nan
    pred = cross_val_predict(KNeighborsClassifier(n_neighbors=kk), Xc, yc, cv=cv)
    return f1_score(yc, pred, average="macro")


def _sub_random(n, cap=6000, seed=0):
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def metrics(Z, tech, dis, ct):
    out = {}
    # --- batch mixing: technology-balanced ---
    bi = _balance(tech, seed=0)
    nrb = _nn(Z[bi])
    out["iLISI"] = float(sm.ilisi_knn(nrb, tech[bi]))
    kb = sm.kbet(nrb, tech[bi])
    out["kBET"] = float(kb[0] if isinstance(kb, tuple) else kb)

    # --- cell-type based scIB (bio conservation + integration), random subsample ---
    si = _sub_random(len(Z), 6000)
    Zs, ts, cs = Z[si], tech[si], ct[si]
    vc = pd.Series(cs).value_counts()
    keep = set(vc[vc >= 20].index)
    m = np.array([c in keep for c in cs])
    Zs, ts, cs = Zs[m], ts[m], cs[m]
    nrs = _nn(Zs)
    out["sil_batch"] = float(sm.silhouette_batch(Zs, cs, ts))
    out["graph_conn"] = float(sm.graph_connectivity(nrs, cs))
    na = sm.nmi_ari_cluster_labels_kmeans(Zs, cs)
    out["NMI"] = float(na["nmi"])
    out["ARI"] = float(na["ari"])
    out["sil_label"] = float(sm.silhouette_label(Zs, cs))
    out["cLISI"] = float(sm.clisi_knn(nrs, cs))

    # --- headline predictability axes ---
    out["BioPred"] = _knn_f1(Z, dis)
    bp = _knn_f1(Z, tech)
    out["BatchMix"] = 1.0 - bp if bp == bp else np.nan
    return out


def run():
    meta, samples, M = load_gf()
    D = cosine_distance_matrix(M)
    sets = define_sets(meta)
    print("cohort adeno(T2):", len(sets[0]), "| cohort normal(T1):", len(sets[1]),
          "| pool adeno:", len(sets[2]), "| pool normal:", len(sets[3]))

    jobs = [("PURE", 0, 0, 1.0)]
    for k in KS:
        jobs.append(("RETRIEVAL", k, 0, 1.0))
        for qq in (0.5, 0.75):
            jobs.append(("QUANTILE", k, 0, qq))
        for s in RANDOM_SEEDS:
            jobs.append(("RANDOM", k, s, 1.0))

    rows = []
    for mode, k, seed, q in jobs:
        sids = plan(mode, k, seed, sets, samples, D, q=q)
        a = build_raw(sids)
        Z, tech, dis, ct = harmony_pca(a)
        m = metrics(Z, tech, dis, ct)
        row = {"mode": mode, "k": k, "seed": seed, "q": q,
               "n_samples": len(sids), "n_cells": a.n_obs, **m}
        rows.append(row)
        pd.DataFrame([row]).to_csv(
            CSV, mode="a", header=not os.path.exists(CSV), index=False)
        print(f"{mode:9} k={k} seed={seed} q={q} | Bio={m['BioPred']:.3f} "
              f"BatchMix={m['BatchMix']:.3f} NMI={m['NMI']:.3f} "
              f"graph={m['graph_conn']:.3f} sil_b={m['sil_batch']:.3f}", flush=True)
        del a, Z
        gc.collect()

    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()
