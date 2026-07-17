"""Parallel, cached, multi-metric Harmony retrieval sweep. 2D axes: CORE x HELD.

  python scripts/harmony_sweep.py crc [--workers 12] [--seeds 5] [--ndem 4] [--nnor 4]
  python scripts/harmony_sweep.py blood
  python scripts/harmony_sweep.py brain

CORE = confounded base cohort we DE-CONFOUND: disease from study A + normal from study B (A!=B).
We sweep A x B (top --ndem disease studies x --nnor normal studies), so "which base cohort" is an
explicit axis alongside HELD (the unseen study we transfer to). References are PAIRS (1 disease +
1 normal from the SAME paired study) that decorrelate diagnosis from study; selection is anchored
on the reservoir, NEVER on held (no leakage). Strategies order the paired studies: coverage /
quantile / random.

Per job, on the Harmony-corrected embedding of core+refs+held cells, record:
  transfer_auc      patient-level disease transfer to held  (needs held: the confounded core cannot
                    self-test biology-vs-batch)
  batch_eta2        mean eta^2(study)     (LOWER = batch removed)          batch axis
  celltype_eta2     mean eta^2(cell_type) (HIGHER = cell biology kept)     bio axis
  batch_mix         kNN fraction of neighbors from a DIFFERENT study (HIGHER = mixed)   scIB-lite
  celltype_purity   kNN fraction of neighbors of the SAME cell_type (HIGHER = kept)     scIB-lite
plus confound_severity = cosine dist between A-disease centroid and B-normal centroid (how hard the
core's confound is). Over-correction (batch down, celltype down) is separable from good correction.

Speed: process pool, BLAS pinned to 1 thread/worker (else Harmony oversubscribes). Data loaded once
per worker via the pool initializer. Long-format parquet, resumable (done rows skipped).
"""
import argparse
import os

import numpy as np
import pandas as pd

RESULTS = "data/harmony_results.parquet"
KEY = ["tissue", "held", "dem_src", "normal_src", "strategy", "K", "seed"]
KS = [1, 2, 4, 6]
BLAS_THREADS = 2

CONFIG = {
    "brain": dict(sig="data/brain/signatures_lf_pfc.parquet", pca="data/brain/pca50.npy",
                  cells="data/brain/cells.parquet", ctype="data/brain/cell_types.parquet",
                  ctcol="cell_type_coarse_brain", pidcol="pid", ycol="y_from_sig",
                  held=["37a17b78", "6f7fd0f1", "5e57cd50", "3b8b5de4", "d3cb449b"]),
    "blood": dict(sig="data/blood/signatures_lf.parquet", pca="data/blood/pca50.npy",
                  cells="data/blood/cells.parquet", ctype="data/blood/cell_types.parquet",
                  ctcol="cell_type_coarse_blood", pidcol="pid", ycol="y",
                  held=["2a498ace", "21d3e683", "30cd5311", "ebc2e1ff", "242c6e7f"]),
    "crc": dict(sig="data/crc/signatures_lf.parquet", pca="data/crc/pca50.npy",
                cells="data/crc/cells.parquet", ctype="data/crc/cell_types.parquet",
                ctcol="cell_type_coarse_crc_atlas", pidcol="sample_id", ycol="tumor",
                held=["Chen_2024_Cancer_Cell", "Lee_2020_Nat_Genet", "Joanito_2022_Nat_Genet",
                      "Uhlitz_2021_EMBO_Mol_Med", "Zhang_2020_Cell"]),
}

G = {}


def _init(tissue):
    from threadpoolctl import threadpool_limits
    threadpool_limits(BLAS_THREADS)
    import harmonypy
    import warnings
    warnings.filterwarnings("ignore")
    cfg = CONFIG[tissue]

    sig = pd.read_parquet(cfg["sig"])
    if cfg["pidcol"] != "pid":
        sig = sig.rename(columns={cfg["pidcol"]: "pid"})
    Scol = [c for c in sig.columns if c.startswith("s") and c[1:].isdigit()]
    cells0 = pd.read_parquet(cfg["cells"]).reset_index(drop=True)
    if cfg["pidcol"] != "pid":
        cells0 = cells0.rename(columns={cfg["pidcol"]: "pid"})
    ctall = pd.read_parquet(cfg["ctype"]).reset_index(drop=True)[cfg["ctcol"]].to_numpy()

    if cfg["ycol"] == "tumor":
        keep = cells0.sample_type.isin(["tumor", "normal"]).to_numpy()
        y_all = (cells0.sample_type == "tumor").astype(int).to_numpy()
        sig = sig[sig.sample_type.isin(["tumor", "normal"])].reset_index(drop=True)
    elif cfg["ycol"] == "y_from_sig":
        keep = cells0.pid.isin(set(sig.pid)).to_numpy()
        y_all = cells0.pid.map(dict(zip(sig.pid, sig.y))).fillna(0).astype(int).to_numpy()
    else:
        keep = np.ones(len(cells0), bool)
        y_all = cells0.y.to_numpy()

    Z = np.load(cfg["pca"]).astype(np.float32)[keep]
    cells = cells0[keep].reset_index(drop=True)
    cells["y"] = y_all[keep]
    cells["ct"] = ctall[keep]
    don = cells.drop_duplicates("pid")[["pid", "study", "y"]].reset_index(drop=True)
    G.update(dict(harmonypy=harmonypy, Z=Z, cells=cells,
                  SIGARR=sig[Scol].to_numpy().astype(np.float32),
                  sidx={p: i for i, p in enumerate(sig.pid)},
                  study_of=dict(zip(don.pid, don.study)), don=don,
                  pid_rows={p: g.index.to_numpy() for p, g in cells.groupby("pid", observed=True)},
                  CPD=120))


def setup_core(H, DEM, NOR):
    don, r = G["don"], np.random.default_rng(0)
    demp = don.pid[(don.study == DEM) & (don.y == 1)].tolist()
    norp = don.pid[(don.study == NOR) & (don.y == 0)].tolist()
    core = (list(r.choice(demp, min(25, len(demp)), replace=False)) +
            list(r.choice(norp, min(25, len(norp)), replace=False)))
    return core, don.pid[don.study == H].tolist()


def confound_severity(DEM, NOR):
    A, ix, don = G["SIGARR"], G["sidx"], G["don"]
    a = A[[ix[p] for p in don.pid[(don.study == DEM) & (don.y == 1)]]].mean(0)
    b = A[[ix[p] for p in don.pid[(don.study == NOR) & (don.y == 0)]]].mean(0)
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def paired_studies(H, DEM, NOR):
    out = {}
    for s, g in G["don"][~G["don"].study.isin([H, DEM, NOR])].groupby("study"):
        dem = g.pid[g.y == 1].tolist()
        nor = g.pid[g.y == 0].tolist()
        if dem and nor:
            out[s] = (dem, nor)
    return out


def medoid(pids):
    A, ix = G["SIGARR"], G["sidx"]
    c = A[[ix[p] for p in pids]].mean(0)
    return min(pids, key=lambda p: np.linalg.norm(A[ix[p]] - c))


def order_studies(pstud, how, seed):
    A, ix = G["SIGARR"], G["sidx"]
    names = list(pstud)
    cents = {s: A[[ix[p] for p in dem + nor]].mean(0) for s, (dem, nor) in pstud.items()}
    gc = np.mean([cents[s] for s in names], 0)
    if how == "random":
        return list(np.random.default_rng(seed).permutation(names))
    if how == "coverage":
        ntrim = 1 if len(names) >= 4 else 0
        return sorted(names, key=lambda s: np.linalg.norm(cents[s] - gc))[:len(names) - ntrim]
    start = min(names, key=lambda s: np.linalg.norm(cents[s] - gc))
    order, rest = [start], [s for s in names if s != start]
    while rest:
        d = np.array([min(np.linalg.norm(cents[s] - cents[o]) for o in order) for s in rest])
        pick = rest[int(np.argmin(np.abs(d - np.quantile(d, 0.8))))]
        order.append(pick)
        rest.remove(pick)
    return order


def build_refs(pstud, ordered, K):
    refs = []
    for s in ordered[:K]:
        dem, nor = pstud[s]
        refs += [medoid(dem), medoid(nor)]
    return refs


def build_samebatch(H, DEM, NOR, core, K, seed):
    """control: K more disease from A + K more normal from B -> same two batches, more data,
    confound preserved. Isolates 'more data' from 'de-confounding'."""
    don, r = G["don"], np.random.default_rng(seed)
    cset = set(core)
    dem_extra = [p for p in don.pid[(don.study == DEM) & (don.y == 1)] if p not in cset]
    nor_extra = [p for p in don.pid[(don.study == NOR) & (don.y == 0)] if p not in cset]
    return (list(r.choice(dem_extra, min(K, len(dem_extra)), replace=False)) +
            list(r.choice(nor_extra, min(K, len(nor_extra)), replace=False)))


def _rows(pids):
    out = []
    for p in pids:
        idx = G["pid_rows"][p]
        out.append(idx if len(idx) <= G["CPD"] else
                   np.random.default_rng(0).choice(idx, G["CPD"], replace=False))
    return np.concatenate(out)


def _sigs(Zc, pid_arr, order, D=512):
    from scipy.spatial.distance import pdist
    Zc = np.asarray(Zc, np.float32)
    r = np.random.default_rng(0)
    s = Zc[r.choice(len(Zc), min(1500, len(Zc)), replace=False)]
    sigma = float(np.median(pdist(s)))
    W = (r.standard_normal((Zc.shape[1], D)) / sigma).astype(np.float32)
    b = r.uniform(0, 2 * np.pi, D).astype(np.float32)
    phi = np.sqrt(2 / D) * np.cos(Zc @ W + b)
    prow = pd.Series(np.arange(len(order)), index=order).loc[pid_arr].to_numpy()
    S = np.zeros((len(order), D), np.float32)
    c = np.zeros(len(order))
    np.add.at(S, prow, phi)
    np.add.at(c, prow, 1.0)
    return S / c[:, None]


def _eta2(E, labels):
    E = np.asarray(E, np.float32)
    gm = E.mean(0)
    sst = ((E - gm) ** 2).sum(0) + 1e-9
    ssb = np.zeros(E.shape[1])
    for lab in pd.unique(labels):
        Eg = E[labels == lab]
        ssb += len(Eg) * (Eg.mean(0) - gm) ** 2
    return float(np.mean(ssb / sst))


def _knn_scib(E, study, ct, k=15, cap=5000):
    from sklearn.neighbors import NearestNeighbors
    E = np.asarray(E, np.float32)
    rng = np.random.default_rng(0)
    q = rng.choice(len(E), min(cap, len(E)), replace=False)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
    idx = nn.kneighbors(E[q], return_distance=False)[:, 1:]
    smix = np.mean(study[idx] != study[q][:, None])
    cpur = np.mean(ct[idx] == ct[q][:, None])
    return float(smix), float(cpur)


def evaluate_full(core, refs, held_pid, do_harmony=True):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    pids = list(core) + list(refs) + list(held_pid)
    rows = _rows(pids)
    Zs = G["Z"][rows]
    meta = G["cells"].iloc[rows][["pid", "study", "y", "ct"]]
    if do_harmony:
        ho = G["harmonypy"].run_harmony(Zs, meta, ["study"], max_iter_harmony=10)
        Zc = np.asarray(ho.Z_corr, np.float32)
        Zs = Zc if Zc.shape[0] == len(meta) else Zc.T
    order = pd.unique(meta.pid.to_numpy())
    S = _sigs(Zs, meta.pid.to_numpy(), order)
    yb = meta.drop_duplicates("pid").set_index("pid").y
    ix = {p: i for i, p in enumerate(order)}
    tr = [ix[p] for p in list(core) + list(refs)]
    te = [ix[p] for p in held_pid]
    sc = StandardScaler().fit(S[tr])
    clf = LogisticRegression(C=0.01, max_iter=5000).fit(sc.transform(S[tr]), yb.loc[list(core) + list(refs)])
    auc = roc_auc_score(yb.loc[held_pid], clf.predict_proba(sc.transform(S[te]))[:, 1])
    smix, cpur = _knn_scib(Zs, meta.study.to_numpy(), meta.ct.to_numpy())
    return dict(transfer_auc=auc, batch_eta2=_eta2(Zs, meta.study.to_numpy()),
                celltype_eta2=_eta2(Zs, meta.ct.to_numpy()),
                batch_mix=smix, celltype_purity=cpur, n_cells=len(rows))


def _job(job):
    H, DEM, NOR, how, K, seed = job
    core, held_pid = setup_core(H, DEM, NOR)
    if how == "raw":
        m = evaluate_full(core, [], held_pid, do_harmony=False)
    elif how == "harmony_noref":
        m = evaluate_full(core, [], held_pid, do_harmony=True)
    elif how == "samebatch":
        m = evaluate_full(core, build_samebatch(H, DEM, NOR, core, K, seed), held_pid)
    else:
        pstud = paired_studies(H, DEM, NOR)
        m = evaluate_full(core, build_refs(pstud, order_studies(pstud, how, seed), K), held_pid)
    return dict(held=H, dem_src=DEM, normal_src=NOR,
                confound_severity=confound_severity(DEM, NOR),
                strategy=how, K=K, seed=seed, **m)


def build_jobs(tissue, seeds, ndem, nnor):
    _init(tissue)
    don = G["don"]
    cfg = CONFIG[tissue]
    jobs = []
    for H in cfg["held"]:
        dem = don[(don.y == 1) & (don.study != H)].study.value_counts()
        nor = don[(don.y == 0) & (don.study != H)].study.value_counts()
        dem = dem[dem >= 2].index[:ndem].tolist()
        nor = nor[nor >= 2].index[:nnor].tolist()
        for A in dem:
            for B in nor:
                if A == B:
                    continue
                jobs.append((H, A, B, "raw", 0, 0))
                jobs.append((H, A, B, "harmony_noref", 0, 0))
                for K in KS:
                    jobs.append((H, A, B, "coverage", K, 0))
                    jobs.append((H, A, B, "quantile", K, 0))
                    for s in range(seeds):
                        jobs.append((H, A, B, "random", K, s))
                        jobs.append((H, A, B, "samebatch", K, s))
    G.clear()
    return jobs


def _flush(rows):
    if not rows:
        return
    new = pd.DataFrame(rows)
    if os.path.exists(RESULTS):
        old = pd.read_parquet(RESULTS)
        old = old[~old.set_index(KEY).index.isin(new.set_index(KEY).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(RESULTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tissue", choices=list(CONFIG))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ndem", type=int, default=4)
    ap.add_argument("--nnor", type=int, default=4)
    args = ap.parse_args()

    jobs = build_jobs(args.tissue, args.seeds, args.ndem, args.nnor)
    done = set()
    if os.path.exists(RESULTS):
        p = pd.read_parquet(RESULTS)
        done = set(map(tuple, p[p.tissue == args.tissue][KEY[1:]].values))
    todo = [j for j in jobs if (j[0], j[1], j[2], j[3], j[4], j[5]) not in done]
    print(f"{args.tissue}: {len(jobs)} jobs, {len(todo)} to run, {args.workers} workers", flush=True)
    if not todo:
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows, n = [], 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.tissue,)) as ex:
        futs = [ex.submit(_job, j) for j in todo]
        for f in as_completed(futs):
            r = f.result()
            r["tissue"] = args.tissue
            rows.append(r)
            n += 1
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
            if n % 100 == 0:
                _flush(rows)
    _flush(rows)
    print(f"done {args.tissue}: {len(rows)} new rows -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
