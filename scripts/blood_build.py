"""Build the blood/COVID replication corpus: raw counts -> our own PCA, same pipeline as CRC/brain.

Design (from the per-assay scan):
  tissue   = blood
  assays   = 5' FAMILY (10x 5' v1 / v2 / 5' transcription profiling) -- COVID lives on 5'; the big
             healthy reservoirs on 5' v2. Mixing in 3' would confound disease with technology
             (COVID is 5'-native, healthy atlases are 3'-heavy), so we stay inside 5'.
  y        = COVID-19 vs normal    (composition-shifting disease -> biology should be readable,
             unlike dementia; that is the point of this pivot -- [[brain-boundary-result]])
  patient  = donor = a distribution of cells

Batch = lab/protocol + the minor v1/v2 sub-chemistry (both classes appear on both -> a mild batch
the lever should navigate, not a hard confound). Held studies are all on 5' v1, so within-held the
assay is fixed and the ceiling is clean.

Memory-safe (same discipline that stopped the machine swapping on brain): Arrow-only plan read,
small SOMA buffers, contiguous per-donor blocks for HVG, narrow-column chunked counts pull,
IncrementalPCA. RUN UNDER A CGROUP CAP:
  systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 python scripts/blood_build.py
Resumable: each stage checks its output file first.
"""
import os
import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import IncrementalPCA
import cellxgene_census as cc

CENSUS_VERSION = "2025-11-08"
OUTDIR = "data/blood"
ARMS = "data/blood_arms_all.parquet"          # cached donor-arms, all assays (from the scan)
ASSAYS = ["10x 5' v1", "10x 5' v2", "10x 5' transcription profiling"]
MIN_CELLS, MAX_CELLS = 200, 400
MAX_DONORS_PER_STUDY = 200                     # one big 5' v2 healthy atlas must not outvote the rest
N_HVG = 2000
HVG_DONORS_PER_STUDY = 4
HVG_CELLS_PER_DONOR = 60
CHUNK_CELLS = 60_000
N_PC = 50
SEED = 0

F_PLAN = f"{OUTDIR}/plan.parquet"
F_HVG = f"{OUTDIR}/hv_genes.csv"
F_CNT = f"{OUTDIR}/counts_hv.npz"
F_CELLS = f"{OUTDIR}/cells.parquet"
F_PCA = f"{OUTDIR}/pca50.npy"


def is_covid(d):
    return "covid" in str(d).lower()


def ctx():
    b = 64 * 1024 ** 2
    return cc.get_default_soma_context().replace(
        tiledb_config={"soma.init_buffer_bytes": b, "py.init_buffer_bytes": b})


def build_plan():
    if os.path.exists(F_PLAN):
        return pd.read_parquet(F_PLAN)
    arms = pd.read_parquet(ARMS)
    arms = arms[arms.assay.isin(ASSAYS) & (arms.n >= MIN_CELLS)].copy()
    arms = arms[arms.disease.astype(str).str.lower().map(is_covid) | arms.disease.eq("normal")]
    arms["y"] = arms.disease.astype(str).str.lower().map(is_covid).astype(int)
    arms["study"] = arms.dataset_id.str[:8]
    rng = np.random.default_rng(SEED)

    keep = []                                  # cap donors/study, keep the study's class balance
    for st, g in arms.groupby("study"):
        if len(g) <= MAX_DONORS_PER_STUDY:
            keep.append(g); continue
        frac = MAX_DONORS_PER_STUDY / len(g)
        keep.append(g.groupby("y", group_keys=False).apply(
            lambda h: h.sample(max(1, int(round(len(h) * frac))), random_state=SEED)))
    arms = pd.concat(keep).reset_index(drop=True)
    print(f"donors after cap: {len(arms)} in {arms.study.nunique()} studies "
          f"({int(arms.y.sum())} COVID / {int((1 - arms.y).sum())} normal)", flush=True)

    with cc.open_soma(census_version=CENSUS_VERSION, context=ctx()) as census:
        obs = census["census_data"]["homo_sapiens"].obs
        alist = ", ".join(f'"{a}"' for a in ASSAYS)
        vf = f'is_primary_data == True and tissue_general == "blood" and assay in [{alist}]'
        cols = ["soma_joinid", "dataset_id", "donor_id"]
        o = pd.concat([t.to_pandas() for t in obs.read(column_names=cols, value_filter=vf)])
    print(f"blood 5'-family cells in census: {len(o):,}", flush=True)

    want = set(zip(arms.dataset_id, arms.donor_id))
    o = o[[k in want for k in zip(o.dataset_id, o.donor_id)]]
    rows = []
    for (ds, dn), g in o.groupby(["dataset_id", "donor_id"]):
        j = np.sort(g.soma_joinid.to_numpy(np.int64))
        if len(j) > MAX_CELLS:
            s = int(rng.integers(0, len(j) - MAX_CELLS + 1))
            j = j[s:s + MAX_CELLS]
        rows.append(pd.DataFrame({"soma_joinid": j, "dataset_id": ds, "donor_id": dn}))
    plan = pd.concat(rows, ignore_index=True).merge(
        arms[["dataset_id", "donor_id", "disease", "assay", "study", "y"]],
        on=["dataset_id", "donor_id"])
    # a few donors are sequenced on BOTH v1 and v2 -> arms has 2 rows for them -> the merge
    # duplicated their cells (same joinid twice). pid ignores assay, so keep one copy per cell.
    plan = plan.drop_duplicates("soma_joinid").reset_index(drop=True)
    plan["pid"] = plan.dataset_id.str[:8] + "_" + plan.donor_id.astype(str)
    os.makedirs(OUTDIR, exist_ok=True)
    plan.to_parquet(F_PLAN)
    print(f"plan: {len(plan):,} cells / {plan.pid.nunique()} donors", flush=True)
    return plan


def pick_hvg(plan):
    """HV genes from a study-balanced subsample. No batch_key (one loess/study = slow+noisy);
    CONTIGUOUS per-donor blocks (scattered rows across the array make TileDB decompress whole
    tiles -> 100% CPU stalls, learned on brain)."""
    if os.path.exists(F_HVG):
        return pd.read_csv(F_HVG)
    donors = (plan.drop_duplicates("pid").groupby("study", group_keys=False)
                  .apply(lambda g: g.sample(min(HVG_DONORS_PER_STUDY, len(g)), random_state=SEED)))
    sub = plan[plan.pid.isin(donors.pid)].groupby("pid", group_keys=False).head(HVG_CELLS_PER_DONOR)
    j = np.sort(sub.soma_joinid.to_numpy(np.int64))
    print(f"HVG subsample: {len(j):,} cells | {sub.pid.nunique()} donors | "
          f"{sub.study.nunique()} studies", flush=True)
    t = time.time()
    with cc.open_soma(census_version=CENSUS_VERSION, context=ctx()) as census:
        ad = cc.get_anndata(census, "Homo sapiens", X_name="raw", obs_coords=j.tolist(),
                            obs_column_names=["soma_joinid", "dataset_id"])
    print(f"  [t] census read (all genes): {time.time() - t:.0f}s -> {ad.shape}", flush=True)
    ad.var_names_make_unique()
    sc.pp.filter_genes(ad, min_cells=10)
    t = time.time()
    sc.pp.highly_variable_genes(ad, n_top_genes=N_HVG, flavor="seurat_v3")
    print(f"  [t] highly_variable_genes: {time.time() - t:.0f}s", flush=True)
    v = ad.var[ad.var.highly_variable]
    hv = pd.DataFrame({"soma_joinid": v.soma_joinid.to_numpy().astype(int),
                       "feature_name": v.feature_name.astype(str).to_numpy()})
    hv.to_csv(F_HVG, index=False)
    print(f"HV genes: {len(hv)}", flush=True)
    return hv


def pull_counts(plan, hv):
    if os.path.exists(F_CNT):
        return sp.load_npz(F_CNT), pd.read_parquet(F_CELLS)
    var = np.sort(hv.soma_joinid.to_numpy(np.int64))
    plan = plan.sort_values("soma_joinid").reset_index(drop=True)
    j = plan.soma_joinid.to_numpy(np.int64)
    blocks = []
    with cc.open_soma(census_version=CENSUS_VERSION, context=ctx()) as census:
        for a in range(0, len(j), CHUNK_CELLS):
            b = min(a + CHUNK_CELLS, len(j))
            ad = cc.get_anndata(census, "Homo sapiens", X_name="raw",
                                obs_coords=j[a:b].tolist(), var_coords=var.tolist(),
                                obs_column_names=["soma_joinid"])
            # robust to Census row order: reorder returned rows to match the requested joinids
            ret = ad.obs.soma_joinid.to_numpy()
            assert len(ret) == b - a, f"census returned {len(ret)} rows for {b - a} requested"
            order = pd.Index(ret).get_indexer(j[a:b])
            assert (order >= 0).all(), "requested joinid missing from census return"
            blocks.append(sp.csr_matrix(ad.X)[order])
            print(f"  counts {b:,}/{len(j):,}", flush=True)
    X = sp.vstack(blocks).tocsr()
    os.makedirs(OUTDIR, exist_ok=True)
    sp.save_npz(F_CNT, X)
    plan.to_parquet(F_CELLS)
    print(f"counts: {X.shape}, nnz={X.nnz:,}", flush=True)
    return X, plan


def pca(X):
    if os.path.exists(F_PCA):
        return np.load(F_PCA)
    ipca = IncrementalPCA(n_components=N_PC, batch_size=20000)

    def norm(blk):
        d = np.asarray(blk.todense(), dtype=np.float32)
        s = d.sum(1, keepdims=True); s[s == 0] = 1
        return np.log1p(d / s * 1e4)
    for a in range(0, X.shape[0], 20000):
        ipca.partial_fit(norm(X[a:a + 20000]))
        print(f"  pca fit {min(a + 20000, X.shape[0]):,}/{X.shape[0]:,}", flush=True)
    out = np.empty((X.shape[0], N_PC), np.float32)
    for a in range(0, X.shape[0], 20000):
        out[a:a + 20000] = ipca.transform(norm(X[a:a + 20000]))
    np.save(F_PCA, out)
    print(f"pca50: {out.shape} | var explained={ipca.explained_variance_ratio_.sum():.3f}", flush=True)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plan = build_plan()
    hv = pick_hvg(plan)
    X, cells = pull_counts(plan, hv)
    Z = pca(X)
    print(f"\nDONE. cells={Z.shape[0]:,} donors={cells.pid.nunique()} studies={cells.study.nunique()}")
    d = cells.drop_duplicates('pid')
    print(f"COVID={int(d.y.sum())} normal={int((1 - d.y).sum())}")
    print(d.groupby('study').y.agg(['sum', 'size']).to_string())


if __name__ == "__main__":
    main()
