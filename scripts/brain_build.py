"""Build the brain replication corpus: raw counts -> our own PCA, exactly like the CRC pipeline.

Why raw counts and not the Census scVI latent: scVI is trained with dataset_id as a batch
covariate, i.e. it subtracts the very axis this project studies. Our own PCA leaves it intact,
which is the only way the brain numbers are comparable to CRC.

Design (frozen from scripts/brain_design.py):
  tissue   = brain
  assay    = 10x 3' v3 ONLY   -> batch is lab/protocol/year, NOT technology (the hard regime)
  y        = neurodegeneration (PD/AD/ALS/FTD/LBD/dementia) vs normal control
  patient  = donor = a distribution of cells

Caps, and why:
  MAX_DONORS_PER_STUDY: 37a17b78 alone has 955 donors; uncapped it would outvote the other ten
                        studies and the "between-study" axis would just be that one lab.
  MAX_CELLS: a patient is a distribution; 400 cells estimate it fine and keep the pull tractable.

Memory: never hold the full matrix in dense form; pull in donor chunks, keep CSR, IncrementalPCA.
Run under a cgroup cap (an unguarded earlier scan swapped the machine to death):
  systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 python scripts/brain_build.py
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
OUTDIR = "data/brain"
ARMS = "data/brain_arms.csv"                 # from scripts/brain_design.py
ASSAY = "10x 3' v3"
MIN_CELLS, MAX_CELLS = 200, 400
MAX_DONORS_PER_STUDY = 200
N_HVG = 2000
HVG_DONORS_PER_STUDY = 4      # donors sampled per study for the gene-selection read
HVG_CELLS_PER_DONOR = 60      # contiguous head of each donor's window (tile-friendly)
CHUNK_CELLS = 60_000                         # cells per Census X read
N_PC = 50
SEED = 0

F_PLAN = f"{OUTDIR}/plan.parquet"
F_HVG = f"{OUTDIR}/hv_genes.csv"
F_CNT = f"{OUTDIR}/counts_hv.npz"
F_CELLS = f"{OUTDIR}/cells.parquet"
F_PCA = f"{OUTDIR}/pca50.npy"

NEURODEG = ["alzheimer", "dementia", "parkinson", "amyotrophic", "lewy body",
            "supranuclear", "pick disease", "huntington"]
# narrow sensitivity axis: the dementia/proteinopathy family only (cortical, shared pathology)
DEMENTIA = ["alzheimer", "dementia", "pick disease", "supranuclear"]


def is_neurodeg(d):
    return any(k in str(d).lower() for k in NEURODEG)


def is_dementia(d):
    return any(k in str(d).lower() for k in DEMENTIA)


def ctx():
    b = 64 * 1024 ** 2
    return cc.get_default_soma_context().replace(
        tiledb_config={"soma.init_buffer_bytes": b, "py.init_buffer_bytes": b})


def build_plan():
    """donor -> the exact soma_joinids we will use. Capped, balanced, deterministic."""
    if os.path.exists(F_PLAN):
        return pd.read_parquet(F_PLAN)
    arms = pd.read_csv(ARMS)
    arms = arms[(arms.assay == ASSAY) & (arms.n >= MIN_CELLS)].copy()
    arms["study"] = arms.dataset_id.str[:8]
    rng = np.random.default_rng(SEED)

    # cap donors per study, keeping the disease/control balance of that study
    keep = []
    for st, g in arms.groupby("study"):
        if len(g) <= MAX_DONORS_PER_STUDY:
            keep.append(g)
            continue
        frac = MAX_DONORS_PER_STUDY / len(g)
        keep.append(g.groupby("y", group_keys=False).apply(
            lambda h: h.sample(max(1, int(round(len(h) * frac))), random_state=SEED)))
    arms = pd.concat(keep).reset_index(drop=True)
    print(f"donors after cap: {len(arms)} in {arms.study.nunique()} studies "
          f"({int(arms.y.sum())} neurodeg / {int((1 - arms.y).sum())} control)", flush=True)

    # per-donor cell joinids (contiguous window, like corpus_pull) -- one obs read per study
    with cc.open_soma(census_version=CENSUS_VERSION, context=ctx()) as census:
        obs = census["census_data"]["homo_sapiens"].obs
        vf = (f'is_primary_data == True and tissue_general == "brain" and assay == "{ASSAY}"')
        cols = ["soma_joinid", "dataset_id", "donor_id"]
        o = pd.concat([t.to_pandas() for t in obs.read(column_names=cols, value_filter=vf)])
    print(f"brain {ASSAY} cells in census: {len(o):,}", flush=True)

    want = set(zip(arms.dataset_id, arms.donor_id))
    o = o[[k in want for k in zip(o.dataset_id, o.donor_id)]]
    rows = []
    for (ds, dn), g in o.groupby(["dataset_id", "donor_id"]):
        j = np.sort(g.soma_joinid.to_numpy(np.int64))
        if len(j) > MAX_CELLS:                       # random contiguous window, same as corpus_pull
            s = int(rng.integers(0, len(j) - MAX_CELLS + 1))
            j = j[s:s + MAX_CELLS]
        rows.append(pd.DataFrame({"soma_joinid": j, "dataset_id": ds, "donor_id": dn}))
    plan = pd.concat(rows, ignore_index=True).merge(
        arms[["dataset_id", "donor_id", "disease", "study", "y"]], on=["dataset_id", "donor_id"])
    plan["pid"] = plan.dataset_id.str[:8] + "_" + plan.donor_id.astype(str)
    plan["y_dementia"] = plan.disease.map(is_dementia).astype(int)
    os.makedirs(OUTDIR, exist_ok=True)
    plan.to_parquet(F_PLAN)
    print(f"plan: {len(plan):,} cells / {plan.pid.nunique()} donors", flush=True)
    return plan


def pick_hvg(plan):
    """HV genes from a subsample balanced across ALL studies, so the gene set is not tuned to any
    single lab (which would bake a batch preference into the representation itself).

    Two things learned the hard way, both about cost rather than statistics:
      * no batch_key -- it runs one loess PER study (170 of them, most over ~100 nuclei): slow and
        noisy. Balance is already enforced by taking the same number of donors from every study.
      * sample CONTIGUOUS per-donor blocks, not scattered cells. This read spans all ~61k genes,
        and TileDB fetches whole tiles: scattered rows across a 21M-row array make it decompress
        enormous amounts of data to keep a few nuclei (100% CPU, zero network -- it stalled twice).
    """
    if os.path.exists(F_HVG):
        return pd.read_csv(F_HVG)
    rng = np.random.default_rng(SEED)
    donors = (plan.drop_duplicates("pid").groupby("study", group_keys=False)
                  .apply(lambda g: g.sample(min(HVG_DONORS_PER_STUDY, len(g)), random_state=SEED)))
    sub = plan[plan.pid.isin(donors.pid)]
    sub = sub.groupby("pid", group_keys=False).head(HVG_CELLS_PER_DONOR)   # contiguous window head
    j = np.sort(sub.soma_joinid.to_numpy(np.int64))
    print(f"HVG subsample: {len(j):,} cells | {sub.pid.nunique()} donors | "
          f"{sub.study.nunique()} studies", flush=True)

    t = time.time()
    with cc.open_soma(census_version=CENSUS_VERSION, context=ctx()) as census:
        ad = cc.get_anndata(census, "Homo sapiens", X_name="raw",
                            obs_coords=j.tolist(),
                            obs_column_names=["soma_joinid", "dataset_id"])
    print(f"  [t] census read (all genes): {time.time() - t:.0f}s -> {ad.shape}", flush=True)

    t = time.time()
    ad.var_names_make_unique()
    sc.pp.filter_genes(ad, min_cells=10)
    print(f"  [t] filter_genes: {time.time() - t:.0f}s -> {ad.shape}", flush=True)

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
            # Census returns rows in joinid order; assert so cells.parquet stays aligned
            assert np.array_equal(ad.obs.soma_joinid.to_numpy(), j[a:b]), "row order drift"
            blocks.append(sp.csr_matrix(ad.X))
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
    def norm(blk):                                    # CPM -> log1p, dense per block only
        d = np.asarray(blk.todense(), dtype=np.float32)
        s = d.sum(1, keepdims=True); s[s == 0] = 1
        return np.log1p(d / s * 1e4)
    for a in range(0, X.shape[0], 20000):             # fit
        ipca.partial_fit(norm(X[a:a + 20000]))
        print(f"  pca fit {min(a + 20000, X.shape[0]):,}/{X.shape[0]:,}", flush=True)
    out = np.empty((X.shape[0], N_PC), np.float32)
    for a in range(0, X.shape[0], 20000):             # transform
        out[a:a + 20000] = ipca.transform(norm(X[a:a + 20000]))
    np.save(F_PCA, out)
    print(f"pca50: {out.shape} | var explained={ipca.explained_variance_ratio_.sum():.3f}",
          flush=True)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plan = build_plan()
    hv = pick_hvg(plan)
    X, cells = pull_counts(plan, hv)
    Z = pca(X)
    print(f"\nDONE. cells={Z.shape[0]:,} donors={cells.pid.nunique()} "
          f"studies={cells.study.nunique()}")
    print(cells.drop_duplicates('pid').groupby('study').y.agg(['sum', 'size']).to_string())


if __name__ == "__main__":
    main()
