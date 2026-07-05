import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

H5 = "4a8b9568-965e-46b8-a427-baab6bf018e5.h5ad"
SLIM = "data/crc_obs_slim.parquet"
OUTX = "data/crc/counts_hv.npz"
OUTCELLS = "data/crc/cells.parquet"
OUTGENES = "data/crc/hv_genes.csv"
MAX_CELLS = 500
MIN_CELLS = 200
SEED = 0
CHUNK = 10_000


def main():
    import os
    os.makedirs("data/crc", exist_ok=True)
    d = pd.read_parquet(SLIM)
    d["row"] = np.arange(len(d))
    d = d[(d.is_primary_data == True) & (d.sample_type.isin(["tumor", "normal"]))]
    keep = d.groupby("sample_id")["row"].transform("size") >= MIN_CELLS
    d = d[keep]

    rng = np.random.default_rng(SEED)
    picks = []
    for sid, g in d.groupby("sample_id"):
        r = g["row"].to_numpy()
        if len(r) > MAX_CELLS:
            r = rng.choice(r, MAX_CELLS, replace=False)
        picks.append(g[g["row"].isin(r)])
    sel = pd.concat(picks).sort_values("row").reset_index(drop=True)
    sel = sel.rename(columns={"study_id": "study", "donor_id": "donor"})
    want = sel["row"].to_numpy()
    print(f"selected {len(sel)} cells from {sel.sample_id.nunique()} samples", flush=True)

    f = h5py.File(H5, "r")
    hv = f["var"]["highly_variable"]
    hv = hv[:] if not isinstance(hv, h5py.Group) else hv["codes"][:].astype(bool)
    hv = hv.astype(bool)
    gene_ids = f["var"]["_index"][:].astype(str)
    pd.DataFrame({"gene": gene_ids[hv]}).to_csv(OUTGENES, index=False)
    print(f"HV genes: {hv.sum()}", flush=True)

    X = f["raw"]["X"]
    data, indices = X["data"], X["indices"]
    indptr = X["indptr"][:]
    ncells = len(indptr) - 1
    hv_idx = np.where(hv)[0]
    mask = np.zeros(ncells, dtype=bool)
    mask[want] = True

    blocks = []
    kept = 0
    nextmark = 50_000
    for a in range(0, ncells, CHUNK):
        b = min(a + CHUNK, ncells)
        local = np.nonzero(mask[a:b])[0]
        if local.size == 0:
            continue
        lo, hi = int(indptr[a]), int(indptr[b])
        m = sp.csr_matrix((data[lo:hi], indices[lo:hi], indptr[a:b + 1] - lo),
                          shape=(b - a, len(hv)))
        blocks.append(m[local][:, hv_idx])
        del m
        kept += int(local.size)
        if kept >= nextmark:
            print(f"  ...{kept}/{len(want)} cells", flush=True)
            nextmark += 50_000
    f.close()

    mat = sp.vstack(blocks).tocsr()
    sp.save_npz(OUTX, mat)
    sel[["sample_id", "sample_type", "study", "assay", "donor", "row"]].to_csv(
        OUTCELLS.replace(".parquet", ".csv"), index=False)
    sel[["sample_id", "sample_type", "study", "assay", "donor"]].to_parquet(OUTCELLS)
    print(f"saved {OUTX} shape={mat.shape} and {OUTCELLS}", flush=True)


if __name__ == "__main__":
    main()
