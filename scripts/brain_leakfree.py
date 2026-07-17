"""Leak-free representation for the brain replication: EVERYTHING that defines the space is fit
on POOL cells only; held-study cells are merely projected through it.

Two leaks are closed here, and they are different:

  1. TRANSDUCTIVE LEAK. The global build (brain_build/brain_landmarks) fit PCA, sigma and the
     landmark sketch on all 760k cells -- held-study cells included. No label was used, so it is
     not label leakage, but the geometry still adapted to labs we then claim to transfer to
     "unseen". Our whole claim is about transfer to an unseen study, so the space must not have
     seen it. (CRC hit this too: scripts/crc_leakfree_build.py.)

  2. REGION LEAK. That build also fit PCA on every region, including substantia nigra, cerebellum
     and thalamus -- 145k cells we then dropped. Midbrain-vs-cortex is the single fattest axis of
     variation in brain, so a chunk of the 50 components was spent resolving anatomy that is not
     in the analysis. Fitting on the analysis region only buys back that resolution.

Standardisation stats (mean/std), PCA basis, sigma and landmarks: all pool-only.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from geosketch import gs
from sklearn.decomposition import IncrementalPCA

COUNTS = "data/brain/counts_hv.npz"
CELLS = "data/brain/cells.parquet"
CTYPE = "data/brain/cell_types.parquet"
OUTSIG = "data/brain/signatures_lf_pfc.parquet"
OUTL = "data/brain/landmarks_lf_pfc.parquet"
CTCOL = "cell_type_coarse_brain"

# prefrontal family: DLPFC and its aliases across labs (Brodmann 9/46 IS dorsolateral prefrontal)
PFC = ["dorsolateral prefrontal cortex", "prefrontal cortex", "frontal cortex",
       "Brodmann (1909) area 9", "Brodmann (1909) area 46"]
HELD = ["37a17b78", "6f7fd0f1", "5e57cd50"]

NCOMP = 50
FLOOR = 15
BUDGET = 100
MIN_TYPE_CELLS = 500
CHUNK = 20000
TARGET = 1e4
CLIP = 10.0
SEED = 0


def norm_rows(X, rows, mean, std):
    """CPM -> log1p -> standardize with POOL stats -> clip. Dense only per chunk."""
    z = X[rows].toarray().astype(np.float32)
    s = z.sum(1, keepdims=True)
    s[s == 0] = 1
    z = np.log1p(z / s * TARGET)
    z -= mean
    z /= std
    np.clip(z, -CLIP, CLIP, out=z)
    return z


def main():
    cells = pd.read_parquet(CELLS).reset_index(drop=True)
    ct = pd.read_parquet(CTYPE).reset_index(drop=True)
    assert (cells.soma_joinid.to_numpy() == ct.soma_joinid.to_numpy()).all(), "joinid drift"
    X = sp.load_npz(COUNTS).tocsr()

    dom = ct.groupby("pid", observed=True).tissue.agg(lambda s: s.value_counts().index[0])
    pfc_pid = set(dom[dom.isin(PFC)].index)
    keep = cells.pid.isin(pfc_pid).to_numpy()
    cells, ct = cells[keep].reset_index(drop=True), ct[keep].reset_index(drop=True)
    X = X[np.where(keep)[0]]

    is_pool = (~cells.study.isin(HELD)).to_numpy()
    pool_idx = np.where(is_pool)[0]
    print(f"PFC cells={len(cells):,} donors={cells.pid.nunique()} studies={cells.study.nunique()}")
    print(f"pool cells={len(pool_idx):,} ({cells[is_pool].pid.nunique()} donors) "
          f"| held studies={HELD}", flush=True)

    # --- standardisation stats from POOL cells only ---
    n = 0
    mean = np.zeros(X.shape[1], np.float64)
    m2 = np.zeros(X.shape[1], np.float64)
    for a in range(0, len(pool_idx), CHUNK):
        z = X[pool_idx[a:a + CHUNK]].toarray().astype(np.float32)
        s = z.sum(1, keepdims=True)
        s[s == 0] = 1
        z = np.log1p(z / s * TARGET)
        n += len(z)
        mean += z.sum(0)
        m2 += (z ** 2).sum(0)
    mean /= n
    std = np.sqrt(np.maximum(m2 / n - mean ** 2, 1e-8)).astype(np.float32)
    mean = mean.astype(np.float32)
    print(f"pool standardisation stats from {n:,} cells", flush=True)

    # --- PCA fit on POOL cells only ---
    ipca = IncrementalPCA(n_components=NCOMP, batch_size=CHUNK)
    for a in range(0, len(pool_idx), CHUNK):
        ipca.partial_fit(norm_rows(X, pool_idx[a:a + CHUNK], mean, std))
    print(f"pca fit on pool | var explained={ipca.explained_variance_ratio_.sum():.3f}", flush=True)

    # --- project EVERY cell (pool + held) through the pool basis ---
    Z = np.empty((X.shape[0], NCOMP), np.float32)
    allr = np.arange(X.shape[0])
    for a in range(0, len(allr), CHUNK):
        rows = allr[a:a + CHUNK]
        Z[rows] = ipca.transform(norm_rows(X, rows, mean, std))
    print(f"projected {len(Z):,} cells", flush=True)

    # --- sigma + landmarks: POOL cells only ---
    rng = np.random.default_rng(SEED)
    Zp = Z[pool_idx]
    sub = Zp[rng.choice(len(Zp), min(5000, len(Zp)), replace=False)]
    d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
    sigma = float(np.median(d[d > 0]))
    denom = 2.0 * sigma * sigma
    print(f"sigma (pool-only)={sigma:.3f}", flush=True)

    ctype_pool = ct[CTCOL].to_numpy()[pool_idx]
    counts = pd.Series(ctype_pool).value_counts()
    parts = []
    print("landmarks (sketched from POOL cells only):", flush=True)
    for t, cnt in counts.items():
        if cnt < MIN_TYPE_CELLS:
            print(f"  {t:18s} cells={cnt:6d} -> dropped", flush=True)
            continue
        n_t = min(FLOOR + int(round(BUDGET * cnt / len(ctype_pool))), cnt)
        loc = np.where(ctype_pool == t)[0]                 # index into pool_idx
        sel = loc if len(loc) <= n_t else loc[np.array(gs(Zp[loc], n_t, replace=False))]
        parts.append(pool_idx[sel])                        # back to global row ids
        print(f"  {t:18s} cells={cnt:6d} -> {len(sel):3d} landmarks", flush=True)
    li = np.sort(np.concatenate(parts))
    L = Z[li]
    Ln = (L * L).sum(1)
    print(f"total landmarks: {len(li)}", flush=True)
    pd.DataFrame({"cell_row": li, CTCOL: ct[CTCOL].to_numpy()[li]}).to_parquet(OUTL)

    # --- Nystrom signatures for ALL PFC donors (held ones are projections) ---
    pid = cells.pid.to_numpy()
    order = pd.unique(pid)
    prow = pd.Series(np.arange(len(order)), index=order).loc[pid].to_numpy()
    sums = np.zeros((len(order), len(li)), np.float64)
    cnt = np.zeros(len(order), np.float64)
    for a in range(0, len(Z), CHUNK):
        zc = Z[a:a + CHUNK]
        d2 = (zc * zc).sum(1)[:, None] + Ln[None, :] - 2.0 * zc @ L.T
        np.add.at(sums, prow[a:a + CHUNK], np.exp(-np.maximum(d2, 0) / denom))
        np.add.at(cnt, prow[a:a + CHUNK], 1.0)
    emb = (sums / cnt[:, None]).astype(np.float32)

    df = pd.DataFrame(emb, columns=[f"s{j}" for j in range(len(li))])
    df.insert(0, "pid", order)
    meta = (cells.groupby("pid", observed=True)
                 .agg(study=("study", "first"), y=("y", "first"), n_cells=("pid", "size"))
                 .reset_index())
    reg = (ct.groupby("pid", observed=True).tissue
             .agg(lambda s: s.value_counts().index[0]).rename("region").reset_index())
    df = meta.merge(reg, on="pid").merge(df, on="pid")
    df["is_held"] = df.study.isin(HELD)
    df.to_parquet(OUTSIG)
    print(f"\nsaved {OUTSIG}: {df.shape}")
    print(f"donors={len(df)} disease={int(df.y.sum())} control={int((1 - df.y).sum())} "
          f"| held={int(df.is_held.sum())} pool={int((~df.is_held).sum())}")


if __name__ == "__main__":
    main()
