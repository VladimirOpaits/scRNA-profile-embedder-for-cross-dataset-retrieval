"""Leak-free representation for the blood/COVID replication: EVERYTHING that defines the space
(standardisation stats, PCA basis, sigma, landmarks) is fit on POOL cells only; held-study cells
are merely projected through it. Our claim is transfer to an UNSEEN study, so the space must not
have seen it (same discipline as brain_leakfree; CRC's scripts/crc_leakfree_build.py).

No region step (blood is one tissue). Held = the paired COVID studies on 5' v1 (both classes on
one assay -> clean within-study ceiling). Pool = every other 5'-family study (COVID from other labs
+ the big healthy 5' v2 atlases) -> the enrichment reservoir.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from geosketch import gs
from sklearn.decomposition import IncrementalPCA

COUNTS = "data/blood/counts_hv.npz"
CELLS = "data/blood/cells.parquet"
CTYPE = "data/blood/cell_types.parquet"
OUTSIG = "data/blood/signatures_lf.parquet"
OUTL = "data/blood/landmarks_lf.parquet"
CTCOL = "cell_type_coarse_blood"

# paired COVID/normal studies on 5' v1 (held candidates), from the design scan
HELD = ["2a498ace", "21d3e683", "30cd5311", "ebc2e1ff", "242c6e7f"]

NCOMP = 50
FLOOR = 15
BUDGET = 100
MIN_TYPE_CELLS = 500
CHUNK = 20000
TARGET = 1e4
CLIP = 10.0
SEED = 0


def norm_rows(X, rows, mean, std):
    z = X[rows].toarray().astype(np.float32)
    s = z.sum(1, keepdims=True); s[s == 0] = 1
    z = np.log1p(z / s * TARGET)
    z -= mean; z /= std
    np.clip(z, -CLIP, CLIP, out=z)
    return z


def main():
    cells = pd.read_parquet(CELLS).reset_index(drop=True)
    ct = pd.read_parquet(CTYPE).reset_index(drop=True)
    assert (cells.soma_joinid.to_numpy() == ct.soma_joinid.to_numpy()).all(), "joinid drift"
    X = sp.load_npz(COUNTS).tocsr()

    is_pool = (~cells.study.isin(HELD)).to_numpy()
    pool_idx = np.where(is_pool)[0]
    print(f"cells={len(cells):,} donors={cells.pid.nunique()} studies={cells.study.nunique()}")
    print(f"pool cells={len(pool_idx):,} ({cells[is_pool].pid.nunique()} donors) | held={HELD}",
          flush=True)

    # standardisation stats from POOL cells only
    n = 0; mean = np.zeros(X.shape[1]); m2 = np.zeros(X.shape[1])
    for a in range(0, len(pool_idx), CHUNK):
        z = X[pool_idx[a:a + CHUNK]].toarray().astype(np.float32)
        s = z.sum(1, keepdims=True); s[s == 0] = 1
        z = np.log1p(z / s * TARGET)
        n += len(z); mean += z.sum(0); m2 += (z ** 2).sum(0)
    mean /= n
    std = np.sqrt(np.maximum(m2 / n - mean ** 2, 1e-8)).astype(np.float32)
    mean = mean.astype(np.float32)

    ipca = IncrementalPCA(n_components=NCOMP, batch_size=CHUNK)
    for a in range(0, len(pool_idx), CHUNK):
        ipca.partial_fit(norm_rows(X, pool_idx[a:a + CHUNK], mean, std))
    print(f"pca fit on pool | var explained={ipca.explained_variance_ratio_.sum():.3f}", flush=True)

    Z = np.empty((X.shape[0], NCOMP), np.float32)
    allr = np.arange(X.shape[0])
    for a in range(0, len(allr), CHUNK):
        rows = allr[a:a + CHUNK]
        Z[rows] = ipca.transform(norm_rows(X, rows, mean, std))
    print(f"projected {len(Z):,} cells", flush=True)

    rng = np.random.default_rng(SEED)
    Zp = Z[pool_idx]
    sub = Zp[rng.choice(len(Zp), min(5000, len(Zp)), replace=False)]
    d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
    sigma = float(np.median(d[d > 0])); denom = 2.0 * sigma * sigma
    print(f"sigma (pool-only)={sigma:.3f}", flush=True)

    ctype_pool = ct[CTCOL].to_numpy()[pool_idx]
    counts = pd.Series(ctype_pool).value_counts()
    parts = []
    print("landmarks (sketched from POOL cells only):", flush=True)
    for t, cnt in counts.items():
        if cnt < MIN_TYPE_CELLS:
            print(f"  {t:14s} cells={cnt:6d} -> dropped", flush=True); continue
        n_t = min(FLOOR + int(round(BUDGET * cnt / len(ctype_pool))), cnt)
        loc = np.where(ctype_pool == t)[0]
        sel = loc if len(loc) <= n_t else loc[np.array(gs(Zp[loc], n_t, replace=False))]
        parts.append(pool_idx[sel])
        print(f"  {t:14s} cells={cnt:6d} -> {len(sel):3d} landmarks", flush=True)
    li = np.sort(np.concatenate(parts))
    L = Z[li]; Ln = (L * L).sum(1)
    print(f"total landmarks: {len(li)}", flush=True)
    pd.DataFrame({"cell_row": li, CTCOL: ct[CTCOL].to_numpy()[li]}).to_parquet(OUTL)

    pid = cells.pid.to_numpy()
    order = pd.unique(pid)
    prow = pd.Series(np.arange(len(order)), index=order).loc[pid].to_numpy()
    sums = np.zeros((len(order), len(li))); cnt = np.zeros(len(order))
    for a in range(0, len(Z), CHUNK):
        zc = Z[a:a + CHUNK]
        d2 = (zc * zc).sum(1)[:, None] + Ln[None, :] - 2.0 * zc @ L.T
        np.add.at(sums, prow[a:a + CHUNK], np.exp(-np.maximum(d2, 0) / denom))
        np.add.at(cnt, prow[a:a + CHUNK], 1.0)
    emb = (sums / cnt[:, None]).astype(np.float32)

    df = pd.DataFrame(emb, columns=[f"s{j}" for j in range(len(li))])
    df.insert(0, "pid", order)
    meta = (cells.groupby("pid", observed=True)
                 .agg(study=("study", "first"), y=("y", "first"),
                      assay=("assay", "first"), n_cells=("pid", "size")).reset_index())
    df = meta.merge(df, on="pid")
    df["is_held"] = df.study.isin(HELD)
    df.to_parquet(OUTSIG)
    print(f"\nsaved {OUTSIG}: {df.shape}")
    print(f"donors={len(df)} COVID={int(df.y.sum())} normal={int((1 - df.y).sum())} "
          f"| held={int(df.is_held.sum())} pool={int((~df.is_held).sum())}")


if __name__ == "__main__":
    main()
