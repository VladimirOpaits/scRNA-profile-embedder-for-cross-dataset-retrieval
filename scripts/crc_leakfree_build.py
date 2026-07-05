import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import IncrementalPCA

COUNTS = "data/crc/counts_hv.npz"
CELLS = "data/crc/cells.parquet"
OUTSIG = "data/crc/signatures_lf.parquet"
HELD = ["Chen_2024_Cancer_Cell", "Lee_2020_Nat_Genet", "Uhlitz_2021_EMBO_Mol_Med",
        "MUI_Innsbruck", "Zhang_2020_Cell"]
NCOMP = 50
D = 1024
CHUNK = 10_000
TARGET = 1e4
CLIP = 10.0
SEED = 0


def prep_chunk(X, rows, mean, std):
    z = X[rows].toarray()
    z -= mean
    z /= std
    np.clip(z, -CLIP, CLIP, out=z)
    return z


def main():
    cells = pd.read_parquet(CELLS).reset_index(drop=True)
    poolmask = ~cells.study.isin(HELD).to_numpy()
    pool_idx = np.where(poolmask)[0]
    n = len(cells)

    X = sp.load_npz(COUNTS).tocsr().astype(np.float32)
    rs = np.asarray(X.sum(1)).ravel().astype(np.float32)
    rs[rs == 0] = 1.0
    scale = (TARGET / rs).astype(np.float32)
    X.data *= np.repeat(scale, np.diff(X.indptr))
    X.data = np.log1p(X.data)
    G = X.shape[1]
    print(f"normalized in-place float32 | pool {len(pool_idx)}/{n}", flush=True)

    s1 = np.zeros(G, np.float64)
    s2 = np.zeros(G, np.float64)
    for a in range(0, len(pool_idx), CHUNK):
        z = X[pool_idx[a:a + CHUNK]].toarray()
        s1 += z.sum(0)
        s2 += (z.astype(np.float64) ** 2).sum(0)
    npool = len(pool_idx)
    mean = (s1 / npool).astype(np.float32)
    std = np.sqrt(np.maximum(s2 / npool - (s1 / npool) ** 2, 1e-8)).astype(np.float32)
    print("gene stats from pool done", flush=True)

    ipca = IncrementalPCA(n_components=NCOMP)
    for a in range(0, npool, CHUNK):
        ipca.partial_fit(prep_chunk(X, pool_idx[a:a + CHUNK], mean, std))
        print(f"  fit {min(a + CHUNK, npool)}/{npool}", flush=True)
    print(f"pca cumvar={ipca.explained_variance_ratio_.sum():.3f}", flush=True)

    P = np.empty((n, NCOMP), np.float32)
    allidx = np.arange(n)
    for a in range(0, n, CHUNK):
        r = allidx[a:a + CHUNK]
        P[r] = ipca.transform(prep_chunk(X, r, mean, std))
    del X
    print("projected all cells", flush=True)

    rng = np.random.default_rng(SEED)
    smp = rng.choice(pool_idx, min(5000, npool), replace=False)
    S = P[smp]
    dp = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    sigma = float(np.median(dp[dp > 0]))
    W = (rng.standard_normal((NCOMP, D)) / sigma).astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, D).astype(np.float32)
    phi = np.sqrt(2.0 / D) * np.cos(P @ W + b)
    print(f"pool-fit sigma={sigma:.3f}", flush=True)

    meta = cells.groupby("sample_id").first()
    rows = []
    for sid, idx in cells.groupby("sample_id").indices.items():
        m = meta.loc[sid]
        rows.append([sid, m.sample_type, m.study, m.assay, m.donor, len(idx)]
                    + phi[idx].mean(0).tolist())
    cols = ["sample_id", "sample_type", "study", "assay", "donor", "n_cells"] + \
        [f"s{j}" for j in range(D)]
    pd.DataFrame(rows, columns=cols).to_parquet(OUTSIG)
    print(f"saved {OUTSIG}", flush=True)


if __name__ == "__main__":
    main()
