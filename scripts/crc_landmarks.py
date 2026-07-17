"""Phase 1: build landmark (Nystrom) sample embeddings, to compare with RFF signatures.
Landmarks = REAL cells chosen by STRATIFIED geometric sketching: geosketch (Hie 2019)
WITHIN each atlas cell type, with a floor + proportional allocation. Plain geosketch over
all cells follows volume/variance and over-picks batch-dominated compartments (Stromal),
starving abundant biological types (T cell) -- unfair for per-type witness. Stratifying
guarantees every type is resolved. Kernel = RBF with the SAME sigma (median heuristic,
SEED=0) as crc_signatures, so RFF and landmark bases approximate the SAME kernel.
  emb_j(P) = mean_{cell i in P} exp(-||x_i - landmark_j||^2 / (2 sigma^2))
Output columns s0..s{L-1} so all crc_* scripts work via CRC_SIG=signatures_landmark.parquet.
"""
import numpy as np
import pandas as pd
from geosketch import gs

PCA = "data/crc/pca50.npy"
CELLS = "data/crc/cells.parquet"
CTYPE = "data/crc/cell_types.parquet"
SIGREF = "data/crc/signatures.parquet"
OUT = "data/crc/signatures_landmark.parquet"
OUTL = "data/crc/landmarks.parquet"
CTCOL = "cell_type_coarse_crc_atlas"
FLOOR = 15          # landmarks guaranteed per cell type
BUDGET = 100        # extra landmarks distributed proportionally to type abundance
SEED = 0
CHUNK = 20000


def median_sigma(X, rng, m=5000):
    idx = rng.choice(len(X), min(m, len(X)), replace=False)
    S = X[idx]
    d = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    return float(np.median(d[d > 0]))


def main():
    X = np.load(PCA).astype(np.float32)
    cells = pd.read_parquet(CELLS).reset_index(drop=True)
    ct = pd.read_parquet(CTYPE).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    sigma = median_sigma(X, rng)                     # same as crc_signatures
    denom = 2.0 * sigma * sigma
    print(f"sigma={sigma:.3f} (matches RFF)", flush=True)

    # STRATIFIED sketching: geosketch within each cell type, floor + proportional
    ctype = ct[CTCOL].to_numpy()
    total = len(ctype)
    counts = pd.Series(ctype).value_counts()
    parts = []
    print("stratified landmark allocation per type:", flush=True)
    for t, cnt in counts.items():
        n_t = FLOOR + int(round(BUDGET * cnt / total))
        idx_t = np.where(ctype == t)[0]
        n_t = min(n_t, len(idx_t))
        sel = idx_t if len(idx_t) <= n_t else idx_t[np.array(gs(X[idx_t], n_t, replace=False))]
        parts.append(sel)
        print(f"  {t:16s} cells={cnt:7d} -> {len(sel):3d} landmarks", flush=True)
    li = np.sort(np.concatenate(parts))
    L = X[li]
    ltype = ctype[li]
    print(f"total landmarks: {len(li)}", flush=True)
    pd.DataFrame({"cell_row": li, CTCOL: ltype}).to_parquet(OUTL)

    # per-sample landmark signature: mean kernel over the sample's cells
    sid = cells.sample_id.to_numpy()
    order = pd.unique(sid)
    row_of = {s: i for i, s in enumerate(order)}
    srow = np.array([row_of[s] for s in sid])
    nS, nL = len(order), len(li)
    sums = np.zeros((nS, nL), np.float64); cnt = np.zeros(nS, np.float64)
    Ln = (L * L).sum(1)
    for a in range(0, len(X), CHUNK):
        b = min(a + CHUNK, len(X))
        xc = X[a:b]
        d2 = (xc * xc).sum(1)[:, None] + Ln[None, :] - 2.0 * xc @ L.T
        K = np.exp(-np.maximum(d2, 0) / denom)
        np.add.at(sums, srow[a:b], K)
        np.add.at(cnt, srow[a:b], 1.0)
    emb = (sums / cnt[:, None]).astype(np.float32)

    df = pd.DataFrame(emb, columns=[f"s{j}" for j in range(nL)])
    df.insert(0, "sample_id", order)
    ref = pd.read_parquet(SIGREF)[["sample_id", "sample_type", "study", "assay", "donor", "n_cells"]]
    df = ref.merge(df, on="sample_id", how="inner")
    df.to_parquet(OUT)
    print(f"saved {OUT}: {df.shape} | tumor/normal={dict(df.sample_type.value_counts())}", flush=True)
    print(f"saved {OUTL}: {len(li)} landmarks with cell types", flush=True)


if __name__ == "__main__":
    main()
