"""Landmark (Nystrom) patient signatures for the brain corpus -- the same construction as
scripts/crc_landmarks.py, so brain and CRC numbers are comparable.

  emb_j(P) = mean_{cell i in P} exp(-||x_i - l_j||^2 / (2 sigma^2))     (L2 between these ~ MMD)

Landmarks are REAL cells chosen by geosketch STRATIFIED within each coarse cell type, with a floor
plus proportional allocation. Plain geosketch follows volume and over-picks whatever compartment
dominates the manifold (in brain that is oligodendrocytes), starving the rest -- which would make
the per-type witness and the leave-one-type-out ablation unfair by construction.

sigma = median heuristic (label-free, hence non-circular: we never tune it on the axes we then
interpret -- see the circularity trap flagged during the CRC ablation).

CORTEX ONLY. Region is confounded with disease by study design (Parkinson's -> substantia nigra,
Alzheimer's -> prefrontal cortex), and regions differ enormously in composition (midbrain is
~2/3 oligodendrocytes, cortex is ~half neurons). Left in, "disease vs control" would partly be
"midbrain vs cortex", and retrieval would import anatomy while we called it batch. Fixing the
region costs almost nothing here -- all 11 paired studies survive, 687 of 719 paired donors --
and it leaves the LAB as the only axis varying between studies, which is exactly the claim.
"""
import numpy as np
import pandas as pd
from geosketch import gs

PCA = "data/brain/pca50.npy"
CELLS = "data/brain/cells.parquet"
CTYPE = "data/brain/cell_types.parquet"
OUT = "data/brain/signatures_landmark.parquet"
OUTL = "data/brain/landmarks.parquet"
CTCOL = "cell_type_coarse_brain"
FLOOR = 15          # landmarks guaranteed per cell type
BUDGET = 100        # extra landmarks distributed proportionally to type abundance
MIN_TYPE_CELLS = 500   # a compartment too rare to sketch is dropped, not silently under-sampled
SEED = 0
CHUNK = 20000
# Brodmann areas are cortex (cytoarchitectonic fields OF the cortex: area 9 = DLPFC, area 4 = M1)
CORTEX = ["cortex", "cortical", "gyrus", "neocortex", "brodmann"]


def median_sigma(X, rng, m=5000):
    idx = rng.choice(len(X), min(m, len(X)), replace=False)
    S = X[idx]
    d = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    return float(np.median(d[d > 0]))


def main():
    X = np.load(PCA).astype(np.float32)
    cells = pd.read_parquet(CELLS).reset_index(drop=True)
    ct = pd.read_parquet(CTYPE).reset_index(drop=True)
    assert len(X) == len(cells) == len(ct), "row misalignment across pca/cells/cell_types"
    # cells.parquet and cell_types.parquet are both built from plan.parquet sorted by soma_joinid
    assert (cells.soma_joinid.to_numpy() == ct.soma_joinid.to_numpy()).all(), "joinid drift"

    # keep only donors whose dominant region is cortex (see module docstring)
    dom = ct.groupby("pid", observed=True).tissue.agg(lambda s: s.value_counts().index[0])
    keep_pid = set(dom[dom.str.lower().str.contains("|".join(CORTEX))].index)
    mask = cells.pid.isin(keep_pid).to_numpy()
    X, cells, ct = X[mask], cells[mask].reset_index(drop=True), ct[mask].reset_index(drop=True)
    print(f"cortex only: {len(X):,} cells | {cells.pid.nunique()} donors "
          f"| {cells.study.nunique()} studies", flush=True)

    rng = np.random.default_rng(SEED)
    sigma = median_sigma(X, rng)
    denom = 2.0 * sigma * sigma
    print(f"sigma={sigma:.3f}", flush=True)

    ctype = ct[CTCOL].to_numpy()
    total = len(ctype)
    counts = pd.Series(ctype).value_counts()
    parts = []
    print("stratified landmark allocation per type:", flush=True)
    for t, cnt in counts.items():
        if cnt < MIN_TYPE_CELLS:
            print(f"  {t:18s} cells={cnt:7d} -> dropped (< {MIN_TYPE_CELLS})", flush=True)
            continue
        n_t = FLOOR + int(round(BUDGET * cnt / total))
        idx_t = np.where(ctype == t)[0]
        n_t = min(n_t, len(idx_t))
        sel = idx_t if len(idx_t) <= n_t else idx_t[np.array(gs(X[idx_t], n_t, replace=False))]
        parts.append(sel)
        print(f"  {t:18s} cells={cnt:7d} -> {len(sel):3d} landmarks", flush=True)
    li = np.sort(np.concatenate(parts))
    L = X[li]
    print(f"total landmarks: {len(li)}", flush=True)
    pd.DataFrame({"cell_row": li, CTCOL: ctype[li]}).to_parquet(OUTL)

    pid = cells.pid.to_numpy()
    order = pd.unique(pid)
    row_of = {s: i for i, s in enumerate(order)}
    prow = np.array([row_of[s] for s in pid])
    nP, nL = len(order), len(li)
    sums = np.zeros((nP, nL), np.float64)
    cnt = np.zeros(nP, np.float64)
    Ln = (L * L).sum(1)
    for a in range(0, len(X), CHUNK):
        b = min(a + CHUNK, len(X))
        xc = X[a:b]
        d2 = (xc * xc).sum(1)[:, None] + Ln[None, :] - 2.0 * xc @ L.T
        K = np.exp(-np.maximum(d2, 0) / denom)
        np.add.at(sums, prow[a:b], K)
        np.add.at(cnt, prow[a:b], 1.0)
    emb = (sums / cnt[:, None]).astype(np.float32)

    df = pd.DataFrame(emb, columns=[f"s{j}" for j in range(nL)])
    df.insert(0, "pid", order)
    meta = (cells.groupby("pid")
                 .agg(study=("study", "first"), y=("y", "first"), n_cells=("pid", "size"))
                 .reset_index())
    reg = (ct.groupby("pid").tissue.agg(lambda s: s.value_counts().index[0])
             .rename("region").reset_index())          # dominant region of the donor
    df = meta.merge(reg, on="pid").merge(df, on="pid")
    df.to_parquet(OUT)
    print(f"\nsaved {OUT}: {df.shape}", flush=True)
    print(f"donors: {len(df)} | disease={int(df.y.sum())} control={int((1 - df.y).sum())} "
          f"| studies={df.study.nunique()} | regions={df.region.nunique()}", flush=True)


if __name__ == "__main__":
    main()
