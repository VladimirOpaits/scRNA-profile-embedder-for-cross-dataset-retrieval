import os
import sys
import numpy as np
import pandas as pd
import cellxgene_census as cc
import tiledbsoma as soma

sys.path.insert(0, "scripts")
from corpus_pull import (CENSUS_VERSION, CELLDIR, META, OBS, MIN_CELLS,
                         MAX_CELLS, SEED, pid, scvi_uri)

MALIG = ["lung adenocarcinoma", "squamous cell lung carcinoma",
         "non-small cell lung carcinoma", "small cell lung carcinoma",
         "lung cancer", "lung large cell carcinoma", "pleomorphic carcinoma"]
TARGET_DS = ["9f222629"]
SPAN_CAP = 3_000_000


def main():
    obs = pd.read_parquet(OBS)
    obs = obs[obs["is_primary_data"] == True]
    for c in ["dataset_id", "donor_id", "assay", "disease"]:
        obs[c] = obs[c].astype(str)
    obs = obs[obs["disease"].isin(MALIG)]
    obs["pid"] = [pid(a, b, c) for a, b, c in
                  zip(obs.dataset_id, obs.donor_id, obs.assay)]
    keep = set(obs.groupby("pid").size().loc[lambda s: s >= MIN_CELLS].index)
    obs = obs[obs.pid.isin(keep)]
    obs = obs[obs.dataset_id.str[:8].isin(TARGET_DS)]
    have = set(x[:-4] for x in os.listdir(CELLDIR))

    rng = np.random.default_rng(SEED)
    plan = {}
    for p, g in obs.groupby("pid"):
        if p in have:
            continue
        j = np.sort(g.soma_joinid.to_numpy(np.int64))
        if len(j) > MAX_CELLS:
            s = int(rng.integers(0, len(j) - MAX_CELLS + 1))
            j = j[s:s + MAX_CELLS]
        if int(j.max() - j.min() + 1) > SPAN_CAP:
            continue
        r = g.iloc[0]
        plan[p] = (j, r.assay, r.disease, r.dataset_id, r.donor_id)

    print(f"pulling {len(plan)} held-out tumor patients", flush=True)
    uri = scvi_uri()
    ctx = cc.get_default_soma_context()
    with soma.open(uri, context=ctx) as E:
        for p, (j, assay, disease, ds, dn) in plan.items():
            lo, hi = int(j.min()), int(j.max())
            buf = np.full((hi - lo + 1, 50), np.nan, dtype=np.float32)
            for tbl in E.read(coords=(slice(lo, hi),)).tables():
                d0 = tbl.column("soma_dim_0").to_numpy() - lo
                d1 = tbl.column("soma_dim_1").to_numpy()
                buf[d0, d1] = tbl.column("soma_data").to_numpy()
            arr = buf[j - lo]
            np.save(os.path.join(CELLDIR, p + ".npy"), arr)
            row = {"pid": p, "assay": assay, "disease": disease,
                   "dataset_id": ds, "donor_id": dn, "n_cells": len(j)}
            pd.DataFrame([row]).to_csv(
                META, mode="a", header=not os.path.exists(META), index=False)
            print(f"  {p} ({disease[:18]}) {len(j)} cells", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
