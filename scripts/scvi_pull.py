import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import cellxgene_census.experimental as ex

CENSUS_VERSION = "2025-11-08"
H5DIR = "data/geneformer_input/hlca"
OUTDIR = "data/scvi"


def fname(sid):
    return sid.replace(" ", "_").replace("|", "-").replace("/", "_") + ".npy"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    obs_cache = os.path.join(OUTDIR, "obs.parquet")
    if os.path.exists(obs_cache):
        obs = pd.read_parquet(obs_cache)
    else:
        obs = []
        for f in sorted(glob.glob(os.path.join(H5DIR, "*.h5ad"))):
            a = sc.read_h5ad(f, backed="r")
            obs.append(a.obs[["soma_joinid", "sample_id", "technology",
                              "disease"]].copy())
        obs = pd.concat(obs, ignore_index=True)
        obs["soma_joinid"] = obs["soma_joinid"].astype(np.int64)
        obs.to_parquet(obs_cache)
    joinids = obs["soma_joinid"].to_numpy(dtype=np.int64)
    print("cells:", len(obs), "| samples:", obs.sample_id.nunique(), flush=True)

    md = [m for m in ex.get_all_available_embeddings(CENSUS_VERSION)
          if m["embedding_name"] == "scvi"
          and m["experiment_name"] == "homo_sapiens"][0]
    uri = "s3://cellxgene-contrib-public" + md["relative_uri"]
    X = ex.get_embedding(CENSUS_VERSION, uri, joinids).astype(np.float32)
    print("scvi dim:", X.shape[1], "| nan rows:",
          int(np.isnan(X).any(axis=1).sum()), flush=True)

    rows = []
    for sid, g in obs.groupby("sample_id"):
        idx = g.index.to_numpy()
        np.save(os.path.join(OUTDIR, fname(sid)), X[idx])
        r = g.iloc[0]
        rows.append({"sample_id": sid, "technology": r.technology,
                     "disease": r.disease, "n_cells": len(idx)})
    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(OUTDIR, "meta.csv"), index=False)
    print(meta.groupby(["technology", "disease"]).size())


if __name__ == "__main__":
    main()
