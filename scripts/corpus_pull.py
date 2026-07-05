import os
import numpy as np
import pandas as pd
import cellxgene_census as cc
import cellxgene_census.experimental as ex
import tiledbsoma as soma

CENSUS_VERSION = "2025-11-08"
OUTDIR = "data/scvi_corpus"
CELLDIR = os.path.join(OUTDIR, "cells")
META = os.path.join(OUTDIR, "meta.csv")
OBS = os.path.join(OUTDIR, "obs_all.parquet")
MIN_CELLS = 200
MAX_CELLS = 800
BIG_SPAN = 3_000_000
SEED = 0
DOWNSTREAM = {"lung adenocarcinoma", "normal"}


def scan_obs():
    if os.path.exists(OBS):
        return pd.read_parquet(OBS)
    vf = 'tissue_general == "lung" and is_primary_data == True'
    print("obs scan (lung/primary/ALL assays)...", flush=True)
    with cc.open_soma(census_version=CENSUS_VERSION) as census:
        o = cc.get_obs(census, "Homo sapiens", value_filter=vf,
                       column_names=["soma_joinid", "dataset_id", "donor_id",
                                     "assay", "disease"])
    o.to_parquet(OBS)
    return o


def pid(ds, dn, asy):
    return (ds[:8] + "_" + str(dn) + "_" + asy).replace(" ", "").replace(
        "'", "").replace("/", "_")


def scvi_uri():
    md = [m for m in ex.get_all_available_embeddings(CENSUS_VERSION)
          if m["embedding_name"] == "scvi"
          and m["experiment_name"] == "homo_sapiens"][0]
    return "s3://cellxgene-contrib-public" + md["relative_uri"]


def main():
    os.makedirs(CELLDIR, exist_ok=True)
    obs = scan_obs()
    for c in ["dataset_id", "donor_id", "assay", "disease"]:
        obs[c] = obs[c].astype(str)
    obs["pid"] = [pid(a, b, c) for a, b, c in
                  zip(obs.dataset_id, obs.donor_id, obs.assay)]
    keep = set(obs.groupby("pid").size().loc[lambda s: s >= MIN_CELLS].index)
    obs = obs[obs.pid.isin(keep)]
    pm = obs.drop_duplicates("pid")
    print("=== assay landscape (patients >=200 cells) ===", flush=True)
    print(pm.assay.value_counts().to_string(), flush=True)
    print("--- adeno/normal by assay ---", flush=True)
    sub = pm[pm.disease.isin(DOWNSTREAM)]
    print(sub.groupby(["assay", "disease"]).size().to_string(), flush=True)

    rng = np.random.default_rng(SEED)
    plan = {}
    for p, g in obs.groupby("pid"):
        j = np.sort(g.soma_joinid.to_numpy(np.int64))
        if len(j) > MAX_CELLS:
            s = int(rng.integers(0, len(j) - MAX_CELLS + 1))
            j = j[s:s + MAX_CELLS]
        r = g.iloc[0]
        plan[p] = (j, r.assay, r.disease, r.dataset_id, r.donor_id)

    order = sorted(plan, key=lambda p: (plan[p][2] not in DOWNSTREAM,
                                        plan[p][2] != "lung adenocarcinoma", p))
    done0 = sum(os.path.exists(os.path.join(CELLDIR, p + ".npy")) for p in order)
    print(f"patients: {len(order)} | already done: {done0}", flush=True)

    uri = scvi_uri()
    ctx = cc.get_default_soma_context()
    with soma.open(uri, context=ctx) as E:
        for i, p in enumerate(order, 1):
            npy = os.path.join(CELLDIR, p + ".npy")
            if os.path.exists(npy):
                continue
            j, assay, disease, ds, dn = plan[p]
            lo, hi = int(j.min()), int(j.max())
            if hi - lo + 1 > BIG_SPAN:
                print(f"{i}/{len(order)} {p} SKIP span {hi-lo+1}", flush=True)
                continue
            buf = np.full((hi - lo + 1, 50), np.nan, dtype=np.float32)
            for tbl in E.read(coords=(slice(lo, hi),)).tables():
                d0 = tbl.column("soma_dim_0").to_numpy() - lo
                d1 = tbl.column("soma_dim_1").to_numpy()
                buf[d0, d1] = tbl.column("soma_data").to_numpy()
            arr = buf[j - lo]
            np.save(npy, arr)
            row = {"pid": p, "assay": assay, "disease": disease,
                   "dataset_id": ds, "donor_id": dn, "n_cells": len(j)}
            pd.DataFrame([row]).to_csv(
                META, mode="a", header=not os.path.exists(META), index=False)
            print(f"{i}/{len(order)} {p} ({disease[:12]}) {len(j)} cells",
                  flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
