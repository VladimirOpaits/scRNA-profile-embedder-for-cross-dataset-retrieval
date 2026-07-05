import os
import re
import gc
import signal
import resource
import numpy as np
import pandas as pd
import cellxgene_census

CENSUS_VERSION = "2025-11-08"
SELECTED = "data/selected_samples.csv"
OUT_DIR = "data/geneformer_input/hlca"
CAP = 1000
SEED = 0
TIMEOUT = 240
OBS_COLS = ["soma_joinid", "assay", "disease", "donor_id", "dataset_id", "cell_type"]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sel = pd.read_csv(SELECTED)
    want = set(sel["sample_id"])
    have = {re.sub(r"\.h5ad$", "", f) for f in os.listdir(OUT_DIR)}
    todo = {sid for sid in want if safe(sid) not in have}
    print(f"{len(want)} selected | {len(want)-len(todo)} done | {len(todo)} to fetch",
          flush=True)
    if not todo:
        print("nothing to do")
        return

    diseases = sorted(sel["disease"].unique())
    assays = sorted(sel["assay"].unique())
    dis_list = ", ".join(f'"{d}"' for d in diseases)
    asy_list = ", ".join(f'"{a}"' for a in assays)
    vf = (f'tissue_general == "lung" and is_primary_data == True '
          f'and disease in [{dis_list}] and assay in [{asy_list}]')

    rng = np.random.default_rng(SEED)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        print("scanning obs (one pass)...", flush=True)
        obs = cellxgene_census.get_obs(
            census, "homo_sapiens", value_filter=vf,
            column_names=["soma_joinid", "donor_id", "assay", "dataset_id"])
        obs["sample_id"] = obs["donor_id"].astype(str) + " | " + obs["assay"].astype(str)
        keys = set(zip(sel["dataset_id"], sel["donor_id"], sel["assay"]))
        tup = list(zip(obs["dataset_id"], obs["donor_id"], obs["assay"]))
        obs = obs[[t in keys for t in tup]]
        obs = obs[obs["sample_id"].isin(todo)]
        print(f"candidate cells: {len(obs)} across {obs['sample_id'].nunique()} samples",
              flush=True)

        sample_ids = {}
        for sid, g in obs.groupby("sample_id"):
            ids = g["soma_joinid"].to_numpy()
            if len(ids) > CAP:
                ids = rng.choice(ids, size=CAP, replace=False)
            sample_ids[sid] = np.sort(ids)

        chunks, cur, ncell = [], [], 0
        for sid in sorted(sample_ids):
            cur.append(sid)
            ncell += len(sample_ids[sid])
            if ncell >= 2500:
                chunks.append(cur); cur, ncell = [], 0
        if cur:
            chunks.append(cur)
        print(f"picked {sum(len(v) for v in sample_ids.values())} cells "
              f"in {len(chunks)} chunks", flush=True)

        for ci, chunk in enumerate(chunks):
            ids = np.sort(np.concatenate([sample_ids[s] for s in chunk])).tolist()
            signal.alarm(TIMEOUT)
            try:
                ad = cellxgene_census.get_anndata(
                    census, "homo_sapiens", X_name="raw",
                    obs_coords=ids, column_names={"obs": OBS_COLS})
                signal.alarm(0)
            except Timeout:
                signal.alarm(0)
                print(f"  chunk {ci+1}/{len(chunks)} TIMEOUT ({chunk}) -> skip",
                      flush=True)
                continue

            ad.var["ensembl_id"] = ad.var["feature_id"].values
            ad.obs["sample_id"] = (ad.obs["donor_id"].astype(str) + " | "
                                   + ad.obs["assay"].astype(str))
            ad.obs["technology"] = ad.obs["assay"].values
            for sid in chunk:
                a = ad[ad.obs["sample_id"] == sid].copy()
                a.obs["n_counts"] = np.asarray(a.X.sum(axis=1)).ravel()
                a.write_h5ad(os.path.join(OUT_DIR, safe(sid) + ".h5ad"))
                del a
            print(f"  chunk {ci+1}/{len(chunks)}: {ad.n_obs} cells -> "
                  f"{len(chunk)} samples | maxRSS={rss_gb():.2f}GB", flush=True)
            del ad
            gc.collect()

    files = [f for f in os.listdir(OUT_DIR) if f.endswith(".h5ad")]
    print(f"\ndone: {len(files)}/{len(want)} h5ad files | maxRSS={rss_gb():.2f}GB",
          flush=True)


if __name__ == "__main__":
    main()
