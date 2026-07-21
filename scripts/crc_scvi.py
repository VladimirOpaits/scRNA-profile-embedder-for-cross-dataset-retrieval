"""scVI reference-mapping version of the retrieval-augmented de-confounding experiment (CRC).

Moves the downstream from Harmony (unsupervised, transductive, no out-of-sample transform) to scVI
(the standard modern scRNA integrator), using its INDUCTIVE reference-mapping: train scVI on the
reference cohort (confounded core + retrieved refs), then project each held patient via
load_query_data WITHOUT retraining the reference. This is both the rigorous protocol (held cells
never in reference training -> closes the transductive caveat) and the product framing (a mini-lib
that augments your scVI reference with diagnosis-matched cross-batch patients).

Same lever as Harmony: does adding PAIRED (1 disease + 1 normal from one outside study) references let
scVI's latent keep disease signal that transfers to an unseen study? Same samebatch control (more
data from the SAME two confounded batches) isolates de-confounding from data-quantity.

Readout unchanged: RFF-MMD patient signatures on the scVI latent -> logreg on core+refs -> transfer
AUC on held. Reuses harmony_sweep's data + core/refs/selection machinery (scripts/harmony_sweep.py).

  python scripts/crc_scvi.py --smoke        # 1 held x 1 core, all arms, timing + GPU
  python scripts/crc_scvi.py                 # small mini-sweep
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "scripts")
import harmony_sweep as hs

COUNTS = "data/crc/counts_hv.npz"
CELLS0 = "data/crc/cells.parquet"
N_LATENT = 10
REF_EPOCHS = 150
QUERY_EPOCHS = 80
BATCH = 512                     # GPU is <0.3GB at batch 128 -> raise batch, fewer steps/epoch
RESULTS = "data/crc_scvi_results.parquet"
KEY = ["held", "dem_src", "normal_src", "strategy", "K", "seed"]
KREF = 6
HELD = ["Chen_2024_Cancer_Cell", "Lee_2020_Nat_Genet", "Joanito_2022_Nat_Genet"]

CF = {}   # counts + filtered-row alignment, loaded once


def init():
    hs._init("crc")                                   # populates hs.G (cells, Z, don, pid_rows, SIGARR)
    cells0 = pd.read_parquet(CELLS0).reset_index(drop=True)
    keep = cells0.sample_type.isin(["tumor", "normal"]).to_numpy()   # same filter hs uses for crc
    counts = sp.load_npz(COUNTS).tocsr()[keep]        # rows now aligned to hs.G["cells"]/pid_rows
    assert counts.shape[0] == len(hs.G["cells"]), "counts/cells row mismatch"
    CF["counts"] = counts


def _adata(pids):
    import anndata as ad
    rows = hs._rows(pids)                              # <=120 cells/donor, indices into filtered array
    obs = hs.G["cells"].iloc[rows][["pid", "study", "y"]].reset_index(drop=True)
    obs["study"] = obs["study"].astype(str)
    a = ad.AnnData(X=CF["counts"][rows].copy(), obs=obs)
    return a, rows


def scvi_latent(ref_pids, held_pid, seed=0):
    """train scVI on ref cohort, map held via scArches, return (latent[all], pid_order, y, tr_mask).
    seed drives scVI init/training so deterministic-selection arms (coverage/quantile) still get
    run-to-run error bars, not a single point."""
    import scvi
    scvi.settings.seed = int(seed)
    ref_a, _ = _adata(ref_pids)
    scvi.model.SCVI.setup_anndata(ref_a, batch_key="study")
    ref = scvi.model.SCVI(ref_a, n_latent=N_LATENT)
    ref.train(max_epochs=REF_EPOCHS, batch_size=BATCH, accelerator="gpu",
              enable_progress_bar=False, early_stopping=True)
    ref_lat = ref.get_latent_representation()

    q_a, _ = _adata(held_pid)
    scvi.model.SCVI.prepare_query_anndata(q_a, ref)
    q = scvi.model.SCVI.load_query_data(q_a, ref)
    q.train(max_epochs=QUERY_EPOCHS, batch_size=BATCH, accelerator="gpu",
            enable_progress_bar=False, plan_kwargs={"weight_decay": 0.0})
    q_lat = q.get_latent_representation()

    lat = np.vstack([ref_lat, q_lat]).astype(np.float32)
    pid_arr = np.concatenate([ref_a.obs.pid.to_numpy(), q_a.obs.pid.to_numpy()])
    yb = np.concatenate([ref_a.obs.y.to_numpy(), q_a.obs.y.to_numpy()])
    # release scVI/torch memory: workers handle hundreds of jobs -> models+CUDA cache accumulate
    # and OOM-kill the process (Python 3.10 has no max_tasks_per_child).
    import gc
    import torch
    del ref, q, ref_a, q_a, ref_lat, q_lat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lat, pid_arr, yb


def pca_latent(ref_pids, held_pid):
    """raw baseline: no scVI, patient signatures on the pool-only PCA."""
    pids = list(ref_pids) + list(held_pid)
    rows = hs._rows(pids)
    lat = hs.G["Z"][rows]
    meta = hs.G["cells"].iloc[rows]
    return lat, meta.pid.to_numpy(), meta.y.to_numpy()


def auc_from_latent(lat, pid_arr, yb, train_pids, held_pid):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    order = pd.unique(pid_arr)
    S = hs._sigs(lat, pid_arr, order)
    yby = pd.Series(yb, index=pid_arr).groupby(level=0).first()
    ix = {p: i for i, p in enumerate(order)}
    tr = [ix[p] for p in train_pids]
    te = [ix[p] for p in held_pid]
    sc = StandardScaler().fit(S[tr])
    clf = LogisticRegression(C=0.01, max_iter=5000).fit(sc.transform(S[tr]), yby.loc[train_pids])
    return roc_auc_score(yby.loc[held_pid], clf.predict_proba(sc.transform(S[te]))[:, 1])


def build_refs_for(H, DEM, NOR, core, arm, K, seed):
    if arm == "noref":
        return []
    if arm == "samebatch":
        return hs.build_samebatch(H, DEM, NOR, core, K, seed)
    pstud = hs.paired_studies(H, DEM, NOR)
    return hs.build_refs(pstud, hs.order_studies(pstud, arm, seed), K)


def run_config(H, DEM, NOR, arm, K, seed):
    core, held_pid = hs.setup_core(H, DEM, NOR)
    if arm == "raw":
        lat, pid_arr, yb = pca_latent(core, held_pid)
        refs = []
    else:
        refs = build_refs_for(H, DEM, NOR, core, arm, K, seed)
        lat, pid_arr, yb = scvi_latent(list(core) + list(refs), held_pid, seed=seed)
    auc = auc_from_latent(lat, pid_arr, yb, list(core) + list(refs), held_pid)
    return auc, len(core) + len(refs)


def smoke():
    import torch
    init()
    H, DEM, NOR = "Chen_2024_Cancer_Cell", None, None
    don = hs.G["don"]
    DEM = don[(don.y == 1) & (don.study != H)].study.value_counts().index[0]
    NOR = don[(don.y == 0) & (don.study != H) & (don.study != DEM)].study.value_counts().index[0]
    print(f"smoke: held={H}  DEM={DEM}  NOR={NOR}", flush=True)
    for arm in ["raw", "noref", "samebatch", "coverage"]:
        t0 = time.time()
        auc, n = run_config(H, DEM, NOR, arm, 6, 0)
        gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"  {arm:10s} AUC={auc:.3f}  n_train_pat={n}  {time.time()-t0:5.1f}s  gpu_peak={gpu:.2f}GB",
              flush=True)


def build_jobs(seeds, ndem, nnor):
    don = hs.G["don"]
    jobs = []
    for H in HELD:
        dem = don[(don.y == 1) & (don.study != H)].study.value_counts()
        nor = don[(don.y == 0) & (don.study != H)].study.value_counts()
        dem = dem[dem >= 2].index[:ndem].tolist()
        nor = nor[nor >= 2].index[:nnor].tolist()
        for A in dem:
            for B in nor:
                if A == B:
                    continue
                jobs.append((H, A, B, "raw", 0, 0))
                jobs.append((H, A, B, "noref", 0, 0))
                for s in range(seeds):
                    jobs.append((H, A, B, "coverage", KREF, s))
                    jobs.append((H, A, B, "quantile", KREF, s))
                    jobs.append((H, A, B, "samebatch", KREF, s))
                    jobs.append((H, A, B, "random", KREF, s))
    return jobs


def _flush(rows):
    import os
    if not rows:
        return
    new = pd.DataFrame(rows)
    if os.path.exists(RESULTS):
        old = pd.read_parquet(RESULTS)
        old = old[~old.set_index(KEY).index.isin(new.set_index(KEY).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(RESULTS)


def sweep(seeds, ndem, nnor):
    import os
    init()
    jobs = build_jobs(seeds, ndem, nnor)
    done = set()
    if os.path.exists(RESULTS):
        p = pd.read_parquet(RESULTS)
        done = set(map(tuple, p[KEY].values))
    todo = [j for j in jobs if j not in done]
    print(f"crc_scvi sweep: {len(jobs)} jobs, {len(todo)} to run", flush=True)
    rows = []
    for i, (H, A, B, arm, K, seed) in enumerate(todo):
        t0 = time.time()
        auc, n = run_config(H, A, B, arm, K, seed)
        rows.append(dict(held=H, dem_src=A, normal_src=B, strategy=arm, K=K, seed=seed,
                         transfer_auc=auc, n_train_pat=n))
        print(f"  [{i+1}/{len(todo)}] {H[:10]} {arm:10s} s{seed} AUC={auc:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if (i + 1) % 6 == 0:
            _flush(rows)
    _flush(rows)
    print(f"done crc_scvi: {len(rows)} new rows -> {RESULTS}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ndem", type=int, default=3)
    ap.add_argument("--nnor", type=int, default=2)
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        sweep(args.seeds, args.ndem, args.nnor)
