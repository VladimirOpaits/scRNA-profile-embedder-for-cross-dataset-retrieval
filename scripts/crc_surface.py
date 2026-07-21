"""BENEFIT-vs-DIFFICULTY surface (the usefulness/product validation, [[usefulness-validation-plan]]).

Question this answers: as a DROP-IN tool, when does augmenting a user's scVI reference with retrieved
diagnosis-matched cross-batch patients help, and does it ever HURT? We sweep a 2D difficulty grid of
REALISTIC user cohorts -- (rho = diagnosis<->batch confound degree) x (n = cohort size) -- and measure
the augmentation benefit at each cell. Expected product pitch: benefit RISES with rho, FALLS with n,
and at rho~0 (balanced data) benefit ~0 but NOT negative (does-no-harm).

User cohort construction (the one new piece): pick 2 REAL paired studies (both classes) as the
cohort's two batches; assign diagnosis to batches at a target correlation via f = 0.5 + 0.5*rho =
disease fraction in batch A (1-f in batch B). rho=0 -> both batches 50/50 (balanced). rho=1 -> batch
A all disease, batch B all normal (= the fully-confounded core we used before). n = total donors.
Everything downstream reuses crc_scvi.py: scVI reference-mapping, RFF+logreg readout, arms + samebatch
control. Held = an unseen study (transfer target), never in reference/cohort.

Arms per (rho,n,held): raw (PCA, no scVI) | noref (scVI on cohort only) | paired (scVI + retrieved
paired refs) | samebatch (scVI + more cohort-batch donors, same confound) | random (scVI + random
diagnosis-matched refs). benefit = paired-noref; de-confound-vs-data = paired-samebatch.

  python scripts/crc_surface.py --workers 4 --seeds 3        # parallel on one GPU
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import harmony_sweep as hs
import crc_scvi as cs

RESULTS = "data/crc_surface_results.parquet"
KEY = ["held", "rho", "n", "strategy", "K", "seed"]
KREF = 6
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
NS = [10, 24, 48]
HELD = ["Chen_2024_Cancer_Cell", "Lee_2020_Nat_Genet", "Joanito_2022_Nat_Genet"]


def paired_pool(exclude):
    out = {}
    for s, g in hs.G["don"][~hs.G["don"].study.isin(exclude)].groupby("study", observed=True):
        dem = g.pid[g.y == 1].tolist()
        nor = g.pid[g.y == 0].tolist()
        if dem and nor:
            out[s] = (dem, nor)
    return out


def pick_batches(pool, k=2):
    """k paired studies with the largest min(disease,normal) capacity -> can hit any (rho,n)."""
    return sorted(pool, key=lambda s: min(len(pool[s][0]), len(pool[s][1])), reverse=True)[:k]


def _draw(pool, study, n_dis, n_nor, used, r):
    dem = [p for p in pool[study][0] if p not in used]
    nor = [p for p in pool[study][1] if p not in used]
    d = list(r.choice(dem, min(n_dis, len(dem)), replace=False)) if n_dis else []
    n = list(r.choice(nor, min(n_nor, len(nor)), replace=False)) if n_nor else []
    return d + n


def build_cohort(held, rho, n, seed):
    """2 real paired studies as batches; diagnosis assigned at target confound rho, size n."""
    pool = paired_pool([held])
    A, B = pick_batches(pool, 2)
    r = np.random.default_rng(seed)
    f = 0.5 + 0.5 * rho
    half = n // 2
    aD = round(f * half); aN = half - aD
    bD = round((1 - f) * half); bN = half - bD
    used = set()
    core = _draw(pool, A, aD, aN, used, r)
    used |= set(core)
    core += _draw(pool, B, bD, bN, used, r)
    return core, A, B


def emp_rho(core):
    """realized diagnosis<->batch correlation in the built cohort (record vs target)."""
    don = hs.G["don"].set_index("pid")
    sub = don.loc[core]
    y = sub.y.to_numpy().astype(float)
    b = (sub.study == sub.study.iloc[0]).to_numpy().astype(float)   # batch A indicator
    if y.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(abs(np.corrcoef(y, b)[0, 1]))


def build_refs(held, A, B, core, arm, K, seed):
    if arm == "noref":
        return []
    if arm == "samebatch":                      # more donors from the SAME 2 batches, same rho
        pool = paired_pool([held])
        r = np.random.default_rng(seed + 991)
        used = set(core)
        extra = _draw(pool, A, K, 0, used, r) + _draw(pool, B, 0, K, used | set(), r)
        return extra
    pstud = paired_pool([held, A, B])           # retrieval reservoir: other paired studies
    return hs.build_refs(pstud, hs.order_studies(pstud, arm, seed), K)


def run(held, rho, n, arm, K, seed):
    core, A, B = build_cohort(held, rho, n, seed)
    held_pid = hs.G["don"].pid[hs.G["don"].study == held].tolist()
    er = emp_rho(core)
    if arm == "raw":
        lat, pid_arr, yb = cs.pca_latent(core, held_pid)
        refs = []
    else:
        refs = build_refs(held, A, B, core, arm, K, seed)
        lat, pid_arr, yb = cs.scvi_latent(list(core) + list(refs), held_pid, seed=seed)
    auc = cs.auc_from_latent(lat, pid_arr, yb, list(core) + list(refs), held_pid)
    return dict(held=held, rho=rho, n=n, emp_rho=er, batchA=A, batchB=B, strategy=arm,
                K=K, seed=seed, transfer_auc=auc, n_train_pat=len(core) + len(refs))


def build_jobs(seeds, kref=KREF):
    jobs = []
    for held in HELD:
        for rho in RHOS:
            for n in NS:
                jobs.append((held, rho, n, "raw", 0, 0))
                for s in range(seeds):
                    for arm in ["noref", "coverage", "quantile", "lowq", "samebatch", "random"]:
                        jobs.append((held, rho, n, arm, 0 if arm == "noref" else kref, s))
    return jobs


def _flush(rows):
    if not rows:
        return
    new = pd.DataFrame(rows)
    if os.path.exists(RESULTS):
        old = pd.read_parquet(RESULTS)
        old = old[~old.set_index(KEY).index.isin(new.set_index(KEY).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(RESULTS)


def _worker(job):
    return run(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kref", type=int, default=KREF)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.seeds, args.kref)
    if args.smoke:
        jobs = [j for j in jobs if j[1] in (0.0, 1.0) and j[2] == 24][:6]
    done = set()
    if os.path.exists(RESULTS):
        p = pd.read_parquet(RESULTS)
        done = set(map(tuple, p[KEY].values))
    todo = [j for j in jobs if j not in done]
    print(f"crc_surface: {len(jobs)} jobs, {len(todo)} to run, {args.workers} workers", flush=True)
    if not todo:
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows, n = [], 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=cs.init) as ex:
        futs = {ex.submit(_worker, j): j for j in todo}
        for fu in as_completed(futs):
            r = fu.result()
            rows.append(r)
            n += 1
            print(f"  [{n}/{len(todo)}] {r['held'][:8]} rho={r['rho']} n={r['n']} "
                  f"{r['strategy']:9s} s{r['seed']} AUC={r['transfer_auc']:.3f}", flush=True)
            if n % 8 == 0:
                _flush(rows)
    _flush(rows)
    print(f"done crc_surface: {len(rows)} new rows -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
