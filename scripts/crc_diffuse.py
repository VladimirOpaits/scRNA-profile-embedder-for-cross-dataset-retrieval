"""Diffuse-confound control (CRC only): does the retrieval-augmented Harmony de-confounding effect
SURVIVE when the starting core is LESS degenerately confounded?

Weak spot in the main result: the confounded core is the WORST case for Harmony -- disease from ONE
study + normal from ONE study, so batch A == 100% disease and batch B == 100% normal. Harmony is
batch-aware but disease-BLIND, so at this extreme ANY A-vs-B difference is, by its assumption, batch
-> it is forced to erase biology with batch. That inflates the headroom our references then recover.

Honest dilution: keep diagnosis strictly tied to the SIDE (disease-source vs normal-source studies,
they must be -- we can only draw a single class from a single-class study), but SPREAD each side
across N distinct source studies instead of 1. At N=1 this is exactly the main design. As N grows
Harmony sees MANY batches per class, the contrast stops being one clean "study A vs study B" vector,
and (hypothesis) harmony_noref should hurt less -> the de-confounding headroom should shrink.

Leakage-safe: held excluded from every pool; the N core-source studies (single-class) are kept
DISJOINT from the paired-study reservoir used for refs, so "how diffuse the start is" (N axis) does
not contaminate "how many paired refs we add" (K axis).

Per (held, N): raw / harmony_noref / samebatch(control, more data same source batches) /
paired@K=6 (coverage, quantile, random x seeds). Same 5 metrics as the main sweep.

  python scripts/crc_diffuse.py [--workers 8] [--seeds 5]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import harmony_sweep as hs

RESULTS = "data/crc_diffuse_results.parquet"
KEY = ["held", "n_src", "strategy", "K", "seed"]
NS = [1, 2, 3]
K_REF = 6


def pick_sources(H, n):
    """top-n disease-only source studies + top-n normal source studies, DISJOINT, excluding held."""
    don = hs.G["don"]
    dvc = don[(don.y == 1) & (don.study != H)].study.value_counts()
    dem = dvc[dvc >= 2].index.tolist()[:n]
    nvc = don[(don.y == 0) & (don.study != H)].study.value_counts()
    nor = [s for s in nvc[nvc >= 2].index.tolist() if s not in dem][:n]
    return dem, nor


def _spread(studies, y, total, cset, rng):
    """draw `total` donors of class y spread ~evenly across `studies`, skipping cset."""
    per = [total // len(studies)] * len(studies)
    for i in range(total % len(studies)):
        per[i] += 1
    out = []
    for s, k in zip(studies, per):
        pool = [p for p in hs.G["don"].pid[(hs.G["don"].study == s) & (hs.G["don"].y == y)]
                if p not in cset]
        out += list(rng.choice(pool, min(k, len(pool)), replace=False))
    return out


def setup_core_diffuse(H, dem, nor):
    r = np.random.default_rng(0)
    core = _spread(dem, 1, 25, set(), r) + _spread(nor, 0, 25, set(), r)
    return core, hs.G["don"].pid[hs.G["don"].study == H].tolist()


def build_samebatch_diffuse(dem, nor, core, K, seed):
    """control: K more disease + K more normal from the SAME N source studies -> more data, same
    (multi-)batch confound preserved."""
    r = np.random.default_rng(seed)
    cset = set(core)
    return _spread(dem, 1, K, cset, r) + _spread(nor, 0, K, cset, r)


def paired_studies_diffuse(H, sources):
    """paired reservoir: studies with BOTH classes, excluding held AND every core-source study."""
    ex = set([H]) | set(sources)
    out = {}
    for s, g in hs.G["don"][~hs.G["don"].study.isin(ex)].groupby("study", observed=True):
        dem = g.pid[g.y == 1].tolist()
        nor = g.pid[g.y == 0].tolist()
        if dem and nor:
            out[s] = (dem, nor)
    return out


def confound_severity_core(core):
    A, ix, don = hs.G["SIGARR"], hs.G["sidx"], hs.G["don"]
    ymap = dict(zip(don.pid, don.y))
    a = A[[ix[p] for p in core if ymap[p] == 1]].mean(0)
    b = A[[ix[p] for p in core if ymap[p] == 0]].mean(0)
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _job(job):
    H, N, how, K, seed = job
    dem, nor = pick_sources(H, N)
    core, held_pid = setup_core_diffuse(H, dem, nor)
    sev = confound_severity_core(core)
    if how == "raw":
        m = hs.evaluate_full(core, [], held_pid, do_harmony=False)
    elif how == "harmony_noref":
        m = hs.evaluate_full(core, [], held_pid, do_harmony=True)
    elif how == "samebatch":
        m = hs.evaluate_full(core, build_samebatch_diffuse(dem, nor, core, K, seed), held_pid)
    else:
        pstud = paired_studies_diffuse(H, dem + nor)
        refs = hs.build_refs(pstud, hs.order_studies(pstud, how, seed), K)
        m = hs.evaluate_full(core, refs, held_pid)
    return dict(held=H, n_src=N, n_dem_src=len(dem), n_nor_src=len(nor),
                confound_severity=sev, strategy=how, K=K, seed=seed, **m)


def build_jobs(seeds):
    hs._init("crc")
    jobs = []
    for H in hs.CONFIG["crc"]["held"]:
        for N in NS:
            jobs.append((H, N, "raw", 0, 0))
            jobs.append((H, N, "harmony_noref", 0, 0))
            jobs.append((H, N, "coverage", K_REF, 0))
            jobs.append((H, N, "quantile", K_REF, 0))
            for s in range(seeds):
                jobs.append((H, N, "random", K_REF, s))
                jobs.append((H, N, "samebatch", K_REF, s))
    hs.G.clear()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    jobs = build_jobs(args.seeds)
    done = set()
    if os.path.exists(RESULTS):
        p = pd.read_parquet(RESULTS)
        done = set(map(tuple, p[KEY].values))
    todo = [j for j in jobs if j not in done]
    print(f"crc_diffuse: {len(jobs)} jobs, {len(todo)} to run, {args.workers} workers", flush=True)
    if not todo:
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows, n = [], 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=hs._init,
                             initargs=("crc",)) as ex:
        futs = [ex.submit(_job, j) for j in todo]
        for f in as_completed(futs):
            rows.append(f.result())
            n += 1
            if n % 20 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
                _flush(rows)
    _flush(rows)
    print(f"done crc_diffuse: {len(rows)} new rows -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
