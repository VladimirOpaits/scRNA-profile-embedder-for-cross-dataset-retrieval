"""Retrieval-augmented batch correction: does supplying Harmony with diagnosis-matched,
batch-diverse REFERENCE patients recover biology that transfers across studies -- and does
RFF-MMD-diverse selection of those references beat random selection?

This is the batch-LABEL-AWARE downstream the lever was always meant for. A plain logreg is
batch-blind, so "which batches you add" can't help it (our earlier null result). Harmony USES the
batch labels, so the reference set changes the batch estimate -> retrieval can matter.

Retrieval, two stages (Vlad's design):
  1. metadata filter (hard): same diagnosis, DIFFERENT study  -> biology fixed, so distance is now
     pure batch.
  2. RFF-MMD farthest-point within that filtered set          -> maximize batch spread for Harmony.

Design: confounded core (COVID from ONE study + normal from ONE atlas -> diagnosis==study), add K
reference patients per class, run Harmony (batch=study) on core+refs+held jointly, recompute patient
signatures on the CORRECTED embedding, train on core+refs, transfer to held.

Arms: raw (no Harmony) | harmony+no-ref | harmony+random-ref | harmony+smart-ref (RFF-MMD diverse).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import harmonypy
sys.path.insert(0, "scripts")
from selection import cosine_distance_matrix, greedy_farthest

PCA = "data/blood/pca50.npy"
CELLS = "data/blood/cells.parquet"
SIG = "data/blood/signatures_lf.parquet"          # patient signatures for retrieval selection
OUT = "blood_harmony.png"

HELD = "30cd5311"                                  # strong ceiling (~1.0), COVID+normal
COVID_SRC = "9dbab10c"                             # confounded core: all core COVID from here
NORMAL_SRC = "c838aec3"                            # confounded core: all core normal from here
CORE_PER_CLASS = 25
CELLS_PER_DONOR = 200                             # cap for Harmony speed
KS = [0, 10, 20, 40]
RFF_D = 512
SEED = 0


def rff_sigs(Z, pid_arr, order, sigma):
    """RFF-MMD patient signatures on a (corrected) cell embedding."""
    rng = np.random.default_rng(SEED)
    Z = np.asarray(Z, dtype=np.float32)
    W = (rng.standard_normal((Z.shape[1], RFF_D)) / sigma).astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, RFF_D).astype(np.float32)
    phi = np.sqrt(2.0 / RFF_D) * np.cos(Z @ W + b)
    prow = pd.Series(np.arange(len(order)), index=order).loc[pid_arr].to_numpy()
    S = np.zeros((len(order), RFF_D), np.float32); c = np.zeros(len(order))
    np.add.at(S, prow, phi); np.add.at(c, prow, 1.0)
    return S / c[:, None]


def median_sigma(Z, rng, m=4000):
    s = Z[rng.choice(len(Z), min(m, len(Z)), replace=False)]
    d = np.sqrt(((s[:, None] - s[None]) ** 2).sum(-1))
    return float(np.median(d[d > 0]))


def transfer(Zsub, meta, train_pid, held_pid):
    """patient reps on embedding Zsub -> train on train_pid, AUC on held_pid."""
    order = pd.unique(meta.pid.to_numpy())
    sigma = median_sigma(Zsub, np.random.default_rng(SEED))
    S = rff_sigs(Zsub, meta.pid.to_numpy(), order, sigma)
    yb = meta.drop_duplicates("pid").set_index("pid").y
    idx = {p: i for i, p in enumerate(order)}
    tr = [idx[p] for p in train_pid]; te = [idx[p] for p in held_pid]
    sc = StandardScaler().fit(S[tr])
    clf = LogisticRegression(C=0.01, max_iter=5000).fit(sc.transform(S[tr]), yb.loc[train_pid])
    return roc_auc_score(yb.loc[held_pid], clf.predict_proba(sc.transform(S[te]))[:, 1])


def run_harmony(Zsub, meta):
    ho = harmonypy.run_harmony(Zsub, meta, ["study"], max_iter_harmony=10)
    Zc = np.asarray(ho.Z_corr, dtype=np.float32)        # np.matrix + orientation vary by version
    return Zc if Zc.shape[0] == len(meta) else Zc.T     # force (n_cells, n_pc)


def main():
    Z = np.load(PCA).astype(np.float32)
    cells = pd.read_parquet(CELLS).reset_index(drop=True)
    sig = pd.read_parquet(SIG)
    Scol = [c for c in sig.columns if c.startswith("s") and c[1:].isdigit()]
    Dsig = cosine_distance_matrix(sig[Scol].to_numpy())
    sidx = {p: i for i, p in enumerate(sig.pid)}      # pid -> row in signature distance matrix

    donors = cells.drop_duplicates("pid")[["pid", "study", "y"]]
    held_pid = donors.pid[donors.study == HELD].tolist()
    core_cov = donors.pid[(donors.study == COVID_SRC) & (donors.y == 1)].tolist()
    core_nor = donors.pid[(donors.study == NORMAL_SRC) & (donors.y == 0)].tolist()
    rng = np.random.default_rng(SEED)
    core_cov = list(rng.choice(core_cov, min(CORE_PER_CLASS, len(core_cov)), replace=False))
    core_nor = list(rng.choice(core_nor, min(CORE_PER_CLASS, len(core_nor)), replace=False))
    core = core_cov + core_nor

    # reference reservoir: metadata-filtered (same dx, different study from core & held)
    ref_cov = donors.pid[(donors.y == 1) & (~donors.study.isin([HELD, COVID_SRC]))].tolist()
    ref_nor = donors.pid[(donors.y == 0) & (~donors.study.isin([HELD, NORMAL_SRC]))].tolist()
    print(f"held={HELD} ({len(held_pid)} donors) | core {len(core_cov)}COVID+{len(core_nor)}normal")
    print(f"reference reservoir: {len(ref_cov)} COVID / {len(ref_nor)} normal donors", flush=True)

    def cellrows(pids):
        m = cells[cells.pid.isin(pids)]
        return (m.groupby("pid", group_keys=False)
                 .apply(lambda g: g.sample(min(CELLS_PER_DONOR, len(g)), random_state=SEED)))

    def pick(reservoir, k, arm, seed):
        r = np.random.default_rng(seed)
        if k == 0 or len(reservoir) == 0:
            return []
        if arm == "random":
            return list(r.choice(reservoir, min(k, len(reservoir)), replace=False))
        pool = [sidx[p] for p in reservoir]
        start = [int(r.choice(pool))]
        chosen = greedy_farthest(Dsig, start, [i for i in pool if i != start[0]], k - 1)
        inv = {v: kk for kk, v in sidx.items()}
        return [inv[i] for i in start + chosen][:k]

    joinid_to_pos = pd.Series(np.arange(len(cells)), index=cells.soma_joinid)

    def evaluate(ref_pids, harmony):
        pids = core + ref_pids + held_pid
        sub = cellrows(pids).reset_index(drop=True)
        Zsub = Z[sub["soma_joinid"].map(joinid_to_pos).to_numpy()]   # cells aligned to sub rows
        meta = sub[["pid", "study", "y"]].copy()
        if harmony:
            Zsub = run_harmony(Zsub, meta)
        return transfer(Zsub, meta, core + ref_pids, held_pid)

    # within-held ceiling (raw space, for reference)
    hc = cellrows(held_pid).reset_index(drop=True)
    pos = hc["soma_joinid"].map(pd.Series(np.arange(len(cells)), index=cells.soma_joinid)).to_numpy()
    order = pd.unique(hc.pid.to_numpy())
    Sh = rff_sigs(Z[pos], hc.pid.to_numpy(), order, median_sigma(Z[pos], rng))
    yh = hc.drop_duplicates("pid").set_index("pid").y.loc[order].to_numpy()
    ceil = roc_auc_score(yh, cross_val_predict(LogisticRegression(C=0.01, max_iter=5000),
            StandardScaler().fit_transform(Sh), yh,
            cv=StratifiedKFold(4, shuffle=True, random_state=0), method="predict_proba")[:, 1])
    print(f"held within-study ceiling = {ceil:.3f}", flush=True)

    res = {"harmony+smart": [], "harmony+random": []}
    raw_core = evaluate([], harmony=False)
    harm_noref = evaluate([], harmony=True)
    print(f"raw core (no harmony, no ref) = {raw_core:.3f}", flush=True)
    print(f"harmony, no ref              = {harm_noref:.3f}", flush=True)
    for k in KS:
        if k == 0:
            continue
        for arm in ["smart", "random"]:
            vals = []
            for s in range(4):
                cov = pick(ref_cov, k, arm, s); nor = pick(ref_nor, k, arm, s)
                vals.append(evaluate(cov + nor, harmony=True))
            res[f"harmony+{arm}"].append((k, np.mean(vals), np.std(vals) / 2))
            print(f"  K={k:3d} harmony+{arm:6s} = {np.mean(vals):.3f} +/- {np.std(vals):.3f}",
                  flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(ceil, ls="--", color="green", label=f"ceiling {ceil:.2f}")
    ax.axhline(raw_core, ls=":", color="black", label=f"raw core {raw_core:.2f}")
    ax.axhline(harm_noref, ls="-.", color="0.5", label=f"harmony no-ref {harm_noref:.2f}")
    for arm, col in [("harmony+smart", "#c0392b"), ("harmony+random", "#2980b9")]:
        a = np.array(res[arm]); ax.errorbar(a[:, 0], a[:, 1], yerr=a[:, 2], fmt="-o", color=col,
                                            capsize=3, label=arm)
    ax.set_xlabel("reference patients retrieved / class"); ax.set_ylabel("transfer AUC on held")
    ax.set_title(f"retrieval-augmented Harmony (held={HELD}, confounded core)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT, dpi=130)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
