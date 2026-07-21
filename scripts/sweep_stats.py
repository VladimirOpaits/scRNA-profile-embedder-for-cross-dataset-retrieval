"""Precision-weighted statistics for the retrieval-augmented Harmony sweep.

The unit of generalization is the HELD study (does the effect transfer to an unseen study?), so we
aggregate cores+seeds within held and test ACROSS held. Naive sign-test throws away magnitude and
lets a thin (few-minority-donor) held cast a near-coin-flip vote; instead we WEIGHT each held by the
size of its minority class (a proxy for how precisely its transfer AUC is estimated). Reported:

  DECONF   = paired(coverage,quantile @maxK) - samebatch(@maxK)   [de-confounding, the claim]
  DATAQ    = samebatch(@maxK) - harmony_noref                     [pure data-quantity, control ~0]

per held; then weighted mean + cluster bootstrap CI (resample held), and a weighted mixed model
(held random intercept) on the paired-vs-samebatch rows as a cross-check.

  python scripts/sweep_stats.py            # all tissues
  python scripts/sweep_stats.py crc
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import harmony_sweep as hs

RESULTS = "data/harmony_results.parquet"


def minority_sizes(tissue):
    hs._init(tissue)
    don = hs.G["don"]
    g = don.groupby("study").y.agg(["sum", "count"])
    m = np.minimum(g["sum"], g["count"] - g["sum"])
    out = m.to_dict()
    hs.G.clear()
    return out


def wmean_ci(vals, w, nboot=20000, seed=0):
    vals, w = np.asarray(vals, float), np.asarray(w, float)
    wm = np.sum(vals * w) / np.sum(w)
    rng = np.random.default_rng(seed)
    n = len(vals)
    boot = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)                       # cluster bootstrap over held
        boot.append(np.sum(vals[idx] * w[idx]) / np.sum(w[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return wm, lo, hi


def mixed_effect(d, K, minor):
    """weighted mixed model: transfer_auc ~ arm(paired vs samebatch) + (1|held), weights=minority."""
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return None
    sub = d[((d.strategy.isin(["coverage", "quantile"])) | (d.strategy == "samebatch")) & (d.K == K)].copy()
    sub["arm"] = np.where(sub.strategy == "samebatch", 0, 1)          # 1 = paired
    sub["w"] = sub.held.map(minor).astype(float)
    try:
        md = smf.mixedlm("transfer_auc ~ arm", sub, groups=sub["held"], re_formula="~1")
        mf = md.fit(reweight=False, method="lbfgs", disp=False)
        beta = mf.params["arm"]
        se = mf.bse["arm"]
        return beta, se, beta - 1.96 * se, beta + 1.96 * se, mf.pvalues["arm"]
    except Exception as e:
        return ("err", str(e))


def analyze(tissue):
    df = pd.read_parquet(RESULTS)
    d = df[df.tissue == tissue]
    if len(d) == 0:
        print(f"\n{tissue}: no rows yet"); return
    K = d.K.max()
    minor = minority_sizes(tissue)

    pr = d[(d.strategy.isin(["coverage", "quantile"])) & (d.K == K)].groupby("held").transfer_auc.mean()
    sb = d[(d.strategy == "samebatch") & (d.K == K)].groupby("held").transfer_auc.mean()
    nr = d[d.strategy == "harmony_noref"].groupby("held").transfer_auc.mean()
    held = sorted(set(pr.index) & set(sb.index) & set(nr.index))
    deconf = (pr - sb).loc[held]
    dataq = (sb - nr).loc[held]
    w = np.array([minor[h] for h in held], float)

    print(f"\n===== {tissue}  (n_held={len(held)}, K={K}) =====")
    tab = pd.DataFrame({"minority": w.astype(int), "DECONF": deconf.round(3).values,
                        "DATAQ": dataq.round(3).values}, index=held).sort_values("minority", ascending=False)
    print(tab.to_string())

    wm, lo, hi = wmean_ci(deconf.values, w)
    um = deconf.mean()
    npos = int((deconf > 0).sum())
    print(f"\nDECONF (de-confounding effect):")
    print(f"  unweighted mean = {um:.3f}   held>0 = {npos}/{len(held)}")
    print(f"  minority-WEIGHTED mean = {wm:.3f}   cluster-bootstrap 95% CI = [{lo:.3f}, {hi:.3f}]"
          f"   {'** excludes 0 **' if lo > 0 else '(crosses 0)'}")
    dm, dlo, dhi = wmean_ci(dataq.values, w)
    print(f"DATAQ (data-quantity control, want ~0):")
    print(f"  weighted mean = {dm:.3f}   95% CI = [{dlo:.3f}, {dhi:.3f}]   "
          f"{'crosses 0 (good)' if dlo < 0 < dhi else 'EXCLUDES 0 (data volume matters here)'}")

    mm = mixed_effect(d, K, minor)
    if mm and not (isinstance(mm, tuple) and mm[0] == "err"):
        beta, se, mlo, mhi, p = mm
        print(f"weighted mixed model (arm effect, held random intercept):")
        print(f"  beta = {beta:.3f} +/- {se:.3f}   95% CI = [{mlo:.3f}, {mhi:.3f}]   p = {p:.4f}")
    elif isinstance(mm, tuple):
        print(f"  mixed model failed: {mm[1]}")


if __name__ == "__main__":
    tissues = [sys.argv[1]] if len(sys.argv) > 1 else ["crc", "blood", "brain"]
    for t in tissues:
        analyze(t)
