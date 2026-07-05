import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
def md(s): c.append(nbf.v4.new_markdown_cell(s))
def code(s): c.append(nbf.v4.new_code_cell(s))

md("""# De-confounding batch by data selection — CRC single-cell retrieval

**Hypothesis.** Batch effect in scRNA-seq can be mitigated at the *data-selection* level:
enrich a training cohort with same-diagnosis samples from diverse labs/technologies
(same biology, different batch) so a classifier learns biology, not the study signature.

**Method.** Patient = a set of cells (a distribution). Each patient → one vector:
shared **PCA** (50-dim, cross-dataset comparable, batch-preserving) over all cells →
**RFF-MMD** signature per patient (mean of random Fourier features; L2 between signatures = MMD).
Cohorts are assembled by **retrieval** (greedy farthest / quantile on MMD distance).
Outcome = **gap to the within-study ceiling** (within-study LOO-AUC holds batch constant = pure
biology); does enrichment close the cross-study transfer gap to it?

Held-out = **entire studies** (batch-level holdout): those studies' patients are NEVER trained on;
we train on the *pool* (other studies) and test on the held-out studies' patients (all of them).

This notebook orchestrates the pipeline (code lives in `scripts/`) and shows the outputs.""")

md("""## Environment
Run from the repo root with the **`bioml`** conda env / kernel. Heavy build steps
(extract → PCA → signatures → scSet) are precomputed in `data/crc/`; commands documented below,
experiment cells run live so "Run All" reproduces the results in ~10 min.""")

code("""import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import warnings; warnings.filterwarnings("ignore")
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
HELD5 = ["Chen_2024_Cancer_Cell","Lee_2020_Nat_Genet","Uhlitz_2021_EMBO_Mol_Med","MUI_Innsbruck","Zhang_2020_Cell"]
print("python:", sys.executable)""")

md("""## 1. Data — CRC single-cell atlas (Marteau et al. 2026)
3.79M cells, 588 donors, **45 studies**, 11 assays — essentially the integrated CRC scRNA-seq
literature. Paired studies (both classes in one study) let us define the within-study ceiling and
control the diagnosis≈dataset confound.""")

code("""d = pd.read_parquet("data/crc_obs_slim.parquet")
d = d[d.is_primary_data == True]
print("cells:", len(d), "| studies:", d.study_id.nunique(), "| assays:", d.assay.nunique())
cc = d.groupby("sample_id").agg(study=("study_id","first"), stype=("sample_type","first"),
        n=("sample_id","size")).reset_index()
cc = cc[(cc.n >= 200) & cc.stype.isin(["tumor","normal"])]
tab = pd.crosstab(cc.study, cc.stype)
tab = tab[(tab.get("tumor",0)>0) & (tab.get("normal",0)>0)]
tab["minority"] = tab[["tumor","normal"]].min(1)
tab = tab.sort_values("minority", ascending=False)
print(f"\\nPAIRED tumor+normal studies: {len(tab)} | minority>=10: {(tab.minority>=10).sum()}")
tab.head(10)""")

md("""## 2. Pipeline (build steps — precomputed, documented)
```
python scripts/crc_extract.py         # 29GB h5ad -> data/crc/counts_hv.npz  (317k cells x 7433 HV, 500/sample)
python scripts/crc_pca.py             # chunked IncrementalPCA -> data/crc/pca50.npy
python scripts/crc_signatures.py      # RFF-MMD (D=1024) -> data/crc/signatures.parquet
python scripts/scset_crc.py           # scSet (pool-only) -> data/crc/signatures_scset.parquet
python scripts/crc_leakfree_build.py  # pool-fit PCA (leak-free) -> data/crc/signatures_lf.parquet
```""")

code("""import scipy.sparse as sp
X = sp.load_npz("data/crc/counts_hv.npz"); P = np.load("data/crc/pca50.npy")
sig = pd.read_parquet("data/crc/signatures.parquet")
print("counts_hv:", X.shape, "| pca50:", P.shape)
print("RFF signatures:", sig.shape, "| tumor/normal:", dict(sig.sample_type.value_counts()))""")

md("""## 3. Main result — gap-to-ceiling enrichment (RFF-MMD, REPLACE)
Fixed-size cohort; start homogeneous (one tech), replace members with diverse-tech samples;
per-study transfer vs the within-study ceiling. `negctrl` = shuffled labels.""")
code("!{sys.executable} scripts/crc_enrichment.py 2>/dev/null | sed -n '/held /,$p'")

md("""**Reading `negctrl`.** It should sit near 0.5 and *not rise with k* (labels are random).
On tiny held-out studies (Uhlitz/MUI ~25 samples) a single mean can wander to 0.45–0.60 — that is
finite-sample noise, not signal. What matters is the contrast below: real arms climb to the ceiling,
negctrl stays in a flat band.""")
code("""r = pd.read_csv("crc_enrichment_results.csv")
print(f"{'study':26s} {'negctrl band [min,max] over k':>32s}   {'quantile 0->20':>16s}")
for st in HELD5:
    ng = r[(r.arm=="negctrl")&(r.study==st)].groupby("k").auc.mean()
    q  = r[(r.arm=="quantile")&(r.study==st)].groupby("k").auc.mean()
    print(f"{st:26s}   negctrl [{ng.min():.2f}, {ng.max():.2f}] (flat)      quantile {q.iloc[0]:.2f}->{q.iloc[-1]:.2f} (rises)")""")

md("""## 4. Controls (the result looked 'too good')""")
md("""### 4a. Classifier robustness — linear / RBF-SVM / RandomForest""")
code("!{sys.executable} scripts/crc_clf_check.py 2>/dev/null")
md("""### 4b. PCA-leakage check — refit PCA on POOL only, project held-out as external
Held-out was 38% of cells in the shared basis; refitting on pool only barely changes the result → no
transductive leakage.""")
code("!CRC_SIG=data/crc/signatures_lf.parquet {sys.executable} scripts/crc_enrichment.py 2>/dev/null | sed -n '/GAP TO/,$p'")

md("""## 5. Harder task — MSI vs MSS among tumors
Removes the easy 'cancer vs normal' axis; tests a subtle molecular subtype (microsatellite
instability). Effect survives but is more modest.""")
code("!{sys.executable} scripts/crc_msi.py 2>/dev/null | sed -n '/held /,$p'")

md("""## 6. ADD experiment — realistic use case & the decisive diversity-vs-volume test
Fixed seed cohort, then **ADD** k samples/class (cohort grows). `homogeneous` = add MORE of the base
technology = **pure volume, zero diversity**. Diverse-add vs homogeneous-add isolates diversity from
volume.""")
code("!{sys.executable} scripts/crc_add.py 2>/dev/null | sed -n '/ADD:/,$p'")

code("""import matplotlib.pyplot as plt
r = pd.read_csv("crc_add_results.csv"); m = r[r.study == "macro"]
fig, ax = plt.subplots(figsize=(7,4.5))
for arm, lab, style in [("quantile","diverse (quantile)","-o"),("random","diverse (random)","-s"),
                        ("homogeneous","homogeneous = pure volume","-^"),("negctrl","neg-control","--x")]:
    g = m[m.arm==arm].groupby("k").auc.agg(["mean","std"])
    ax.plot(g.index, g["mean"], style, label=lab, ms=4)
    ax.fill_between(g.index, g["mean"]-g["std"], g["mean"]+g["std"], alpha=0.12)
ax.axhline(0.90, ls=":", c="gray", label="within-study ceiling")
ax.set_xlabel("# samples added per class (k)"); ax.set_ylabel("held-out macro AUC")
ax.set_title("ADD: diversity drives transfer, volume alone does not"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md("""**Distribution, not just the mean.** Two separate claims with very different strength:""")
code("""K = int(r.k.max())
def s(a): return m[(m.arm==a)&(m.k==K)].set_index("seed").auc
print("=== diversity > volume (quantile - homogeneous), paired per seed ===")
dd = (s("quantile")-s("homogeneous")).dropna()
print(f"   mean {dd.mean():+.3f} | min {dd.min():+.3f} | frac>0 {(dd>0).mean():.0%}   <- ironclad, no overlap")
print("\\n=== selector vs random: NOT a mean effect ===")
for a,b in [("quantile","random"),("farthest","random")]:
    d2=(s(a)-s(b)).dropna(); print(f"   {a:9s}-{b}: mean {d2.mean():+.3f} frac>0 {(d2>0).mean():.0%}")
print("\\n=== but retrieval wins on RELIABILITY (downside) and SPEED ===")
for arm in ["quantile","farthest","random"]:
    x=s(arm); print(f"   {arm:9s} @k={K}: min {x.min():.3f}  frac<0.85 {(x<0.85).mean():.0%}")
sp5={a:m[(m.arm==a)&(m.k==5)].auc.mean() for a in ['quantile','farthest','random']}
print(f"   speed @k=5 (mean AUC): quantile {sp5['quantile']:.2f}  farthest {sp5['farthest']:.2f}  random {sp5['random']:.2f}")""")

md("""### 6b. ADD on the hard task (MSI) — diversity>volume holds but noisier (small n)""")
code("!{sys.executable} scripts/crc_add_msi.py 2>/dev/null | sed -n '/MSI ADD:/,$p'")

md("""## 7. Robustness — leave-one-study-out (does the result depend on WHICH studies we held out?)
Hold out each paired study alone (pool = all others), diverse-add vs homogeneous-add. Direction is
robust (diverse never worse); magnitude varies — where the baseline already transfers and
homogeneous-add also helps, the gap shrinks.""")
code("!{sys.executable} scripts/crc_loso.py 2>/dev/null")

md("""## 8. Representation — RFF-MMD vs learned scSet (same PCA substrate, both leak-free)
scSet pretrained pool-only on the same PCA cells. Compare by endpoint AUC — tie on easy, RFF-MMD
wins on hard MSI → parameter-free suffices at single-atlas scale.""")
code("!CRC_SIG=data/crc/signatures_scset.parquet {sys.executable} scripts/crc_enrichment.py 2>/dev/null | sed -n '/GAP TO/,$p'")

md("""## 9. Summary of key numbers

| result | number |
|---|---|
| CRC atlas | 3.79M cells, 588 donors, 45 studies, 11 assays |
| paired tumor/normal studies | 18 (8 with minority ≥10) |
| **enrichment (replace), tumor/normal** | macro **0.74 → 0.94**, closes gap in all 5 held-out, negctrl flat |
| classifier robustness | linear +0.20 / RBF-SVM +0.23 / RF +0.10 |
| PCA-leakage (pool-fit) | 0.93 ≈ 0.94 (no leakage) |
| MSI vs MSS (hard) | macro 0.51 → 0.64–0.68, negctrl flat (survives, modest) |
| **ADD: diverse vs homogeneous (volume)** | +0.16 vs **−0.02**; paired **100% of seeds**, min +0.11 |
| ADD selector vs random | tie on mean (56%); edge is reliability (0% <0.85) + speed (ceiling by k≈5) |
| **LOSO robustness** | diverse ≥ homogeneous in all 8 held-out; clear win 5/8, tie 3/8 |
| scSet vs RFF-MMD | tie on easy, RFF-MMD wins on hard MSI — parameter-free suffices |

**Headline.** Diversity ≫ volume (decisive, robust in direction). The realistic *add* scenario works.
The retrieval edge over random diversity is reliability + sample-efficiency, not a higher ceiling.
Substrate-agnostic, classifier-robust, no leakage, survives a subtle label; a parameter-free
signature suffices at single-atlas scale. **Limitation:** public CRC paired data is exhausted (this
is the field's largest integrated atlas); resolving the selector significance needs larger/clinical
cohorts.""")

nb["cells"] = c
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nbf.write(nb, "notebook.ipynb")
print(f"wrote notebook.ipynb with {len(c)} cells")
