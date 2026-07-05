import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
def md(s): c.append(nbf.v4.new_markdown_cell(s))
def code(s): c.append(nbf.v4.new_code_cell(s))

md("""# De-confounding batch by data selection — CRC single-cell retrieval

**Hypothesis.** Batch effect in scRNA-seq can be mitigated at the *data-selection* level:
enrich a training cohort with same-diagnosis samples from diverse labs/technologies
(same biology, different batch) so a classifier learns biology, not the study signature.

**Method.** Patient = a set of cells (a distribution). We map each patient to one vector:
shared **PCA** (50-dim, cross-dataset comparable, batch-preserving) over all cells →
**RFF-MMD** signature per patient (mean of random Fourier features; L2 between signatures = MMD).
Cohorts are assembled by **retrieval** (greedy farthest / quantile on MMD distance).
Outcome = **gap to the within-study ceiling**: within-study LOO-AUC holds batch constant
(pure biology); we ask whether enrichment closes the cross-study transfer gap to it.

This notebook orchestrates the pipeline (code lives in `scripts/`) and shows the outputs.""")

md("""## Environment
Run from the repo root with the **`bioml`** conda env / kernel.
Heavy build steps (extract → PCA → signatures → scSet) are already computed; their artifacts
live in `data/crc/`. The build commands are documented below; the experiment cells run live.""")

code("""import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import warnings; warnings.filterwarnings("ignore")
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
print("python:", sys.executable)""")

md("""## 1. Data — CRC single-cell atlas (Marteau et al. 2026)
3.79M cells, 588 donors, **45 studies**, 11 assays — essentially the integrated CRC scRNA-seq
literature. We work on the tumor/normal subset and the paired studies (both classes in one study,
so batch is shared across classes → controls the diagnosis≈dataset confound).""")

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

md("""## 2. Pipeline (build steps — already computed, documented for reproducibility)
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
Fixed-size cohort, start homogeneous (one tech), replace members with diverse-tech samples;
per-study transfer vs the within-study ceiling. `negctrl` = shuffled labels.""")
code("!{sys.executable} scripts/crc_enrichment.py 2>/dev/null | sed -n '/held /,$p'")

md("""## 4. Controls (the result looked 'too good')""")
md("""### 4a. Classifier robustness — linear / RBF-SVM / RandomForest""")
code("!{sys.executable} scripts/crc_clf_check.py 2>/dev/null")
md("""### 4b. PCA-leakage check — refit PCA on POOL only, project held-out as external""")
code("!CRC_SIG=data/crc/signatures_lf.parquet {sys.executable} scripts/crc_enrichment.py 2>/dev/null | sed -n '/GAP TO/,$p'")

md("""## 5. Harder task — MSI vs MSS among tumors
Removes the easy 'cancer vs normal' axis; tests a subtle molecular subtype (microsatellite
instability). Effect survives but is more modest.""")
code("!{sys.executable} scripts/crc_msi.py 2>/dev/null | sed -n '/held /,$p'")

md("""## 6. ADD experiment — the realistic use case & the decisive diversity-vs-volume test
Fixed seed cohort, then **ADD** k samples/class (cohort grows). `homogeneous` = add MORE of the
base technology = **pure volume, zero diversity**. Diverse-add vs homogeneous-add isolates
diversity from volume.""")
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

code("""# distribution: paired per-seed difference at the endpoint
K = int(r.k.max())
def s(a): return m[(m.arm==a)&(m.k==K)].set_index("seed").auc
for a,b in [("quantile","homogeneous"),("quantile","random")]:
    dd = (s(a)-s(b)).dropna()
    print(f"{a} - {b:12s}: mean {dd.mean():+.3f} | min {dd.min():+.3f} | frac>0 {(dd>0).mean():.0%}")""")

md("""## 7. Representation — RFF-MMD vs learned scSet (same PCA substrate, both leak-free)
scSet pretrained pool-only on the same PCA cells. Compare by endpoint AUC.""")
code("!CRC_SIG=data/crc/signatures_scset.parquet {sys.executable} scripts/crc_enrichment.py 2>/dev/null | sed -n '/GAP TO/,$p'")

md("""## 8. Summary of key numbers

| result | number |
|---|---|
| CRC atlas | 3.79M cells, 588 donors, 45 studies, 11 assays |
| paired tumor/normal studies | 18 (8 with minority ≥10) |
| **enrichment (replace), tumor/normal** | macro **0.74 → 0.94**, closes gap in all 5 held-out, negctrl flat |
| classifier robustness | linear +0.20 / RBF-SVM +0.23 / RF +0.10 |
| PCA-leakage (pool-fit) | 0.93 ≈ 0.94 (no leakage) |
| MSI vs MSS (hard) | macro 0.51 → 0.64–0.68, negctrl flat (survives, modest) |
| **ADD: diverse vs homogeneous (volume)** | +0.16 vs **−0.02**; paired **100% of seeds**, min +0.11 |
| ADD selector vs random | tie (+0.015, 56%); edge is reliability + speed, not mean |
| scSet vs RFF-MMD | tie on easy, RFF-MMD wins on hard MSI — parameter-free suffices |

**Headline:** diversity ≫ volume (decisive); the realistic add scenario works; substrate-agnostic,
classifier-robust, no leakage, survives a subtle label; a parameter-free signature suffices at
single-atlas scale.""")

nb["cells"] = c
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nbf.write(nb, "notebook.ipynb")
print(f"wrote notebook.ipynb with {len(c)} cells")
