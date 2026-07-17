"""Per-cell cell types for the brain corpus + coarsening into compartments.

Landmarks are allocated per cell type (stratified geosketch) and the causal ablation removes one
type at a time, so the granularity of this map decides what those experiments can even say. Census
cell_type is a fine ontology (hundreds of labels, inconsistent across labs); we collapse it into
the compartments that brain biology actually distinguishes -- the analogue of the 12 coarse CRC
types.

Compartments and why each is here:
  ExcitatoryNeuron  glutamatergic projection neurons; the cortical bulk
  InhibitoryNeuron  GABAergic interneurons (PVALB/SST/VIP/LAMP5)
  Dopaminergic      midbrain DA neurons -- the cells that die in Parkinson's
  Oligodendrocyte   myelinating; numerically dominant in white matter/midbrain. Prime suspect for
                    carrying batch (nuclei quality / dissociation), i.e. the "Epithelial" of CRC
  OPC               oligodendrocyte precursors
  Astrocyte         reactive in every pathology
  Microglia         brain-resident immune. Prime suspect for carrying transferable disease biology
  Vascular          endothelial + pericyte + VLMC/fibroblast
  Ependymal         ciliated, ventricle lining
  Other             anything unmatched (kept, never silently dropped)

Rules are ordered: the FIRST match wins, so specific patterns must precede generic ones
("dopaminergic neuron" before the generic neuron catch-alls).
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import cellxgene_census as cc

CENSUS_VERSION = "2025-11-08"
BUF = 64 * 1024 ** 2
ASSAY = "10x 3' v3"
CELLS = "data/brain/cells.parquet"     # NOT plan.parquet: pull_counts sorts cells by soma_joinid
OUT = "data/brain/cell_types.parquet"  # before reading X, so cells.parquet/pca50.npy share THAT
                                       # row order. Building types off plan.parquet misaligns
                                       # every cell's label against its coordinates.
CTCOL = "cell_type_coarse_brain"

RULES = [
    ("Dopaminergic", ["dopaminergic"]),
    ("Microglia", ["microglia", "microglial"]),
    ("Oligodendrocyte", ["oligodendrocyte"]),          # after 'precursor' check below
    ("OPC", ["oligodendrocyte precursor", "opc"]),
    ("Astrocyte", ["astrocyte"]),
    ("Ependymal", ["ependymal"]),
    ("Vascular", ["endothelial", "pericyte", "vascular", "leptomeningeal", "fibroblast",
                  "smooth muscle", "vlmc"]),
    ("InhibitoryNeuron", ["gabaergic", "inhibitory", "interneuron", "pvalb", "sst ",
                          "vip ", "lamp5", "sncg", "chandelier", "medium spiny"]),
    ("ExcitatoryNeuron", ["glutamatergic", "excitatory", "pyramidal", "granule",
                          "intratelencephalic", "near-projecting", "corticothalamic",
                          "extratelencephalic", "l2/3", "l4", "l5", "l6"]),
    ("Immune", ["macrophage", "t cell", "b cell", "lymphocyte", "monocyte", "leukocyte"]),
    ("Neuron", ["neuron"]),                            # generic neuron catch-all, LAST of neurons
]


def coarse(ct):
    s = str(ct).lower()
    if "precursor" in s and "oligodendrocyte" in s:    # OPC must beat 'oligodendrocyte'
        return "OPC"
    for name, keys in RULES:
        if any(k in s for k in keys):
            return name
    return "Other"


def main():
    plan = pd.read_parquet(CELLS)
    want = plan.soma_joinid.to_numpy(np.int64)
    print(f"corpus cells: {len(want):,}", flush=True)

    ctx = cc.get_default_soma_context().replace(
        tiledb_config={"soma.init_buffer_bytes": BUF, "py.init_buffer_bytes": BUF})
    parts = []
    with cc.open_soma(census_version=CENSUS_VERSION, context=ctx) as census:
        obs = census["census_data"]["homo_sapiens"].obs
        vf = f'is_primary_data == True and tissue_general == "brain" and assay == "{ASSAY}"'
        for tbl in obs.read(column_names=["soma_joinid", "cell_type", "tissue"], value_filter=vf):
            parts.append(tbl)
    ct = pa.concat_tables(parts).to_pandas()
    print(f"census brain {ASSAY} obs: {len(ct):,}", flush=True)

    out = plan[["soma_joinid", "pid", "study", "y"]].merge(ct, on="soma_joinid", how="left")
    assert len(out) == len(plan) and out.cell_type.notna().all(), "cell_type join incomplete"
    out[CTCOL] = out.cell_type.map(coarse)
    out.to_parquet(OUT)

    print("\n=== coarse compartments ===")
    v = out[CTCOL].value_counts()
    print(pd.DataFrame({"cells": v, "pct": (100 * v / len(out)).round(1)}).to_string())
    other = out.loc[out[CTCOL] == "Other", "cell_type"].value_counts().head(10)
    if len(other):
        print("\ntop unmatched labels (-> Other):")
        print(other.to_string())
    print(f"\nregions (tissue): {out.tissue.nunique()}")
    print(out.tissue.value_counts().head(12).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
