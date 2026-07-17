"""Per-cell immune compartments for the blood/COVID corpus.

COVID is a COMPOSITIONAL disease: lymphopenia (T/NK down), monocyte influx, plasmablast
expansion, interferon-high states. So the compartment map decides what the per-type witness and
the leave-one-type-out ablation can see. We collapse the fine Census cell_type ontology into the
immune compartments blood biology actually distinguishes -- the analogue of the CRC coarse types.

Aligned to cells.parquet row order (NOT plan.parquet): pull_counts sorts cells by soma_joinid
before reading X, so cells.parquet / pca50.npy share THAT order. Building types off plan.parquet
would misalign every cell's label against its coordinates (the bug caught on brain).

Rules are ordered: first match wins, so specific patterns precede generic ones.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import cellxgene_census as cc

CENSUS_VERSION = "2025-11-08"
BUF = 64 * 1024 ** 2
ASSAYS = ["10x 5' v1", "10x 5' v2", "10x 5' transcription profiling"]
CELLS = "data/blood/cells.parquet"
OUT = "data/blood/cell_types.parquet"
CTCOL = "cell_type_coarse_blood"

RULES = [
    ("Plasma", ["plasma", "plasmablast"]),
    ("B", ["b cell", "naive b", "memory b", "b-1", "germinal"]),
    ("NK", ["natural killer", "nk cell", " nk "]),
    ("CD4T", ["cd4", "t-helper", "helper t", "regulatory t"]),
    ("CD8T", ["cd8", "cytotoxic t"]),
    ("T", ["t cell", "thymocyte", "mucosal", "gamma-delta", "double negative",
           "double-positive"]),
    ("Monocyte", ["monocyte", "cd14", "cd16"]),
    ("DC", ["dendritic", "plasmacytoid", "pdc", "cdc"]),
    ("Granulocyte", ["neutrophil", "eosinophil", "basophil", "granulocyte", "mast"]),
    ("Megakaryocyte", ["megakaryocyte", "platelet"]),
    ("Erythroid", ["erythrocyte", "erythroid", "reticulocyte"]),
    ("Progenitor", ["progenitor", "stem cell", "hematopoietic", "hspc", "blast"]),
    ("Myeloid", ["macrophage", "myeloid", "phagocyte"]),
    ("Lymphocyte", ["lymphocyte", "leukocyte", "mononuclear"]),   # generic, late
]


def coarse(ct):
    s = str(ct).lower()
    for name, keys in RULES:
        if any(k in s for k in keys):
            return name
    return "Other"


def main():
    cells = pd.read_parquet(CELLS)
    print(f"corpus cells: {len(cells):,}", flush=True)
    ctx = cc.get_default_soma_context().replace(
        tiledb_config={"soma.init_buffer_bytes": BUF, "py.init_buffer_bytes": BUF})
    parts = []
    with cc.open_soma(census_version=CENSUS_VERSION, context=ctx) as census:
        obs = census["census_data"]["homo_sapiens"].obs
        alist = ", ".join(f'"{a}"' for a in ASSAYS)
        vf = f'is_primary_data == True and tissue_general == "blood" and assay in [{alist}]'
        for tbl in obs.read(column_names=["soma_joinid", "cell_type"], value_filter=vf):
            parts.append(tbl)
    ct = pa.concat_tables(parts).to_pandas()
    print(f"census blood 5'-family obs: {len(ct):,}", flush=True)

    ct = ct.drop_duplicates("soma_joinid").set_index("soma_joinid")
    out = cells[["soma_joinid", "pid", "study", "y"]].copy()
    out["cell_type"] = ct.loc[out.soma_joinid.to_numpy()].cell_type.to_numpy()
    assert out.cell_type.notna().all(), "cell_type join incomplete"
    out[CTCOL] = out.cell_type.map(coarse)
    out.to_parquet(OUT)

    print("\n=== immune compartments ===")
    v = out[CTCOL].value_counts()
    print(pd.DataFrame({"cells": v, "pct": (100 * v / len(out)).round(1)}).to_string())
    other = out.loc[out[CTCOL] == "Other", "cell_type"].value_counts().head(12)
    if len(other):
        print("\ntop unmatched labels (-> Other):")
        print(other.to_string())
    print(f"\ndistinct raw cell types: {out.cell_type.nunique()} | wrote {OUT}")


if __name__ == "__main__":
    main()
