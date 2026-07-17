"""Align per-cell cell_type labels to our extracted matrix (counts_hv/pca50 row order).
cells.csv carries `row` = original h5ad obs index (matrix rows are sorted by it). We
read the atlas-harmonized coarse cell-type (12 categories, consistent across studies)
plus middle/fine, index by `row`, and save aligned to data/crc/cell_types.parquet.
"""
import h5py
import numpy as np
import pandas as pd

H5 = "4a8b9568-965e-46b8-a427-baab6bf018e5.h5ad"
CELLS = "data/crc/cells.csv"
OUT = "data/crc/cell_types.parquet"
COLS = ["cell_type_coarse_crc_atlas", "cell_type_middle_crc_atlas", "cell_type"]


def read_cat(obs, key, rows):
    g = obs[key]
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
    codes = g["codes"][:]                      # full-length int codes
    sub = codes[rows]
    out = np.where(sub >= 0, cats[np.clip(sub, 0, len(cats) - 1)], "NA")
    return out.astype(object)


def main():
    cells = pd.read_csv(CELLS)
    rows = cells["row"].to_numpy()
    print(f"cells: {len(cells)} | row range [{rows.min()}, {rows.max()}]")
    f = h5py.File(H5, "r")
    obs = f["obs"]
    df = pd.DataFrame({"sample_id": cells["sample_id"].to_numpy()})
    for c in COLS:
        df[c] = read_cat(obs, c, rows)
        print(f"  {c}: {df[c].nunique()} types | top:",
              dict(pd.Series(df[c]).value_counts().head(5)))
    f.close()
    df.to_parquet(OUT)
    print(f"saved {OUT} ({len(df)} cells, aligned to pca50.npy row order)")


if __name__ == "__main__":
    main()
