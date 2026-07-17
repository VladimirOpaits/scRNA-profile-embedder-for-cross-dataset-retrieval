"""What is actually inside our brain corpus: brain REGION and CELL TYPE, per study and per label.

The thing to look for: disease is coupled to region by study design (PD -> substantia nigra,
AD -> prefrontal cortex, ALS -> motor cortex). If we ignore it, "neurodegeneration vs control"
partly becomes "midbrain vs cortex" -- a biological confound riding on the study axis.
"""
import pyarrow as pa
import pandas as pd
import cellxgene_census as cc

CENSUS_VERSION = "2025-11-08"
BUF = 32 * 1024 ** 2
ASSAY = "10x 3' v3"
OUT = "data/brain/inventory.csv"
KEYS = ["dataset_id", "donor_id", "tissue", "cell_type", "disease"]


def main():
    plan = pd.read_parquet("data/brain/plan.parquet")
    want = set(zip(plan.dataset_id, plan.donor_id))

    ctx = cc.get_default_soma_context().replace(
        tiledb_config={"soma.init_buffer_bytes": BUF, "py.init_buffer_bytes": BUF})
    parts = []
    with cc.open_soma(census_version=CENSUS_VERSION, context=ctx) as census:
        obs = census["census_data"]["homo_sapiens"].obs
        vf = (f'is_primary_data == True and tissue_general == "brain" and assay == "{ASSAY}"')
        for tbl in obs.read(column_names=KEYS, value_filter=vf):
            g = tbl.group_by(KEYS).aggregate([(KEYS[0], "count")])
            parts.append(g.rename_columns(KEYS + ["n"]))
    d = (pa.concat_tables(parts).group_by(KEYS).aggregate([("n", "sum")])
           .rename_columns(KEYS + ["n"]).to_pandas())
    d = d[[k in want for k in zip(d.dataset_id, d.donor_id)]].copy()
    d["study"] = d.dataset_id.str[:8]
    d.to_csv(OUT, index=False)

    lab = plan.drop_duplicates("pid")[["dataset_id", "donor_id", "y"]]
    d = d.merge(lab, on=["dataset_id", "donor_id"], how="left")

    print("=== REGIONS (tissue) x label: cells ===")
    r = d.pivot_table(index="tissue", columns="y", values="n", aggfunc="sum", fill_value=0)
    r.columns = ["control", "neurodeg"][: r.shape[1]]
    r["donors"] = d.groupby("tissue").apply(lambda g: g.drop_duplicates(["dataset_id", "donor_id"]).shape[0])
    print(r.sort_values("donors", ascending=False).head(25).to_string())

    print("\n=== REGION x STUDY (donors), paired studies only ===")
    ps = d[d.study.isin(d.groupby("study").y.nunique().loc[lambda s: s > 1].index)]
    t = ps.drop_duplicates(["dataset_id", "donor_id", "tissue"])
    print(pd.crosstab(t.study, t.tissue).to_string())

    print("\n=== CELL TYPES (top 30 by cells) ===")
    c = d.groupby("cell_type").n.sum().sort_values(ascending=False)
    print(c.head(30).to_string())
    print(f"\ndistinct cell types: {d.cell_type.nunique()} | wrote {OUT}")


if __name__ == "__main__":
    main()
