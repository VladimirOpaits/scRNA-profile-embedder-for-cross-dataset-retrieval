"""Design scan for the blood/COVID replication — pick the assay and confirm cross-study breadth
BEFORE the heavy counts pull (same discipline that saved us on brain).

Axis: COVID-19 vs normal. Single assay (10x 5' v1, the modal COVID assay) so batch = lab/protocol,
not technology — comparable to the brain design. COVID is a COMPOSITIONAL shift (lymphopenia,
monocyte influx), so unlike dementia the biology should be strongly readable — that is the whole
point of pivoting here ([[brain-boundary-result]]).

Reports: paired COVID/normal studies (held candidates) and the size of the control+COVID reservoir
(the enrichment fuel). Arrow-only aggregation, small SOMA buffers, run under a cgroup cap.
"""
import pyarrow as pa
import pandas as pd
import cellxgene_census as cc

CENSUS_VERSION = "2025-11-08"
BUF = 64 * 1024 ** 2
MIN_CELLS = 200
ASSAY = "10x 5' v1"
OUT = "data/blood_arms.csv"
KEYS = ["dataset_id", "donor_id", "assay", "disease"]


def is_covid(d):
    return "covid" in str(d).lower()


def main():
    ctx = cc.get_default_soma_context().replace(
        tiledb_config={"soma.init_buffer_bytes": BUF, "py.init_buffer_bytes": BUF})
    parts = []
    with cc.open_soma(census_version=CENSUS_VERSION, context=ctx) as census:
        obs = census["census_data"]["homo_sapiens"].obs
        vf = 'is_primary_data == True and tissue_general == "blood"'
        for tbl in obs.read(column_names=KEYS, value_filter=vf):
            g = tbl.group_by(KEYS).aggregate([(KEYS[0], "count")])
            parts.append(g.rename_columns(KEYS + ["n"]))
    d = (pa.concat_tables(parts).group_by(KEYS).aggregate([("n", "sum")])
           .rename_columns(KEYS + ["n"]).to_pandas())
    d = d[d.n >= MIN_CELLS].copy()
    d["study"] = d.dataset_id.str[:8]

    # keep the COVID-vs-normal labeled set on the chosen assay
    a = d[d.assay == ASSAY].copy()
    a = a[a.disease.map(is_covid) | a.disease.eq("normal")]
    a["y"] = a.disease.map(is_covid).astype(int)
    a.to_csv(OUT, index=False)

    t = a.groupby("study").agg(covid=("y", "sum"), n=("y", "size"))
    t["normal"] = t.n - t.covid
    t["minority"] = t[["covid", "normal"]].min(1)
    t = t.sort_values("minority", ascending=False)
    print(f"=== blood, assay={ASSAY}, COVID vs normal ===")
    print(f"donor-arms: {len(a)} | {int(a.y.sum())} COVID / {int((1 - a.y).sum())} normal "
          f"| {a.study.nunique()} studies")
    paired = t[t.minority >= 5]
    print(f"\npaired studies (minority>=5, held candidates): {len(paired)}")
    print(paired.to_string())
    pool = t.drop(index=paired.index)
    print(f"\nreservoir (pool) studies: {len(pool)} | "
          f"{int(pool.covid.sum())} COVID / {int(pool.normal.sum())} normal donors")
    print(f"donors inside paired studies: {int(paired.n.sum())} | total: {len(a)}")


if __name__ == "__main__":
    main()
