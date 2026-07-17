"""Freeze the design for the brain replication: donor-level arms, restricted to ONE assay so the
batch axis is lab/protocol/year rather than technology (the harder, more realistic regime).

Disease axis = neurodegeneration vs control -- the axis that recurs across independent studies,
which is what a cross-study transfer task needs. Same memory discipline as census_paired_scan:
Arrow-only aggregation, small SOMA buffers, run under a cgroup cap.
"""
import pyarrow as pa
import pandas as pd
import cellxgene_census as cc

CENSUS_VERSION = "2025-11-08"
BUF = 64 * 1024 ** 2
MIN_CELLS = 200
ASSAY = "10x 3' v3"
OUT = "data/brain_arms.csv"
KEYS = ["dataset_id", "donor_id", "assay", "disease"]

NEURODEG = ["alzheimer", "dementia", "parkinson", "amyotrophic", "lewy body",
            "supranuclear", "pick disease", "huntington"]


def is_neurodeg(d):
    s = str(d).lower()
    return any(k in s for k in NEURODEG)


def main():
    ctx = cc.get_default_soma_context().replace(
        tiledb_config={"soma.init_buffer_bytes": BUF, "py.init_buffer_bytes": BUF})
    parts = []
    with cc.open_soma(census_version=CENSUS_VERSION, context=ctx) as census:
        obs = census["census_data"]["homo_sapiens"].obs
        vf = 'is_primary_data == True and tissue_general == "brain"'
        for tbl in obs.read(column_names=KEYS, value_filter=vf):
            g = tbl.group_by(KEYS).aggregate([(KEYS[0], "count")])
            parts.append(g.rename_columns(KEYS + ["n"]))
    d = (pa.concat_tables(parts).group_by(KEYS).aggregate([("n", "sum")])
           .rename_columns(KEYS + ["n"]).to_pandas())
    d = d[d.n >= MIN_CELLS].copy()
    d["study"] = d.dataset_id.str[:8]
    d["y"] = d.disease.map(is_neurodeg).astype(int)          # 1 = neurodegeneration
    d = d[d.y.eq(1) | d.disease.eq("normal")]                # drop unrelated diseases
    d.to_csv(OUT, index=False)
    print(f"brain donor-arms (>= {MIN_CELLS} cells, neurodeg or normal): {len(d)}")

    a = d[d.assay == ASSAY]
    t = a.groupby("study").agg(dis=("y", "sum"), n=("y", "size"))
    t["norm"] = t.n - t.dis
    t["minority"] = t[["dis", "norm"]].min(1)
    t["diseases"] = a[a.y == 1].groupby("study").disease.agg(
        lambda s: ";".join(sorted(set(s))[:2]))
    t = t.sort_values("minority", ascending=False)
    print(f"\n=== SINGLE ASSAY = {ASSAY} | batch is lab/protocol only ===")
    print(t.to_string())
    p = t[t.minority >= 5]
    print(f"\npaired studies (minority>=5): {len(p)}"
          f" | donors: {int(t.n.sum())} ({int(t.dis.sum())} neurodeg / {int(t.norm.sum())} control)")
    print(f"donors inside paired studies: {int(p.n.sum())}")


if __name__ == "__main__":
    main()
