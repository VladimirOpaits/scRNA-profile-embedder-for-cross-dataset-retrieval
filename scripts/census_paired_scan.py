"""Census-wide hunt for the ONE thing that gates our experiment: studies (dataset_id) holding
BOTH a disease and a control arm at the DONOR level. Not cells -- donors. A tumor full of
normal cells is not a normal sample.

MEMORY-SAFE BY CONSTRUCTION (an earlier version swapped the machine to death):
  * aggregation stays inside Arrow (group_by/aggregate). Never to_pylist() a batch -- that
    materializes millions of Python str/tuple objects, which is what blew up.
  * SOMA read buffers pinned small, so batch size is bounded rather than "whatever fits".
Peak RAM is one batch + the running aggregate (~1e5 unique donor-arms) => a few hundred MB.
Run it under a cgroup cap, NOT setrlimit -- RLIMIT_AS bounds virtual address space and TileDB
reserves a lot of it up front for its thread pool (fails with "concurrency level 0"):
  systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 python scripts/census_paired_scan.py

Also reports the SINGLE-ASSAY view: paired studies that stay paired inside one assay, where
batch = lab/protocol/year rather than technology -- the harder, more realistic regime.
"""
import pyarrow as pa
import pandas as pd
import cellxgene_census as cc

CENSUS_VERSION = "2025-11-08"
BUF = 64 * 1024 ** 2     # SOMA read buffer; keeps Arrow batches small
MIN_CELLS = 200          # per donor-arm, same floor as corpus_pull
MIN_MINORITY = 5         # usable paired study: >=5 donors in the smaller arm
OUT = "data/census_paired_scan.csv"
KEYS = ["tissue_general", "dataset_id", "donor_id", "assay", "disease"]


def fold(parts):
    """collapse a list of per-batch aggregates into one (stays in Arrow)."""
    t = pa.concat_tables(parts)
    return t.group_by(KEYS).aggregate([("n", "sum")]).rename_columns(KEYS + ["n"])


def stream_donor_arms():
    ctx = cc.get_default_soma_context().replace(
        tiledb_config={"soma.init_buffer_bytes": BUF, "py.init_buffer_bytes": BUF})
    parts, seen = [], 0
    with cc.open_soma(census_version=CENSUS_VERSION, context=ctx) as census:
        obs = census["census_data"]["homo_sapiens"].obs
        for tbl in obs.read(column_names=KEYS, value_filter="is_primary_data == True"):
            # count rows per donor-arm WITHOUT leaving Arrow
            g = tbl.group_by(KEYS).aggregate([(KEYS[0], "count")])
            parts.append(g.rename_columns(KEYS + ["n"]))
            seen += tbl.num_rows
            if len(parts) >= 64:                      # fold early so the list stays short
                parts = [fold(parts)]
                print(f"  {seen:,} cells | {parts[0].num_rows:,} donor-arms", flush=True)
    out = fold(parts)
    print(f"streamed {seen:,} cells | {out.num_rows:,} raw donor-arms", flush=True)
    return out.to_pandas().rename(columns={"tissue_general": "tissue"})


def main():
    d = stream_donor_arms()
    d = d[d.n >= MIN_CELLS].copy()
    d["is_normal"] = d.disease.eq("normal")
    print(f"donor-arms with >= {MIN_CELLS} cells: {len(d):,}", flush=True)

    rows = []
    for (tis, ds), g in d.groupby(["tissue", "dataset_id"]):
        n_norm = int(g.is_normal.sum())
        n_dis = int((~g.is_normal).sum())
        if n_norm == 0 or n_dis == 0:
            continue                                   # unpaired study -> useless to us
        best_a, best_m = None, 0                       # best assay that is ITSELF paired
        for a, ga in g.groupby("assay"):
            m = min(int(ga.is_normal.sum()), int((~ga.is_normal).sum()))
            if m > best_m:
                best_a, best_m = a, m
        rows.append(dict(tissue=tis, dataset_id=ds, n_disease=n_dis, n_normal=n_norm,
                         minority=min(n_dis, n_norm), n_assays=g.assay.nunique(),
                         best_assay=best_a, minority_1assay=best_m,
                         diseases=";".join(sorted(set(g.loc[~g.is_normal, "disease"]))[:3])))
    P = pd.DataFrame(rows).sort_values(["tissue", "minority"], ascending=[True, False])
    P.to_csv(OUT, index=False)

    agg = (P.groupby("tissue")
             .agg(paired_studies=("dataset_id", "nunique"),
                  usable=("minority", lambda s: int((s >= MIN_MINORITY).sum())),
                  usable_1assay=("minority_1assay", lambda s: int((s >= MIN_MINORITY).sum())),
                  donors_dis=("n_disease", "sum"), donors_norm=("n_normal", "sum"))
             .sort_values(["usable", "paired_studies"], ascending=False))
    print("\n=== TISSUE RANKING: studies with BOTH disease and control donors ===")
    print(f"usable        = paired studies with >={MIN_MINORITY} donors in the smaller arm")
    print("usable_1assay = same, but the pairing survives INSIDE a single assay\n")
    print(agg[agg.paired_studies >= 2].to_string())
    print(f"\nwrote {OUT}  ({len(P)} paired studies, {P.tissue.nunique()} tissues)")


if __name__ == "__main__":
    main()
