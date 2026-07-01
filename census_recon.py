import os
import cellxgene_census
import pandas as pd

CENSUS_VERSION = "stable"
TISSUE = "lung"
OUT_DIR = "data"

OBS_COLS = [
    "assay", "disease", "dataset_id", "donor_id",
    "cell_type", "tissue", "suspension_type", "is_primary_data",
]


def load_obs():
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        obs = cellxgene_census.get_obs(
            census, "homo_sapiens",
            value_filter=(
                f"tissue_general == '{TISSUE}' and is_primary_data == True"
            ),
            column_names=OBS_COLS,
        )
    return obs


def sample_table(obs):
    g = obs.groupby(["dataset_id", "donor_id", "assay", "disease"], observed=True)
    return g.size().rename("n_cells").reset_index()


def disease_coverage(samp):
    rows = []
    for d, sub in samp.groupby("disease", observed=True):
        rows.append({
            "disease": d,
            "n_samples": len(sub),
            "n_donors": sub["donor_id"].nunique(),
            "n_datasets": sub["dataset_id"].nunique(),
            "n_assays": sub["assay"].nunique(),
            "assays": ", ".join(sorted(sub["assay"].unique())),
        })
    return pd.DataFrame(rows).sort_values("n_assays", ascending=False)


def assay_by_disease(samp, diseases):
    sub = samp[samp["disease"].isin(diseases)]
    return pd.crosstab(sub["assay"], sub["disease"], values=sub["donor_id"],
                       aggfunc="nunique").fillna(0).astype(int)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    obs = load_obs()
    print("total lung primary cells:", len(obs))
    print("obs columns:", list(obs.columns))

    samp = sample_table(obs)
    out_csv = os.path.join(OUT_DIR, "hlca_samples.csv")
    samp.to_csv(out_csv, index=False)
    print(f"\n=== sample-unit table saved to {out_csv} ===")
    print("samples:", len(samp), "| donors:", samp["donor_id"].nunique(),
          "| datasets:", samp["dataset_id"].nunique(),
          "| assays:", samp["assay"].nunique())

    cov = disease_coverage(samp)
    cov.to_csv(os.path.join(OUT_DIR, "disease_coverage.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    print("\n=== disease coverage (sorted by #assays) ===")
    print(cov.to_string(index=False))

    cands = cov[(cov["n_assays"] >= 3) & (cov["disease"] != "normal")]["disease"].tolist()
    keep = ["normal"] + cands[:6]
    print("\n=== donors per assay x disease (candidates spanning >=3 assays) ===")
    print(assay_by_disease(samp, keep).to_string())
