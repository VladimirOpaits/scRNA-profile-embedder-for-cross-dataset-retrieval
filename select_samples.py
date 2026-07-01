import pandas as pd

SAMPLES = "data/hlca_samples.csv"
OUT = "data/selected_samples.csv"

DISEASES = ["normal", "lung adenocarcinoma"]
ASSAYS = ["10x 3' v2", "10x 3' v3", "Smart-seq2"]
MIN_CELLS = 500
N_PER_CELL = 8
SEED = 0


def pick(sub, n, seed):
    sub = sub.sort_values("dataset_id")
    picked = []
    for _, ds in sub.groupby("dataset_id", sort=False):
        picked.append(ds)
    inter = pd.concat(picked) if picked else sub
    inter = inter.sample(frac=1.0, random_state=seed)
    inter = inter.sort_values("dataset_id", kind="stable")
    return inter.head(n)


def main():
    samp = pd.read_csv(SAMPLES)
    samp = samp[samp["n_cells"] >= MIN_CELLS]
    samp["sample_id"] = (
        samp["donor_id"].astype(str) + " | " + samp["assay"].astype(str)
    )
    samp = samp.drop_duplicates("sample_id")

    rows = []
    for dis in DISEASES:
        for asy in ASSAYS:
            sub = samp[(samp["disease"] == dis) & (samp["assay"] == asy)]
            sel = pick(sub, N_PER_CELL, SEED)
            rows.append(sel)
    out = pd.concat(rows).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    print("selected", len(out), "samples ->", OUT)
    print("\n=== per (disease x assay) ===")
    print(pd.crosstab(out["assay"], out["disease"]).to_string())
    print("\n=== labs (datasets) per group ===")
    print(out.groupby(["disease", "assay"])["dataset_id"].nunique().to_string())
    print("\n=== cells per sample: total", int(out["n_cells"].sum()),
          "| capped@1000 est", int((out["n_cells"].clip(upper=1000)).sum()))
    print(out[["sample_id", "disease", "assay", "n_cells", "dataset_id"]].to_string(index=False))


if __name__ == "__main__":
    main()
