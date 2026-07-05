import numpy as np
import pandas as pd

PCA = "data/crc/pca50.npy"
CELLS = "data/crc/cells.parquet"
OUT = "data/crc/signatures.parquet"
D = 1024
SEED = 0


def median_sigma(X, rng, m=5000):
    idx = rng.choice(len(X), min(m, len(X)), replace=False)
    S = X[idx]
    d = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    return float(np.median(d[d > 0]))


def main():
    X = np.load(PCA).astype(np.float32)
    cells = pd.read_parquet(CELLS)
    rng = np.random.default_rng(SEED)

    sigma = median_sigma(X, rng)
    W = (rng.standard_normal((X.shape[1], D)) / sigma).astype(np.float32)
    b = (rng.uniform(0, 2 * np.pi, D)).astype(np.float32)
    print(f"median sigma={sigma:.3f} | RFF D={D}", flush=True)

    phi = np.sqrt(2.0 / D) * np.cos(X @ W + b)
    del X

    cells = cells.reset_index(drop=True)
    rows = []
    meta = cells.groupby("sample_id").first()
    for sid, idx in cells.groupby("sample_id").indices.items():
        sig = phi[idx].mean(0)
        m = meta.loc[sid]
        rows.append([sid, m.sample_type, m.study, m.assay, m.donor, len(idx)] + sig.tolist())
    cols = ["sample_id", "sample_type", "study", "assay", "donor", "n_cells"] + \
        [f"s{j}" for j in range(D)]
    sig_df = pd.DataFrame(rows, columns=cols)
    sig_df.to_parquet(OUT)
    print(f"saved {OUT} | samples={len(sig_df)} | "
          f"tumor/normal={dict(sig_df.sample_type.value_counts())}", flush=True)


if __name__ == "__main__":
    main()
