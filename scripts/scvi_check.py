import os
import numpy as np
import pandas as pd
from selection import cosine_distance_matrix

DIR = "data/scvi"


def fname(sid):
    return sid.replace(" ", "_").replace("|", "-").replace("/", "_") + ".npy"


def main():
    meta = pd.read_csv(os.path.join(DIR, "meta.csv"))
    pb = {}
    for r in meta.itertuples():
        pb[r.sample_id] = np.load(os.path.join(DIR, fname(r.sample_id))).mean(0)
    samples = list(pb)
    M = np.vstack([pb[s] for s in samples])
    m = meta.set_index("sample_id")
    tech = np.array([m.loc[s, "technology"] for s in samples])
    dis = np.array([m.loc[s, "disease"] for s in samples])

    for name, D in [("cosine", cosine_distance_matrix(M)),
                    ("l2", l2_matrix(M))]:
        print(f"\n=== {name} on scVI pseudobulk ===")
        report(D, tech, dis)


def l2_matrix(M):
    d = np.linalg.norm(M[:, None] - M[None], axis=-1)
    return d


def mean_pair(D, labels, same):
    n = len(labels)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            if (labels[i] == labels[j]) == same:
                vals.append(D[i, j])
    return np.mean(vals) if vals else np.nan


def report(D, tech, dis):
    st = mean_pair(D, tech, True)
    dt = mean_pair(D, tech, False)
    sd = mean_pair(D, dis, True)
    dd = mean_pair(D, dis, False)
    print(f"technology: same={st:.4f} diff={dt:.4f} gap={dt-st:+.4f}")
    print(f"diagnosis : same={sd:.4f} diff={dd:.4f} gap={dd-sd:+.4f}")
    print(f"tech-gap / bio-gap = {(dt-st)/(dd-sd):.2f}x "
          f"(>1 => technology dominates)")


if __name__ == "__main__":
    main()
