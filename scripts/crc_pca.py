import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import IncrementalPCA

COUNTS = "data/crc/counts_hv.npz"
OUT = "data/crc/pca50.npy"
NCOMP = 50
CHUNK = 10_000
TARGET = 1e4
CLIP = 10.0


def main():
    X = sp.load_npz(COUNTS).tocsr().astype(np.float32)
    n = X.shape[0]
    print(f"counts {X.shape} nnz={X.nnz}", flush=True)

    rs = np.asarray(X.sum(1)).ravel()
    rs[rs == 0] = 1.0
    X = X.multiply((TARGET / rs)[:, None]).tocsr()
    X.data = np.log1p(X.data)
    print("normalized + log1p", flush=True)

    mean = np.asarray(X.mean(0)).ravel()
    sq = X.copy()
    sq.data **= 2
    var = np.asarray(sq.mean(0)).ravel() - mean ** 2
    del sq
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    mean = mean.astype(np.float32)

    ipca = IncrementalPCA(n_components=NCOMP)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        z = X[a:b].toarray().astype(np.float32)
        z -= mean
        z /= std
        np.clip(z, -CLIP, CLIP, out=z)
        ipca.partial_fit(z)
        print(f"  fit {b}/{n}", flush=True)
    print(f"fit done | explained var (top5): "
          f"{np.round(ipca.explained_variance_ratio_[:5], 3)}", flush=True)

    out = np.empty((n, NCOMP), dtype=np.float32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        z = X[a:b].toarray()
        z -= mean
        z /= std
        np.clip(z, -CLIP, CLIP, out=z)
        out[a:b] = ipca.transform(z)
    np.save(OUT, out)
    print(f"saved {OUT} shape={out.shape} "
          f"cumvar={ipca.explained_variance_ratio_.sum():.3f}", flush=True)


if __name__ == "__main__":
    main()
