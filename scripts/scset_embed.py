import os
import numpy as np
import pandas as pd
import torch

from scset_model import build_encoder

CELLDIR = "data/scvi_corpus/cells"
META = "data/scvi_corpus/meta.csv"
CKPT = "data/scset/pretrained.pt"
OUT = "data/scset/patient_emb.parquet"
MAX_CELLS = 2048


@torch.no_grad()
def embed_patient(enc, cells, dev):
    if cells.shape[0] > MAX_CELLS:
        idx = np.random.default_rng(0).choice(cells.shape[0], MAX_CELLS, False)
        cells = cells[idx]
    x = torch.from_numpy(cells).float().unsqueeze(0).to(dev)
    return enc(x).squeeze(0).cpu().numpy()


def main():
    ck = torch.load(CKPT, map_location="cpu")
    dev = torch.device("cuda")
    enc = build_encoder(ck["encoder_kind"], ck["sample_dim"], ck["model_dim"])
    enc.load_state_dict(ck["encoder"])
    enc.to(dev).eval()

    meta = pd.read_csv(META)
    rows, vecs = [], []
    for r in meta.itertuples():
        f = os.path.join(CELLDIR, r.pid + ".npy")
        if not os.path.exists(f):
            continue
        vecs.append(embed_patient(enc, np.load(f), dev))
        rows.append({"pid": r.pid, "technology": r.assay, "disease": r.disease})
    V = np.vstack(vecs).astype(np.float32)
    df = pd.DataFrame(rows)
    for j in range(V.shape[1]):
        df[f"e{j}"] = V[:, j]
    df.to_parquet(OUT)
    print("wrote", OUT, "| patients:", len(df), "| dim:", V.shape[1])


if __name__ == "__main__":
    main()
