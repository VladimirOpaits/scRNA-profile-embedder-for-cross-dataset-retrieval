import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import lancedb
from datasets import load_from_disk
from geneformer import EmbExtractor

MODEL = ("/home/vlad/.cache/huggingface/hub/models--ctheodoris--Geneformer/"
         "snapshots/04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5")
LABELS = ["sample_id", "technology", "disease", "cell_type"]
FBS = int(os.environ.get("FBS", "8"))
MAXLEN = int(os.environ.get("MAXLEN", "2048"))
SRC = "tokenized/hlca.dataset"
TRUNC = "tokenized/hlca_trunc.dataset"


def prepare():
    if os.path.exists(TRUNC):
        return TRUNC
    d = load_from_disk(SRC)
    if max(d.select(range(min(500, d.num_rows)))["length"]) <= MAXLEN:
        return SRC

    def trunc(ex):
        ids = ex["input_ids"][:MAXLEN]
        ex["input_ids"] = ids
        ex["length"] = len(ids)
        return ex

    d = d.map(trunc, num_proc=4)
    d.save_to_disk(TRUNC)
    print(f"truncated dataset to <= {MAXLEN} -> {TRUNC}")
    return TRUNC


def main():
    os.makedirs("emb", exist_ok=True)
    src = prepare()
    embex = EmbExtractor(
        model_type="Pretrained", num_classes=0, emb_mode="cls",
        max_ncells=None, emb_layer=-1, emb_label=LABELS,
        forward_batch_size=FBS, nproc=4, model_version="V2",
    )
    df = embex.extract_embs(
        MODEL, src, "emb", "hlca",
        output_torch_embs=False,
    )

    emb_cols = [c for c in df.columns if c not in LABELS]
    X = df[emb_cols].to_numpy(dtype=np.float32)
    meta = df[LABELS].reset_index(drop=True)
    print("embeddings:", X.shape)

    rows = []
    for i in range(len(df)):
        sid = str(meta["sample_id"][i])
        rows.append({
            "cell_id": f"hlca:{i}",
            "dataset": "hlca",
            "technology": str(meta["technology"][i]),
            "donor": sid.split(" | ")[0],
            "cell_type": str(meta["cell_type"][i]),
            "disease": str(meta["disease"][i]),
            "sample_id": sid,
            "vector": X[i].tolist(),
        })

    db = lancedb.connect("vectors")
    db.create_table("cells", data=rows, mode="overwrite")
    print("LanceDB built: vectors/cells.lance |", len(rows), "cells, dim", X.shape[1])


if __name__ == "__main__":
    main()
