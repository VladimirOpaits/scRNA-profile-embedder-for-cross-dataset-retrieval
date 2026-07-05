import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scset_model import build_encoder, ConditionalDenoisingMLP, DiffusionProcess

PCA = "data/crc/pca50.npy"
CELLS = "data/crc/cells.parquet"
OUT = "data/crc/signatures_scset.parquet"
CKPT = "data/crc/scset_crc.pt"
HELD = ["Chen_2024_Cancer_Cell", "Lee_2020_Nat_Genet", "Uhlitz_2021_EMBO_Mol_Med",
        "MUI_Innsbruck", "Zhang_2020_Cell"]
ENCODER = "cell_transformer"
MODEL_DIM = 256
SAMPLE_DIM = 50
MIN_CELLS = 128
MAX_ANCHOR = 512
NUM_TARGET = 16
STEPS_PER_SAMPLE = 8
NUM_TIMESTEPS = 1000
BATCH = 16
EPOCHS = 200
LR = 1e-3
CLIP = 0.1
VAL_FRAC = 0.1
SEED = 0
MAX_EMB = 2048


class BagDS(Dataset):
    def __init__(self, arrs):
        self.arrs = arrs

    def __len__(self):
        return len(self.arrs)

    def __getitem__(self, i):
        c = torch.from_numpy(self.arrs[i]).float()
        n = min(MAX_ANCHOR + NUM_TARGET, c.shape[0])
        idx = torch.randperm(c.shape[0])[:n]
        c = c[idx]
        return {"target": c[:NUM_TARGET], "anchor": c[NUM_TARGET:]}

    @staticmethod
    def collate(batch):
        mc = max(b["anchor"].shape[0] for b in batch)
        an, mask, tg = [], [], []
        for b in batch:
            a = b["anchor"]
            pad = mc - a.shape[0]
            an.append(torch.cat([a, torch.zeros(pad, a.shape[1])], 0))
            m = torch.zeros(mc, dtype=torch.bool)
            m[a.shape[0]:] = True
            mask.append(m)
            tg.append(b["target"])
        return {"anchor": torch.stack(an), "anchor_mask": torch.stack(mask),
                "target": torch.stack(tg)}


def run_epoch(dl, enc, den, dp, opt, dev, train):
    enc.train(train)
    den.train(train)
    tot = 0.0
    for batch in dl:
        a = batch["anchor"].to(dev)
        m = batch["anchor_mask"].to(dev)
        t = batch["target"].to(dev)
        emb = enc(a, X_mask=m)
        emb = emb.unsqueeze(1).expand(-1, NUM_TARGET * STEPS_PER_SAMPLE, -1)
        emb = emb.reshape(-1, emb.shape[-1])
        t = t.unsqueeze(2).expand(-1, -1, STEPS_PER_SAMPLE, -1)
        t = t.reshape(-1, t.shape[-1])
        tidx = torch.randint(0, NUM_TIMESTEPS, (t.shape[0],), device=dev)
        loss = dp.p_loss(den, t, tidx, emb)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), CLIP)
            torch.nn.utils.clip_grad_norm_(den.parameters(), CLIP)
            opt.step()
        tot += loss.item()
    return tot / len(dl)


@torch.no_grad()
def embed(enc, cells, dev):
    if cells.shape[0] > MAX_EMB:
        idx = np.random.default_rng(0).choice(cells.shape[0], MAX_EMB, False)
        cells = cells[idx]
    x = torch.from_numpy(cells).float().unsqueeze(0).to(dev)
    return enc(x).squeeze(0).cpu().numpy()


def main():
    P = np.load(PCA).astype(np.float32)
    cells = pd.read_parquet(CELLS).reset_index(drop=True)
    groups = cells.groupby("sample_id").indices
    meta = cells.groupby("sample_id").first()

    sample_ids = list(groups.keys())
    arrs = {sid: P[np.sort(groups[sid])] for sid in sample_ids}

    pool_ids = [s for s in sample_ids
                if meta.loc[s, "study"] not in HELD and arrs[s].shape[0] >= MIN_CELLS]
    rng = np.random.default_rng(SEED)
    rng.shuffle(pool_ids)
    nval = max(1, int(len(pool_ids) * VAL_FRAC))
    tr = [arrs[s] for s in pool_ids[nval:]]
    va = [arrs[s] for s in pool_ids[:nval]]
    print(f"pool patients train/val: {len(tr)}/{len(va)} | total samples {len(sample_ids)}", flush=True)

    dev = torch.device("cuda")
    enc = build_encoder(ENCODER, SAMPLE_DIM, MODEL_DIM).to(dev)
    den = ConditionalDenoisingMLP(SAMPLE_DIM, MODEL_DIM).to(dev)
    dp = DiffusionProcess(NUM_TIMESTEPS).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(den.parameters()), lr=LR)

    dltr = DataLoader(BagDS(tr), batch_size=BATCH, shuffle=True,
                      collate_fn=BagDS.collate, num_workers=2)
    dlva = DataLoader(BagDS(va), batch_size=BATCH, shuffle=False,
                      collate_fn=BagDS.collate, num_workers=2)

    best = float("inf")
    for ep in range(EPOCHS):
        trl = run_epoch(dltr, enc, den, dp, opt, dev, True)
        with torch.no_grad():
            val = run_epoch(dlva, enc, den, dp, opt, dev, False)
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"epoch {ep} train {trl:.4f} val {val:.4f}", flush=True)
        if val < best:
            best = val
            torch.save({"encoder": enc.state_dict()}, CKPT)

    enc.load_state_dict(torch.load(CKPT)["encoder"])
    enc.eval()
    rows = []
    for sid in sample_ids:
        v = embed(enc, arrs[sid], dev)
        m = meta.loc[sid]
        rows.append([sid, m.sample_type, m.study, m.assay, m.donor, arrs[sid].shape[0]]
                    + v.tolist())
    cols = ["sample_id", "sample_type", "study", "assay", "donor", "n_cells"] + \
        [f"s{j}" for j in range(MODEL_DIM)]
    pd.DataFrame(rows, columns=cols).to_parquet(OUT)
    print(f"saved {OUT} | samples={len(rows)} best_val={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
