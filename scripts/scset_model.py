import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(a):
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(),
            "tanh": nn.Tanh()}[a]


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_pos=1200, temperature=10000.0):
        super().__init__()
        assert dim % 2 == 0
        pos = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                        * -(torch.log(torch.tensor(temperature)) / dim))
        pe = torch.zeros(max_pos, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, idx):
        return self.pe[idx]


class ABMIL(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers=2, activation="gelu"):
        super().__init__()
        self.V = nn.Linear(input_dim, model_dim)
        self.U = nn.Linear(input_dim, model_dim)
        self.w = nn.Linear(model_dim, 1)
        layers, last = [], input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(last, model_dim), get_activation(activation)]
            last = model_dim
        layers.append(nn.Linear(model_dim, model_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, X, X_mask=None):
        a = self.w(torch.tanh(self.V(X) * torch.sigmoid(self.U(X))))
        if X_mask is not None:
            a = a.masked_fill(X_mask.unsqueeze(-1), -1e9)
        a = F.softmax(a, dim=1)
        return self.mlp(torch.sum(a * X, dim=1))


class CellTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=256, num_blocks=2, num_heads=4,
                 feedforward_dim=512, dropout=0.0, activation="gelu"):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=num_heads,
                dim_feedforward=feedforward_dim, dropout=dropout,
                activation=activation, batch_first=True)
            for _ in range(num_blocks)])

    def forward(self, X, X_mask=None):
        X = self.embedding(X)
        cls = self.cls_token.expand(X.shape[0], 1, -1)
        X = torch.cat([cls, X], dim=1)
        if X_mask is not None:
            cm = torch.zeros(X_mask.shape[0], 1, dtype=torch.bool,
                             device=X_mask.device)
            X_mask = torch.cat([cm, X_mask], dim=1)
        for b in self.blocks:
            X = b(X, src_key_padding_mask=X_mask)
        return X[:, 0]


def build_encoder(kind, input_dim, model_dim):
    if kind == "abmil":
        return ABMIL(input_dim, model_dim)
    if kind == "cell_transformer":
        return CellTransformer(input_dim, model_dim)
    raise ValueError(kind)


class ResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim, activation="silu"):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), get_activation(activation),
                                 nn.Linear(dim, dim))
        self.mod = nn.Sequential(get_activation(activation),
                                 nn.Linear(cond_dim, 3 * dim))
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, cond):
        bias, s1, s2 = self.mod(cond).chunk(3, dim=-1)
        r = self.ln(x) * (1 + s1) + bias
        return x + self.mlp(r) * s2


class ConditionalDenoisingMLP(nn.Module):
    def __init__(self, sample_dim, cond_input_dim, time_emb_dim=128,
                 hidden_dim=256, num_res_blocks=5, activation="silu"):
        super().__init__()
        self.time_embedding = PositionalEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim), get_activation(activation),
            nn.Linear(time_emb_dim, hidden_dim))
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, hidden_dim), get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim))
        self.in_adapter = nn.Linear(sample_dim, hidden_dim)
        self.out_adapter = nn.Linear(hidden_dim, sample_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim, activation)
             for _ in range(num_res_blocks)])

    def forward(self, x, tidx, condition):
        c = self.time_mlp(self.time_embedding(tidx)) + self.cond_mlp(condition)
        x = self.in_adapter(x)
        for b in self.res_blocks:
            x = b(x, c)
        return self.out_adapter(x)


def cosine_schedule(T, s=0.008):
    x = torch.linspace(0, T, T + 1)
    ab = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2) ** 2
    ab = ab / ab[0]
    beta = 1 - ab[1:] / ab[:-1]
    return torch.clip(beta, 0.0001, 0.9999)


def _extract(sched, tidx, x):
    return sched[tidx].reshape(x.shape[0], *([1] * (x.dim() - 1)))


class DiffusionProcess(nn.Module):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        betas = cosine_schedule(num_timesteps)
        self.num_timesteps = num_timesteps
        alphas = 1 - betas
        ab = torch.cumprod(alphas, 0)
        self.register_buffer("sqrt_ab", torch.sqrt(ab))
        self.register_buffer("sqrt_ab_neg", torch.sqrt(1 - ab))

    def q_sample(self, x0, tidx, noise):
        return (_extract(self.sqrt_ab, tidx, x0) * x0
                + _extract(self.sqrt_ab_neg, tidx, x0) * noise)

    def p_loss(self, model, x0, tidx, condition):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, tidx, noise)
        return F.mse_loss(noise, model(xt, tidx, condition))
