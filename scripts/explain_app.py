"""Interactive explorer (FAIR cows-on-beach): tight single-study seed.
Seed = tumors from ONE study (Borras_2023) + normals from ONE study (Scheid_2023) =
two COMPACT blobs, a study-level confound = a realistic single-lab cohort. We ADD
diverse same-label samples from OTHER studies. Two panels contrast retrieval quantiles
(dropdowns): NEAR (low q) pulls studies close to the seed (redundant batch, little
de-confounding); FAR (high q) reaches distant studies. Canvas = orthonormal
(d_batch, d_bio) plane tailored to THIS confound (batch = Borras↔Scheid direction with
biology removed). Black arrow = LR weight vector w. Sliders: seed batch purity rho
(fraction of seed from the home study; 1.0 = pure/tight/most confounded), #added k.

Run:  conda activate bioml && python scripts/explain_app.py   -> http://127.0.0.1:8050
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from selection import greedy_quantile
import crc_enrichment as ce

TUMOR_STUDY = "Borras_2023_Cell_Discov"     # tight tumor cluster
NORMAL_STUDY = "Scheid_2023_J_EXP_Med"      # tight normal cluster
N, K = 12, 12
QGRID = [0.1, 0.25, 0.5, 0.75, 0.9]
RHOS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]       # seed batch purity (home-study fraction)
FRAME_SEED = 0
AGG_SEEDS = 12
ARROW_LEN = 11.0
TECH_COLORS = {}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def qtag(q):
    return "near" if q <= 0.25 else ("far" if q >= 0.75 else "mid")


# ---------------- precompute ----------------
print("precomputing ...")
s = ce.load(); C = ce.SCOLS
held, pool = ce.split(s)
poolr = pool.reset_index(drop=True)
scaler = StandardScaler().fit(poolr[C].to_numpy())
Zp = pd.DataFrame(scaler.transform(poolr[C].to_numpy()), index=poolr.index)
Zall = scaler.transform(s[C].to_numpy())

# biology axis: within-study tumor-normal, averaged
bio = []
for st, g in poolr.groupby("study"):
    if g.sample_type.nunique() < 2:
        continue
    bio.append(Zp.loc[g[g.sample_type == "tumor"].index].mean(0).to_numpy()
               - Zp.loc[g[g.sample_type == "normal"].index].mean(0).to_numpy())
d_bio = unit(np.mean(bio, 0))

# batch axis tailored to THIS confound: (home tumor mean - home normal mean), biology removed
home_t = poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index
home_n = poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index
raw = Zp.loc[home_t].mean(0).to_numpy() - Zp.loc[home_n].mean(0).to_numpy()
d_batch = unit(raw - (raw @ d_bio) * d_bio)
e_batch, e_bio = d_batch, d_bio           # already orthogonal by construction


def proj(M):
    return np.c_[M @ e_batch, M @ e_bio]


BG = proj(Zall)
BG_TYPE = s.sample_type.to_numpy()
XR = [BG[:, 0].min() - 2, BG[:, 0].max() + 2]
YR = [BG[:, 1].min() - 2, BG[:, 1].max() + 2]

Zheld = {st: scaler.transform(g[C].to_numpy()) for st, g in held.groupby("study")
         if g.y.nunique() == 2}
yheld = {st: g.y.to_numpy() for st, g in held.groupby("study") if g.y.nunique() == 2}

HT = poolr[(poolr.study == TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()     # home tumor
HN = poolr[(poolr.study == NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()    # home normal
OT = poolr[(poolr.study != TUMOR_STUDY) & (poolr.y == 1)].index.to_numpy()     # other tumor
ON = poolr[(poolr.study != NORMAL_STUDY) & (poolr.y == 0)].index.to_numpy()    # other normal

PZ = proj(Zp.to_numpy())
PALETTE = ["#e377c2", "#17becf", "#bcbd22", "#8c564b", "#9467bd", "#ff7f0e",
           "#2ca02c", "#1f77b4", "#d62728", "#7f7f7f"]
for i, a in enumerate(sorted(poolr.study.unique())):
    TECH_COLORS[a] = PALETTE[i % len(PALETTE)]

D = ce.l2_distance_matrix(Zp.to_numpy())


def seed_for(rho, seed):
    rng = np.random.default_rng(seed)
    nH = int(round(rho * N)); nF = N - nH
    sp = np.concatenate([rng.choice(HT, nH, replace=False),
                         rng.choice(OT, nF, replace=False)]) if nF else rng.choice(HT, nH, replace=False)
    sn = np.concatenate([rng.choice(HN, nH, replace=False),
                         rng.choice(ON, nF, replace=False)]) if nF else rng.choice(HN, nH, replace=False)
    return sp, sn


def added_for(sp, sn, q):
    apool = np.setdiff1d(OT, sp); anpool = np.setdiff1d(ON, sn)
    return (greedy_quantile(D, list(sp), list(apool), K, q),
            greedy_quantile(D, list(sn), list(anpool), K, q))


def fit(coh):
    clf = LogisticRegression(max_iter=2000).fit(
        Zp.loc[coh].to_numpy(), poolr.loc[coh, "y"].to_numpy())
    w = unit(clf.coef_[0])
    auc = np.mean([roc_auc_score(yheld[st], clf.predict_proba(Zheld[st])[:, 1]) for st in Zheld])
    return w, float(auc)


def pick_dist(seed_idx, picks):
    return float(np.mean([D[np.ix_([p], list(seed_idx))].min() for p in picks]))


STATE, AGG = {}, {}
for rho in RHOS:
    for q in QGRID:
        sp, sn = seed_for(rho, FRAME_SEED)
        ap, an = added_for(sp, sn, q)
        rec = {"sp": PZ[sp], "sn": PZ[sn],
               "ap_xy": PZ[np.array(ap, int)], "an_xy": PZ[np.array(an, int)],
               "ap_st": poolr.loc[ap, "study"].to_numpy(),
               "an_st": poolr.loc[an, "study"].to_numpy(),
               "pdist": (pick_dist(sp, ap) + pick_dist(sn, an)) / 2,
               "w2": [], "cosb": [], "cosbio": [], "auc": []}
        for k in range(K + 1):
            coh = np.concatenate([sp, np.array(ap[:k], int), sn, np.array(an[:k], int)])
            w, auc = fit(coh)
            rec["w2"].append([w @ e_batch, w @ e_bio])
            rec["cosb"].append(w @ d_batch); rec["cosbio"].append(w @ d_bio)
            rec["auc"].append(auc)
        STATE[(rho, q)] = rec
        curK = np.zeros(K + 1)
        for sd in range(AGG_SEEDS):
            sp2, sn2 = seed_for(rho, sd)
            ap2, an2 = added_for(sp2, sn2, q)
            for k in range(K + 1):
                coh = np.concatenate([sp2, np.array(ap2[:k], int), sn2, np.array(an2[:k], int)])
                curK[k] += fit(coh)[1]
        AGG[(rho, q)] = curK / AGG_SEEDS
print("precompute done.")


# ---------------- figure ----------------
def panel(rho, k, q):
    r = STATE[(rho, q)]
    fig = go.Figure()
    for tp, col in [("tumor", "#d62728"), ("normal", "#1f77b4")]:
        m = BG_TYPE == tp
        fig.add_scatter(x=BG[m, 0], y=BG[m, 1], mode="markers",
                        marker=dict(size=4, color=col, opacity=0.10),
                        hoverinfo="skip", showlegend=False)
    for xy, col in [(r["sp"], "#d62728"), (r["sn"], "#1f77b4")]:
        fig.add_scatter(x=xy[:, 0], y=xy[:, 1], mode="markers",
                        marker=dict(size=9, color=col, symbol="circle-open",
                                    line=dict(width=2)),
                        hoverinfo="skip", showlegend=False)
    shown = set()
    for xy, stds in [(r["ap_xy"][:k], r["ap_st"][:k]), (r["an_xy"][:k], r["an_st"][:k])]:
        for i in range(len(xy)):
            t = stds[i]
            fig.add_scatter(x=[xy[i, 0]], y=[xy[i, 1]], mode="markers",
                            marker=dict(size=13, color=TECH_COLORS.get(t, "#333"),
                                        symbol="star", line=dict(width=0.8, color="black")),
                            name=t, legendgroup=t, showlegend=t not in shown,
                            hovertext=t, hoverinfo="text")
            shown.add(t)
    w2 = np.array(r["w2"][k]); w2 = w2 / np.linalg.norm(w2) * ARROW_LEN
    fig.add_annotation(x=w2[0], y=w2[1], ax=0, ay=0, xref="x", yref="y",
                       axref="x", ayref="y", showarrow=True, arrowhead=2,
                       arrowsize=1.6, arrowwidth=3, arrowcolor="black")
    fig.add_hline(y=0, line=dict(color="gray", width=0.6, dash="dot"))
    fig.add_vline(x=0, line=dict(color="gray", width=0.6, dash="dot"))
    agg = AGG[(rho, q)][k]
    fig.update_layout(
        title=(f"<b>q={q:.2f} ({qtag(q)})</b>  k={k}<br>"
               f"<sub>pick-dist={r['pdist']:.0f} | cos(w,batch)={r['cosb'][k]:+.2f} "
               f"cos(w,bio)={r['cosbio'][k]:+.2f} | AUC seed {r['auc'][k]:.2f} / mean {agg:.2f}</sub>"),
        xaxis=dict(title="batch axis  →", range=XR, zeroline=False),
        yaxis=dict(title="biology axis  ↑", range=YR, zeroline=False),
        legend=dict(font=dict(size=8), title="added study"),
        margin=dict(l=40, r=10, t=60, b=40), height=560)
    return fig


qopts = [{"label": f"q={q:.2f} ({qtag(q)})", "value": q} for q in QGRID]
app = Dash(__name__)
app.layout = html.Div([
    html.H3("Fair cows-on-beach: tight single-study seed. Does retrieval distance matter?"),
    html.Div(f"seed = tumors from {TUMOR_STUDY} + normals from {NORMAL_STUDY} "
             f"(two tight blobs); ADD diverse samples from OTHER studies.",
             style={"width": "80%", "margin": "auto", "color": "#333", "fontSize": 13}),
    html.Div([
        html.Div([html.Label("LEFT panel quantile"),
                  dcc.Dropdown(qopts, 0.9, id="ql", clearable=False)],
                 style={"width": "24%", "display": "inline-block"}),
        html.Div([html.Label("RIGHT panel quantile"),
                  dcc.Dropdown(qopts, 0.1, id="qr", clearable=False)],
                 style={"width": "24%", "display": "inline-block", "marginLeft": "2%"}),
    ], style={"width": "70%", "margin": "auto", "marginTop": "8px"}),
    html.Div([
        html.Label("seed batch purity  ρ  (home-study fraction; 1.0 = pure tight blob)"),
        dcc.Slider(0.5, 1.0, 0.1, value=1.0, id="rho",
                   marks={round(x, 1): f"{x:.1f}" for x in RHOS}),
        html.Label("diverse samples added per class  k"),
        dcc.Slider(0, K, 1, value=6, id="k", marks={i: str(i) for i in range(0, K + 1, 2)}),
    ], style={"width": "70%", "margin": "auto", "marginTop": "10px"}),
    html.Div([
        dcc.Graph(id="left", style={"display": "inline-block", "width": "49%"}),
        dcc.Graph(id="right", style={"display": "inline-block", "width": "49%"}),
    ]),
    html.Div("Open circles = seed cohort (tight, fixed). Stars = added samples, colored "
             "by source study. Faint cloud = all patients (red tumor / blue normal). "
             "pick-dist = mean distance of added samples to the seed. "
             "Single illustrative seed; 'mean' AUC averaged over seeds.",
             style={"width": "80%", "margin": "auto", "color": "#555", "fontSize": 13}),
])


@app.callback(Output("left", "figure"), Output("right", "figure"),
              Input("rho", "value"), Input("k", "value"),
              Input("ql", "value"), Input("qr", "value"))
def update(rho, k, ql, qr):
    return panel(round(rho, 1), k, ql), panel(round(rho, 1), k, qr)


if __name__ == "__main__":
    app.run(debug=False, port=8050)
