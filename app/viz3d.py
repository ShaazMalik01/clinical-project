"""True 3D (WebGL) charts: a 'risk city' of site x KRI towers and a 3D map of subjects."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from theme import INK, MUTED, ORANGE, RED, TEAL, style_fig

CUBE_FACES = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
              (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
RISK_SCALE = [[0, "#1B3A6B"], [0.25, "#2DD4BF"], [0.45, "#F5C451"], [0.7, "#FF944D"], [1, "#FF4D6D"]]


def _cuboid(x, y, h, w=0.9):
    v = [(x, y, 0), (x + w, y, 0), (x + w, y + w, 0), (x, y + w, 0),
         (x, y, h), (x + w, y, h), (x + w, y + w, h), (x, y + w, h)]
    return v


def risk_city(M: pd.DataFrame, labels: dict, tolerance: float = 1.0, cap: float = 5.0, height: int = 560):
    """M: sites (rows, already ordered) x KRI (columns) of posterior-rate / peer-rate. Towers that pierce the
    amber plane are above the peer rate by more than the tolerance."""
    xs, ys, zs, ii, jj, kk, inten, hover = [], [], [], [], [], [], [], []
    gap = 1.35
    for r, site in enumerate(M.index):
        for c, k in enumerate(M.columns):
            val = float(np.clip(M.loc[site, k], 0.05, cap))
            base = len(xs)
            for vx, vy, vz in _cuboid(c * gap, r * gap, val):
                xs.append(vx); ys.append(vy); zs.append(vz); inten.append(val)
                hover.append(f"{site}: {labels.get(k, k)}<br>{val:.2f}x peer rate")
            for a, b, d in CUBE_FACES:
                ii.append(base + a); jj.append(base + b); kk.append(base + d)
    fig = go.Figure(go.Mesh3d(
        x=xs, y=ys, z=zs, i=ii, j=jj, k=kk, intensity=inten, intensitymode="vertex", colorscale=RISK_SCALE,
        cmin=0, cmax=cap, flatshading=True, hovertext=hover, hoverinfo="text",
        lighting=dict(ambient=0.55, diffuse=0.9, specular=0.35, roughness=0.45, fresnel=0.3),
        lightposition=dict(x=200, y=-300, z=400),
        colorbar=dict(title=dict(text="x peers"), thickness=12, len=0.6)))
    w, d = len(M.columns) * gap, len(M.index) * gap
    fig.add_trace(go.Mesh3d(x=[-0.3, w, w, -0.3], y=[-0.3, -0.3, d, d], z=[tolerance] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
                            color=ORANGE, opacity=0.16, hoverinfo="skip", showscale=False))
    fig.update_layout(scene=dict(
        xaxis=dict(title="", tickmode="array", tickvals=[c * gap + 0.45 for c in range(len(M.columns))],
                   ticktext=[labels.get(k, k).split(" (")[0][:16] for k in M.columns], backgroundcolor="rgba(0,0,0,0)",
                   gridcolor="rgba(255,255,255,.10)", showbackground=False),
        yaxis=dict(title="", tickmode="array", tickvals=[r * gap + 0.45 for r in range(len(M.index))],
                   ticktext=list(M.index), backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,.10)",
                   showbackground=False),
        zaxis=dict(title="x peer rate", range=[0, cap], backgroundcolor="rgba(0,0,0,0)",
                   gridcolor="rgba(255,255,255,.10)", showbackground=False),
        camera=dict(eye=dict(x=1.7, y=-1.9, z=1.15)), aspectmode="manual", aspectratio=dict(x=1.5, y=1.5, z=0.8)),
        title="Risk city: each tower is one site and one indicator (amber plane = tolerance)")
    return style_fig(fig, height)


def _pca3(X: np.ndarray):
    med = np.median(X, axis=0)
    q = np.subtract(*np.percentile(X, [75, 25], axis=0))
    q = np.where(q < 1e-9, X.std(axis=0) + 1e-9, q)
    Z = np.clip((X - med) / q, -6, 6)
    Z = Z - Z.mean(axis=0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = (S ** 2) / np.sum(S ** 2)
    return Z @ Vt[:3].T, var[:3]


def anomaly_space(features: pd.DataFrame, alerts: pd.DataFrame, fz: pd.DataFrame, height: int = 600):
    """3D PCA map of subjects. Flagged subjects are larger and coloured by the worst alert severity."""
    sid = features["subject_id"] if "subject_id" in features.columns else features.index.to_series()
    num = features.select_dtypes("number").drop(columns=["miss_rate"], errors="ignore")
    P, var = _pca3(num.to_numpy(float))
    df = pd.DataFrame(P, columns=["x", "y", "z"])
    df["subject_id"] = sid.to_numpy()
    df["site_id"] = features["site_id"].to_numpy() if "site_id" in features.columns else ""
    a = alerts[alerts["subject_id"].notna()]
    sev_rank = {"critical": 3, "major": 2, "minor": 1}
    worst = a.assign(r=a["severity"].map(sev_rank)).sort_values("r", ascending=False).drop_duplicates("subject_id")
    df = df.merge(worst[["subject_id", "severity"]], on="subject_id", how="left") \
           .merge(a.groupby("subject_id").size().rename("n_alerts").reset_index(), on="subject_id", how="left")
    df["n_alerts"] = df["n_alerts"].fillna(0).astype(int)
    df = df.merge(fz[["subject_id", "tier", "suspicion"]], on="subject_id", how="left")
    base = df[df["severity"].isna()]
    hot = df[df["severity"].notna()]
    colors = {"critical": RED, "major": ORANGE, "minor": "#8B9CFF"}
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=base.x, y=base.y, z=base.z, mode="markers", name="no alerts",
        marker=dict(size=2.6, color="rgba(159,176,198,.45)"),
        hovertext=base.subject_id + " (" + base.site_id.astype(str) + ")", hoverinfo="text"))
    for sev in ("minor", "major", "critical"):
        h = hot[hot["severity"] == sev]
        if h.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=h.x, y=h.y, z=h.z, mode="markers", name=f"worst alert: {sev}",
            marker=dict(size=4 + 1.6 * h.n_alerts.clip(upper=5), color=colors[sev], opacity=0.92,
                        line=dict(color="rgba(255,255,255,.6)", width=0.5)),
            hovertext=h.subject_id + " (" + h.site_id.astype(str) + ")<br>" + h.n_alerts.astype(str) +
            " alerts, forensic tier " + h.tier.fillna("n/a"), hoverinfo="text"))
    frames = [go.Frame(layout=dict(scene_camera=dict(eye=dict(
        x=2.1 * np.cos(t), y=2.1 * np.sin(t), z=0.9)))) for t in np.linspace(0, 2 * np.pi, 72)]
    fig.frames = frames
    fig.update_layout(
        scene=dict(xaxis=dict(title=f"PC1 ({var[0]:.0%})", showbackground=False, gridcolor="rgba(255,255,255,.10)"),
                   yaxis=dict(title=f"PC2 ({var[1]:.0%})", showbackground=False, gridcolor="rgba(255,255,255,.10)"),
                   zaxis=dict(title=f"PC3 ({var[2]:.0%})", showbackground=False, gridcolor="rgba(255,255,255,.10)"),
                   camera=dict(eye=dict(x=1.8, y=1.4, z=0.9))),
        title="Subject map: each point is a subject; bigger and warmer means more severe alerts",
        updatemenus=[dict(type="buttons", direction="left", x=0.02, y=0.02, showactive=False, buttons=[
            dict(label="Rotate", method="animate", args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True,
                                                                     transition=dict(duration=0), mode="immediate")]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                      mode="immediate")])])])
    return style_fig(fig, height)
