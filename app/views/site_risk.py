import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import KRI_LABEL, KRI_ORDER, TIER_COLOR, TIER_LABEL, TIERS, plot, table, tier_tag
from theme import page_header
from viz3d import risk_city


def render(ctx):
    page_header("Site risk", "Bayesian posterior risk per site. Small sites are pulled towards their peers, so a "
                "few events at a small site do not trigger an unfair alarm.",
                chips=["7 key risk indicators", "95% credible intervals", "4 recommended actions"])
    sr = ctx.d["site_risk"].sort_values("rank")
    sr = sr[sr.site_id.isin(ctx.sites)]
    if sr.empty:
        st.info("No sites selected.")
        return
    kri = ctx.d["kri"]
    order = sr.site_id.tolist()
    cols = [k for k in KRI_ORDER if k in set(kri.kri)]
    M = kri.pivot(index="site_id", columns="kri", values="risk_ratio").loc[order, cols]

    view = st.radio("View", ["3D risk city", "2D heatmap"], horizontal=True)
    if view == "3D risk city":
        plot(risk_city(M, KRI_LABEL, tolerance=ctx.cfg.get("risk", {}).get("posterior", {}).get("tolerance", 1.5)),
             key="sr_city")
        st.caption("Drag to rotate, scroll to zoom, hover a tower for the exact value.")
    else:
        fig = go.Figure(go.Heatmap(
            z=M.clip(upper=5).to_numpy(), x=[KRI_LABEL.get(k, k) for k in cols], y=order, zmin=0, zmax=5,
            text=np.round(M.to_numpy(), 1), texttemplate="%{text}",
            colorscale=[[0, "#16213A"], [0.3, "#1F7A8C"], [0.5, "#F5C451"], [0.75, "#FF944D"], [1, "#FF4D6D"]],
            colorbar=dict(title="x peers")))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Posterior rate relative to peer rate (1.0 = same as peers)",
                          height=130 + 34 * len(order))
        plot(fig, key="sr_heat")

    fig = go.Figure()
    for t in TIERS:
        d = sr[sr.tier == t]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d.risk_index, y=d.site_id, mode="markers", name=TIER_LABEL[t],
            marker=dict(color=TIER_COLOR[t], size=12, line=dict(color="white", width=1)),
            error_x=dict(type="data", symmetric=False, array=(d.ci_hi - d.risk_index).clip(lower=0),
                         arrayminus=(d.risk_index - d.ci_lo).clip(lower=0), color=TIER_COLOR[t])))
    tiers_cfg = ctx.cfg.get("risk", {}).get("tiers", {})
    for key, dash in (("remote_review", "dot"), ("targeted_onsite", "dash"), ("full_audit", "solid")):
        v = tiers_cfg.get(key, {}).get("index")
        if v is not None:
            fig.add_vline(x=v, line_dash=dash, line_color="#9FB0C6", line_width=1)
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=order)
    fig.update_layout(title="Risk index with 95% credible interval", xaxis_title="weighted excess risk index",
                      height=130 + 34 * len(order))
    plot(fig, key="sr_index")

    st.subheader("Site detail")
    focus = st.session_state.get("focus_site")
    site = st.selectbox("Site", order, index=order.index(focus) if focus in order else 0)
    r = sr[sr.site_id == site].iloc[0]
    st.markdown(f"{tier_tag(r.tier)} rank {int(r['rank'])} of {len(ctx.d['site_risk'])}, "
                f"{int(r.n_subjects)} subjects", unsafe_allow_html=True)
    st.markdown(f"**Recommended action:** {r.recommendation}")
    st.markdown(f"**Why:** {r.reason}")
    st.markdown(f"**Main drivers:** {r.drivers}")
    k = kri[kri.site_id == site].copy()
    k["indicator"] = k.kri.map(KRI_LABEL)
    k["events / opportunities"] = k.events.astype(str) + " / " + k.opportunities.astype(str)
    k["posterior 95% interval"] = k.ci_lo.map("{:.1%}".format) + " to " + k.ci_hi.map("{:.1%}".format)
    show = k[["indicator", "events / opportunities", "raw_rate", "post_mean", "posterior 95% interval",
              "ref_rate", "p_exceed"]].rename(columns={"raw_rate": "raw rate", "post_mean": "posterior mean",
                                                       "ref_rate": "peer rate", "p_exceed": "P(above tolerance)"})
    table(show.style.format({"raw rate": "{:.1%}", "posterior mean": "{:.1%}", "peer rate": "{:.1%}",
                             "P(above tolerance)": "{:.2f}"}))
    if ctx.reveal and ctx.truth is not None:
        t = ctx.truth.set_index("site_id")["profile"].get(site, "unknown")
        st.info(f"Synthetic ground truth for {site}: {t}")
