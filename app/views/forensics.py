import numpy as np
import plotly.graph_objects as go
import streamlit as st

from theme import page_header
from common import TIER_COLOR, csv_download, plot, table

BENFORD = np.log10(1 + 1 / np.arange(1, 10))
TIER_PAL = {"HIGH": "#FF4D6D", "MEDIUM": "#FF944D", "LOW": "#8FA3BF"}


def render(ctx):
    page_header("Data-integrity forensics", "Statistical fingerprints of fabricated data: rounded last digits, flat values, copied visits, duplicate subjects and Benford's law where it applies. A high score is a reason to look, not proof of misconduct.", chips=["terminal-digit test", "variance test", "Benford with applicability gate"])
    fs = ctx.d["fz_sites"].sort_values("suspicion", ascending=False)
    fs = fs[fs.site_id.isin(ctx.sites)]
    if fs.empty:
        st.info("No sites selected.")
        return
    fig = go.Figure(go.Bar(x=fs.site_id, y=fs.suspicion, marker_color=[TIER_PAL.get(t, "#8FA3BF") for t in fs.tier],
                           text=fs.tier, textposition="outside"))
    fig.update_layout(title="Fabrication-suspicion score by site", yaxis_title="score (0 to 1)",
                      yaxis_range=[0, 1.12], height=320)
    plot(fig, key="fz_site")

    site = st.selectbox("Site detail", fs.site_id.tolist(), index=0)
    r = fs[fs.site_id == site].iloc[0]
    st.markdown(f"**{site}**: {r.tier} tier, score {r.suspicion:.2f}. "
                f"{r.reasons if isinstance(r.reasons, str) and r.reasons else 'No significant forensic signal.'}")
    if ctx.reveal and ctx.truth is not None:
        st.info(f"Synthetic ground truth: {ctx.truth.set_index('site_id')['profile'].get(site, 'unknown')}")

    c1, c2 = st.columns(2)
    with c1:
        dh = ctx.d.get("digit_hist")
        if dh is not None:
            dh = dh.set_index("site_id")
            dh.columns = [int(c) for c in dh.columns]
            others = dh.drop(index=site, errors="ignore").mean()
            fig = go.Figure()
            fig.add_bar(x=list(range(10)), y=dh.loc[site].to_numpy(), name=site, marker_color="#FF4D6D")
            fig.add_bar(x=list(range(10)), y=others.to_numpy(), name="other sites (mean)", marker_color="#8FA3BF")
            fig.add_hline(y=0.1, line_dash="dash", line_color="#C7D3E3")
            fig.update_layout(title="Last digit of BP and heart-rate readings (uniform = 0.10)",
                              barmode="group", xaxis=dict(dtick=1), height=340)
            plot(fig, key="fz_digits")
    with c2:
        bs, bm = ctx.d.get("benford_sites"), ctx.d.get("benford_meta")
        applied = sorted(bs.field.unique()) if bs is not None and len(bs) else []
        if applied:
            fld = st.selectbox("Benford field", applied)
            t = bs[bs.field == fld]
            dcols = [f"d{i}" for i in range(1, 10)]
            mine = t[t.site_id == site]
            fig = go.Figure()
            fig.add_scatter(x=list(range(1, 10)), y=BENFORD, name="Benford", mode="lines+markers",
                            line=dict(color="#C7D3E3", dash="dash"))
            fig.add_scatter(x=list(range(1, 10)), y=t[dcols].median().to_numpy(), name="median site",
                            mode="lines+markers", line=dict(color="#8FA3BF"))
            if len(mine):
                p = float(mine.p_benford.iloc[0])
                fig.add_scatter(x=list(range(1, 10)), y=mine[dcols].iloc[0].to_numpy(), mode="lines+markers",
                                name=f"{site} (p={p:.3g})", line=dict(color="#FF4D6D"))
            fig.update_layout(title=f"First digits of {fld} (one value per subject)", xaxis=dict(dtick=1),
                              height=340)
            plot(fig, key="fz_benford")
        else:
            st.info("No lab field qualified for Benford's law (see the applicability table below).")
    if ctx.d.get("benford_meta") is not None:
        with st.expander("Why Benford is only applied to some fields"):
            st.caption("Benford needs values spread over at least two orders of magnitude. Blood pressure "
                       "and most routine labs are not, so a Benford test on them would raise false alarms.")
            table(ctx.d["benford_meta"])

    st.subheader("Subjects")
    fz = ctx.d["fz_subjects"]
    fz = fz[fz.site_id.isin(ctx.sites)]
    tiers = st.multiselect("Tier", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
    s = fz[fz.tier.isin(tiers)].sort_values("suspicion", ascending=False)
    table(s[["subject_id", "site_id", "tier", "suspicion", "primary_pattern", "reasons"]], height=320)
    csv_download("Download subject forensics (CSV)", s, "forensics_subjects.csv", ctx, "dl_fz")
    d1, d2 = st.columns(2)
    with d1:
        dp = ctx.d.get("dup_pairs")
        if dp is not None:
            st.markdown("**Near-duplicate subject pairs**")
            fl = dp[dp.flagged.astype(bool)] if "flagged" in dp else dp
            table(fl if len(fl) else dp.head(0))
    with d2:
        cr = ctx.d.get("copy_records")
        if cr is not None:
            st.markdown("**Copy-forward visits**")
            table(cr[cr.site_id.isin(ctx.sites)].head(200))
