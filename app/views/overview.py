import plotly.express as px
import streamlit as st

from common import (KRI_LABEL, KRI_ORDER, SEV_COLOR, SEVERITIES, SOURCES, TIER_COLOR, TIER_LABEL, by_filters,
                    csv_download, goto, plot, site_summary, table)
from theme import hero, tile_html
from viz3d import risk_city


def render(ctx):
    m = ctx.manifest
    alerts = by_filters(ctx.d["alerts"], ctx)
    sr = ctx.d["site_risk"].sort_values("rank")
    sr_f = sr[sr.site_id.isin(ctx.sites)]
    fz = ctx.d["fz_subjects"]
    fz = fz[fz.site_id.isin(ctx.sites)]

    hero("Central monitoring", "Trial compliance overview",
         "Rules, machine learning and data-integrity forensics, combined into one risk picture per site. "
         "Start with the sites that need action, then drill into the evidence.",
         chips=[f"{m.get('n_subjects', '?')} subjects", f"{m.get('n_sites', '?')} sites",
                f"{len(alerts):,} alerts in view", f"run {str(m.get('run_id', '-'))[-16:]}"],
         cubes=[TIER_COLOR[t] for t in sr.head(5).tier])

    c = st.columns(5)
    c[0].metric("Alerts in queue", f"{len(alerts):,}")
    c[1].metric("Critical alerts", int((alerts.severity == "critical").sum()))
    c[2].metric("Sites needing action", int((sr_f.tier != "ROUTINE").sum()),
                help="Sites whose recommended action is above routine monitoring.")
    c[3].metric("High-suspicion subjects", int((fz.tier == "HIGH").sum()),
                help="Subjects in the HIGH tier of the data-integrity screen.")
    c[4].metric("Audit chain", "Verified" if ctx.report.ok else "Broken",
                delta=f"{ctx.report.n_entries:,} entries", delta_color="off")

    st.subheader("Where to act first")
    act = sr_f[sr_f.tier != "ROUTINE"].head(4)
    if act.empty:
        st.success("No site is above routine monitoring for the current selection.")
    else:
        cols = st.columns(len(act))
        for col, r in zip(cols, act.itertuples()):
            with col:
                st.markdown(tile_html(r.site_id, TIER_LABEL[r.tier], TIER_COLOR[r.tier], r.risk_index, r.ci_lo,
                                      r.ci_hi, r.recommendation), unsafe_allow_html=True)
                st.button("Open site detail", key=f"open_{r.site_id}", on_click=goto, args=("Site risk", r.site_id))

    t1, t2, t3 = st.tabs(["Alerts by detector", "Alerts by site", "3D risk city"])
    with t1:
        g = alerts.groupby(["source", "severity"]).size().reset_index(name="alerts")
        plot(px.bar(g, x="source", y="alerts", color="severity", color_discrete_map=SEV_COLOR,
                    category_orders={"source": SOURCES, "severity": SEVERITIES},
                    title="Alerts by detector and severity"), key="ov_source")
    with t2:
        g = alerts.groupby(["site_id", "severity"]).size().reset_index(name="alerts")
        plot(px.bar(g, x="site_id", y="alerts", color="severity", color_discrete_map=SEV_COLOR,
                    category_orders={"severity": SEVERITIES}, title="Alerts by site"), key="ov_site")
    with t3:
        kri = ctx.d["kri"]
        order = [s for s in sr.site_id if s in ctx.sites]
        cols_k = [k for k in KRI_ORDER if k in set(kri.kri)]
        M = kri.pivot(index="site_id", columns="kri", values="risk_ratio").loc[order, cols_k]
        plot(risk_city(M, KRI_LABEL, tolerance=ctx.cfg.get("risk", {}).get("posterior", {}).get("tolerance", 1.5)),
             key="ov_city")
        st.caption("Drag to rotate, scroll to zoom. Towers that rise through the amber plane are above the "
                   "tolerated multiple of the peer rate.")

    st.subheader("Reports")
    st.caption("Each download is recorded in the audit log.")
    top = site_summary(ctx)
    top = top[top.site_id.isin(ctx.sites)]
    a, b, d = st.columns(3)
    with a:
        csv_download("Site summary (CSV)", top, "site_summary.csv", ctx, "dl_site_summary")
    with b:
        csv_download("Alert queue (CSV)", alerts.drop(columns=["card"], errors="ignore"),
                     "alert_queue.csv", ctx, "dl_alerts")
    with d:
        csv_download("Deviation log (CSV)", by_filters(ctx.d["deviations"], ctx),
                     "deviation_log.csv", ctx, "dl_devs")
    with st.expander("Preview: site summary"):
        table(top.drop(columns=["recommendation"]))
