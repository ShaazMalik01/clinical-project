import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import page_header
from viz3d import anomaly_space
from common import SOURCES, by_filters, csv_download, render_card, table


def render(ctx):
    page_header("Anomaly explorer", "Every alert comes with a plain-English card: what is unusual, how far from normal, and what to check next.", chips=["why-flagged cards", "subject drill-down", "3D subject map"])
    tab_q, tab_s, tab_3 = st.tabs(["Alert queue with explanations", "Subject drill-down", "3D subject map"])
    alerts = by_filters(ctx.d["alerts"], ctx)

    with tab_q:
        c1, c2 = st.columns([2, 3])
        src = c1.multiselect("Detector", SOURCES, default=SOURCES)
        q = c2.text_input("Search subject, site or headline", "", key="an_search")
        a = alerts[alerts.source.isin(src)]
        if q.strip():
            s = q.strip().lower()
            a = a[a.subject_id.astype(str).str.lower().str.contains(s)
                  | a.headline.astype(str).str.lower().str.contains(s)
                  | a.site_id.astype(str).str.lower().str.contains(s)]
        st.write(f"{len(a):,} alerts, most severe first")
        if a.empty:
            st.info("Nothing matches the current filters.")
        else:
            table(a[["alert_id", "source", "severity", "site_id", "subject_id", "visit_num", "priority",
                     "headline"]], height=320)
            csv_download("Download filtered alerts (CSV)", a.drop(columns=["card"], errors="ignore"),
                         "alerts_filtered.csv", ctx, "dl_alert_page")
            ids = a.alert_id.tolist()[:600]
            labels = dict(zip(a.alert_id, a.headline))
            sel = st.selectbox("Open the explanation card", ids, format_func=lambda i: f"{i}: {labels[i]}"[:110])
            render_card(a[a.alert_id == sel].iloc[0])

    with tab_s:
        vit = ctx.raw.get("vitals")
        if vit is None:
            st.info("Raw vitals not found in data/raw, so the drill-down chart is unavailable.")
            return
        flagged = alerts.subject_id.dropna().astype(str).value_counts()
        subs = list(flagged.index) + [s for s in sorted(vit.subject_id.unique()) if s not in flagged.index]
        subs = [s for s in subs if s in set(vit[vit.site_id.isin(ctx.sites)].subject_id)]
        if not subs:
            st.info("No subjects for the selected sites.")
            return
        sid = st.selectbox("Subject (most-flagged first)", subs[:800])
        v = vit[vit.subject_id == sid].sort_values("visit_num")
        marks = alerts[(alerts.subject_id == sid) & alerts.visit_num.notna()]
        fig = go.Figure()
        for col, color in (("sbp", "#2DD4BF"), ("dbp", "#8B9CFF"), ("hr", "#FF944D")):
            fig.add_trace(go.Scatter(x=v.visit_num, y=v[col], mode="lines+markers", name=col.upper(),
                                     line=dict(color=color)))
        for vn in sorted(set(marks.visit_num.astype(int))):
            fig.add_vrect(x0=vn - 0.4, x1=vn + 0.4, fillcolor="#FF4D6D", opacity=0.12, line_width=0)
        fig.update_layout(title=f"Vital signs for {sid} (shaded visits have alerts)",
                          xaxis_title="visit", yaxis_title="value", height=380)
        from common import plot
        plot(fig, key="an_subject")
        sa = alerts[alerts.subject_id == sid]
        if sa.empty:
            st.success("No alerts for this subject.")
        for _, r in sa.head(6).iterrows():
            render_card(r)

    with tab_3:
        feats = ctx.d.get("subject_features")
        if feats is None:
            st.info("Subject features not found. Run `python build_dashboard_data.py` first.")
            return
        feats = feats[feats.site_id.isin(ctx.sites)] if "site_id" in feats.columns else feats
        from common import plot
        plot(anomaly_space(feats, alerts, ctx.d["fz_subjects"]), key="an_space")
        st.caption("A 3D projection (PCA) of 15 subject features. Drag to rotate, or press Rotate. Subjects far "
                   "from the dense core, or with large warm markers, deserve a closer look.")
