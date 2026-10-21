import plotly.express as px
import streamlit as st

from theme import page_header
from common import SEV_COLOR, SEVERITIES, by_filters, csv_download, plot, table


def render(ctx):
    page_header("Protocol deviations", "Violations found by the rule engine. Every row cites the rule and its guideline reference.", chips=["knowledge-based rules", "ICH E6 references"])
    dev = by_filters(ctx.d["deviations"], ctx)
    c1, c2, c3 = st.columns([2, 2, 2])
    rules = sorted(dev.rule_id.unique())
    pick = c1.multiselect("Rule", rules, default=rules)
    cats = sorted(dev.category.unique())
    cat = c2.multiselect("Category", cats, default=cats,
                         help="PD protocol deviation, OUT data plausibility, SAE safety reporting")
    q = c3.text_input("Search subject or text", "")
    dev = dev[dev.rule_id.isin(pick) & dev.category.isin(cat)]
    if q.strip():
        s = q.strip().lower()
        dev = dev[dev.subject_id.astype(str).str.lower().str.contains(s)
                  | dev.evidence.astype(str).str.lower().str.contains(s)]
    st.write(f"{len(dev):,} deviations")
    if dev.empty:
        st.info("Nothing matches the current filters.")
        return
    g = dev.groupby(["rule_id", "severity"]).size().reset_index(name="deviations")
    fig = px.bar(g, x="deviations", y="rule_id", color="severity", orientation="h",
                 color_discrete_map=SEV_COLOR, category_orders={"severity": SEVERITIES},
                 title="Deviations by rule")
    fig.update_yaxes(categoryorder="total ascending")
    plot(fig, key="dev_rule")
    cols = ["deviation_id", "rule_id", "severity", "site_id", "subject_id", "visit_num", "observed",
            "expected", "evidence"]
    table(dev[cols], height=360)
    csv_download("Download filtered deviations (CSV)", dev, "deviations_filtered.csv", ctx, "dl_dev_page")

    st.subheader("Inspect one deviation")
    ids = dev.deviation_id.tolist()[:500]
    sel = st.selectbox("Deviation", ids)
    r = dev[dev.deviation_id == sel].iloc[0]
    st.markdown(f"**{r.rule_name}** ({r.rule_id}, {r.severity}) for subject {r.subject_id} at site {r.site_id}")
    st.write(r.evidence)
    st.caption(f"Observed {r.observed}; expected {r.expected}. Reference: {r.reference}")
    esc = ctx.d.get("escalations")
    if esc is not None and len(esc):
        with st.expander("Escalations raised by meta-rules"):
            table(esc[esc.site_id.isin(ctx.sites)])
