"""Clinical trial compliance monitor: dashboard entry point.  Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent / "src"))
sys.path.insert(0, str(APP))

import streamlit as st

st.set_page_config(page_title="Trial compliance monitor", layout="wide")

import common as C                                   # noqa: E402
import theme                                         # noqa: E402
from ctmon.audit import AuditLog, ConfigWatcher      # noqa: E402
from views import anomalies, audit_page, deviations, forensics, overview, site_risk, triage   # noqa: E402

ICON = {"Overview": "◈", "Site risk": "▦", "Deviations": "☰", "Anomalies": "✦", "Forensics": "⌬",
        "Feedback and triage": "⇄", "Audit trail": "⛓"}
PAGES = {"Overview": overview, "Site risk": site_risk, "Deviations": deviations,
         "Anomalies": anomalies, "Forensics": forensics, "Feedback and triage": triage,
         "Audit trail": audit_page}


@st.cache_resource
def get_audit(path: str) -> AuditLog:
    return AuditLog(path)


theme.inject_css()
stamp = C._stamp()
d = C.load_bundle(stamp)
missing = [k for k in C.REQUIRED if k not in d]
if missing:
    st.title("Trial compliance monitor")
    st.error("Results are not built yet. Run `python build_dashboard_data.py` from the project folder, "
             "then reload this page.")
    st.caption(f"Missing: {', '.join(C.FILES[k] for k in missing)} in {C.PROC}")
    st.stop()

audit = get_audit(str(C.AUDIT_DIR / "audit_log.jsonl"))
cfg = C.load_settings()
if "cfg_checked" not in st.session_state:             # log any config edit made since the last run
    cfg_files = {n: C.ROOT / "config" / f"{n}.yaml" for n in ("settings", "protocol", "rules")}
    ConfigWatcher(audit, cfg_files, C.AUDIT_DIR / "config_snapshots").check("dashboard")
    st.session_state["cfg_checked"] = True

all_sites = sorted(d["site_risk"].site_id.unique())
with st.sidebar:
    st.markdown("<div style='font-size:1.3rem;font-weight:700'>Trial monitor</div>"
                "<div class='small'>CTX-301 central monitoring</div>", unsafe_allow_html=True)
    page = st.radio("Page", list(PAGES), key="nav", label_visibility="collapsed",
                    format_func=lambda p: f"{ICON[p]}  {p}")
    st.divider()
    reviewer = st.text_input("Your name (recorded in the audit log)", "monitor1")
    sites = st.multiselect("Sites", all_sites, default=all_sites)
    sevs = st.multiselect("Severity", C.SEVERITIES, default=C.SEVERITIES)
    truth = C.load_truth()
    reveal = st.toggle("Show synthetic ground truth", value=False) if truth is not None else False
    st.divider()
    report = audit.verify()
    st.markdown(C.chain_badge(report), unsafe_allow_html=True)

ctx = C.Ctx(d=d, raw=C.load_raw(stamp), cfg=cfg, manifest=C.load_manifest(), audit=audit, report=report,
            reviewer=reviewer.strip() or "monitor1", sites=sites or all_sites, sevs=sevs or C.SEVERITIES,
            truth=truth, reveal=reveal)
PAGES[page].render(ctx)
