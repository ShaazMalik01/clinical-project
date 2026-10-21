import tempfile
from pathlib import Path

import streamlit as st

from common import chain_badge, plot, table
from theme import hero
from ctmon.audit import AuditLog, run_tamper_suite


def render(ctx):
    log, rep = ctx.audit, ctx.report
    colors = ["#2DD4BF"] * 6 if rep.ok else ["#2DD4BF", "#2DD4BF", "#FF4D6D", "#5B6B80", "#5B6B80", "#5B6B80"]
    hero("Tamper-evident log", "Audit trail", "Every flag, decision, export and configuration change is chained to the entry before it with a SHA-256 hash. Changing, deleting or reordering any entry breaks the chain.", chips=[f"{rep.n_entries:,} entries", "verified" if rep.ok else "BROKEN", "append-only"], cubes=colors)
    c = st.columns(4)
    c[0].metric("Entries", f"{rep.n_entries:,}")
    c[1].metric("Chain status", "Verified" if rep.ok else "BROKEN")
    c[2].metric("Head hash", rep.head_hash[:12] if rep.head_hash else "-")
    c[3].metric("Protection", ("HMAC + " if rep.hmac_used else "") + ("anchor" if rep.anchor_checked else "chain only"))
    st.markdown(chain_badge(rep), unsafe_allow_html=True)
    if rep.ok:
        st.success(rep.summary())
    else:
        st.error(rep.summary())
        table(rep.to_frame())
    if st.button("Verify now"):
        r = log.verify()
        log.log_verification(r, ctx.reviewer)
        st.write(r.summary())

    t_log, t_file, t_tamper = st.tabs(["Log", "Verify a file", "Tamper test"])
    with t_log:
        df = log.to_frame()
        events = sorted(df.event.dropna().unique())
        pick = st.multiselect("Event type", events, default=events)
        actors = sorted(df.actor.dropna().unique())
        who = st.multiselect("Actor", actors, default=actors)
        view = df[df.event.isin(pick) & df.actor.isin(who)].sort_values("seq", ascending=False)
        st.write(f"{len(view):,} of {len(df):,} entries")
        table(view.head(500), height=380)
        if log.path.exists():
            st.download_button("Download audit log (JSON lines)", log.path.read_bytes(),
                               file_name="audit_log.jsonl", mime="application/json", key="dl_audit")
    with t_file:
        st.write("Download the log, change one character in any text editor, upload it here, and see it "
                 "get caught.")
        up = st.file_uploader("Audit log file (.jsonl)", type=["jsonl", "json", "txt"])
        if up is not None:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "uploaded.jsonl"
                p.write_bytes(up.getvalue())
                chk = AuditLog(p, key=log.key if log.key else False, auto_anchor=False)
                r = chk.verify(check_anchor=False, expected_head=log.read_anchor())
            (st.success if r.ok else st.error)(r.summary())
            if not r.ok:
                table(r.to_frame())
    with t_tamper:
        st.write("Attack copies of the log in eleven ways and check whether each is detected. "
                 "Your real log is never touched.")
        if st.button("Run tamper simulation"):
            try:
                with tempfile.TemporaryDirectory() as td:
                    st.session_state["tamper"] = run_tamper_suite(log, Path(td) / "t")
            except ValueError as e:
                st.warning(str(e))
        res = st.session_state.get("tamper")
        if res is not None:
            show = res.copy()
            show["detected"] = show.detected.map({True: "detected", False: "NOT detected"})
            show["as designed"] = show.as_designed.map({True: "yes", False: "NO"})
            table(show[["scenario", "attack", "protection", "detected", "as designed", "problems"]])
            if res.as_designed.all():
                st.success("Every attack was detected where a protection layer applies. The two rows marked "
                           "NOT detected show what a bare hash chain cannot catch, which is why the anchor "
                           "and the optional HMAC key exist.")
            else:
                st.error("Some scenarios did not behave as designed. Check the audit module.")
