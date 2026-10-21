import numpy as np
import plotly.graph_objects as go
import streamlit as st

from theme import page_header
from common import AUDIT_DIR, plot, render_card, table
from ctmon.feedback import FeedbackLearner, FeedbackLog, LearnerConfig, run_feedback_experiment

FB_PATH = AUDIT_DIR / "feedback_log.csv"
STREAMS = {"Subject profiles": "stream_subject", "Visit values": "stream_visit"}


def _log() -> FeedbackLog:
    if "fb_log" not in st.session_state:
        st.session_state["fb_log"] = FeedbackLog.load(FB_PATH) if FB_PATH.exists() else FeedbackLog()
    return st.session_state["fb_log"]


def _decide(ctx, alert_id: str, is_true: bool, source: str, comment_key: str) -> None:
    comment = st.session_state.get(comment_key, "")
    log = _log()
    log.add(alert_id, is_true, ctx.reviewer, comment)
    log.save(FB_PATH)
    ctx.audit.log_decision(alert_id, is_true, ctx.reviewer, comment, source)
    st.toast(f"Decision on {alert_id} recorded in the audit chain")


def render(ctx):
    page_header("Feedback and triage", "Mark alerts as a true issue or a false alarm. The system learns which signals matter and re-ranks what is left. Every decision is written to the audit log.", chips=["human in the loop", "adaptive threshold", "audited decisions"])
    name = st.radio("Alert stream", list(STREAMS), horizontal=True)
    S = ctx.d.get(STREAMS[name])
    if S is None:
        st.info("This stream is not available. Run `python build_dashboard_data.py` first.")
        return
    S = S.set_index("alert_id")
    X = S.drop(columns=["base", "y_true"], errors="ignore")
    base = S["base"]
    fb = ctx.cfg.get("feedback", {})
    cfg = LearnerConfig(**fb.get("learner", {}))
    log = _log()
    learner = FeedbackLearner(X, base.to_numpy(), cfg)
    fitted = learner.fit_from_log(log)
    labels = {k: v for k, v in log.labels().items() if k in set(X.index)}

    if "last_logged_labels" not in st.session_state:
        st.session_state["last_logged_labels"] = len(labels)
    if fitted and len(labels) != st.session_state["last_logged_labels"]:
        co = learner.coefficients().sort_values(key=np.abs, ascending=False)
        ctx.audit.log_model_update({"stream": name, "n_labels": len(labels), "learner": "logistic regression",
                                    "C": cfg.C, "p_star": cfg.p_star,
                                    "top_coefficients": {k: round(float(v), 3) for k, v in co.head(5).items()}})
    st.session_state["last_logged_labels"] = len(labels)

    c = st.columns(4)
    c[0].metric("Decisions recorded", len(labels))
    c[1].metric("Confirmed as true", f"{sum(labels.values())}")
    c[2].metric("Marked false alarm", f"{len(labels) - sum(labels.values())}")
    c[3].metric("Model", "Learning" if fitted else "Warming up",
                help=f"Needs at least {cfg.min_labels} decisions with both outcomes.")

    proba = learner.proba()
    score = learner.score()
    unrev = [i for i in range(len(X)) if X.index[i] not in labels]
    order = sorted(unrev, key=lambda i: -score[i])
    base_rank = {X.index[i]: r + 1 for r, i in enumerate(np.argsort(-base.to_numpy()))}
    if fitted:
        n_flag = int(sum(proba[i] >= cfg.p_star for i in unrev))
        st.markdown(f"**Adapted threshold:** review an alert when P(true issue) is at least "
                    f"{cfg.p_star:.2f}. {n_flag} of {len(unrev)} unreviewed alerts qualify.")

    st.subheader("Next alerts to review")
    n_show = st.slider("Alerts to show", 3, 20, 6)
    alerts = ctx.d["alerts"].set_index("alert_id", drop=False)
    source = "ML-subject" if name == "Subject profiles" else "ML-visit"
    for k, i in enumerate(order[:n_show]):
        aid = X.index[i]
        extra = f", P(true) {proba[i]:.2f}" if fitted else ""
        with st.expander(f"{aid}: queue position {k + 1}, detector rank {base_rank[aid]}{extra}", expanded=(k == 0)):
            if aid in alerts.index:
                render_card(alerts.loc[aid])
            else:
                st.info("No explanation card: this alert is below the detector's flag threshold, so it is "
                        "shown only because feedback ranked it as worth a look.")
            ckey = f"cm_{aid}"
            st.text_input("Comment (optional)", key=ckey)
            b1, b2 = st.columns(2)
            b1.button("True issue", key=f"t_{aid}", on_click=_decide, args=(ctx, aid, True, source, ckey))
            b2.button("False alarm", key=f"f_{aid}", on_click=_decide, args=(ctx, aid, False, source, ckey))

    if fitted:
        st.subheader("What the system has learned")
        co = learner.coefficients().sort_values(key=np.abs, ascending=False).head(10)[::-1]
        fig = go.Figure(go.Bar(x=co.to_numpy(), y=list(co.index), orientation="h",
                               marker_color=["#FF4D6D" if v > 0 else "#8B9CFF" for v in co]))
        fig.update_layout(title="Signals that raise (red) or lower (blue) suspicion", height=340)
        plot(fig, key="tri_coef")

    if "y_true" in S.columns:
        with st.expander("Simulation: how much does feedback help? (uses synthetic ground truth)"):
            st.caption("A simulated monitor reviews alerts in rounds and makes occasional mistakes. "
                       "This is only possible because the data is synthetic and the truth is known.")
            batch = int(fb.get("subject" if name == "Subject profiles" else "visit", {}).get("batch", 10))
            if st.button("Run simulation"):
                st.session_state["sim"] = run_feedback_experiment(
                    X, S["y_true"].to_numpy(), base.to_numpy(), rounds=8, batch=batch, n_sims=5,
                    label_noise=float(fb.get("label_noise", 0.05)), cfg=cfg, seed=42)
            sim = st.session_state.get("sim")
            if sim:
                cv = sim["curves"]
                fig = go.Figure()
                for strat, color in (("oracle", "#E2E8F0"), ("static", "#8FA3BF"), ("feedback-top", "#2DD4BF"),
                                     ("feedback-mixed", "#FF944D")):
                    d = cv[cv.strategy == strat]
                    fig.add_scatter(x=d.reviewed, y=d.found, mode="lines+markers", name=strat,
                                    line=dict(color=color, dash="dot" if strat == "oracle" else "solid"))
                fig.update_layout(title="True issues found vs alerts reviewed", xaxis_title="alerts reviewed",
                                  yaxis_title="true issues found", height=340)
                plot(fig, key="tri_sim")
    if len(log):
        with st.expander("Decision history"):
            table(log.to_frame().tail(50))
