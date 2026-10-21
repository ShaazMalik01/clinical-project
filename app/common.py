"""Shared helpers for the dashboard: data loading, theme, filters, cards, audited downloads."""
from __future__ import annotations

import html
import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from theme import style_fig

ROOT = Path(os.environ.get("CTMON_ROOT") or Path(__file__).resolve().parents[1])
PROC, RAW, GT, AUDIT_DIR = (ROOT / "data" / "processed", ROOT / "data" / "raw",
                            ROOT / "data" / "ground_truth", ROOT / "data" / "audit")

TIERS = ["ROUTINE", "REMOTE_REVIEW", "TARGETED_ONSITE", "FULL_AUDIT"]
TIER_COLOR = {"ROUTINE": "#8FA3BF", "REMOTE_REVIEW": "#F5C451", "TARGETED_ONSITE": "#FF944D",
              "FULL_AUDIT": "#FF4D6D"}
TIER_LABEL = {"ROUTINE": "Routine monitoring", "REMOTE_REVIEW": "Remote review",
              "TARGETED_ONSITE": "Targeted on-site visit", "FULL_AUDIT": "For-cause audit"}
SEVERITIES = ["critical", "major", "minor"]
SEV_COLOR = {"critical": "#FF4D6D", "major": "#FF944D", "minor": "#8FA3BF"}
SOURCES = ["RULE", "ML", "FORENSIC", "RISK"]
SOURCE_COLOR = {"RULE": "#2DD4BF", "ML": "#8B9CFF", "FORENSIC": "#D58CFF", "RISK": "#E2E8F0"}
KRI_ORDER = ["deviation_rate", "critical_deviation_rate", "query_rate", "sae_late",
             "missing_data", "anomaly_rate", "forensic_rate"]
KRI_LABEL = {"deviation_rate": "Protocol deviations", "critical_deviation_rate": "Critical deviations",
             "query_rate": "Data queries (proxy)", "sae_late": "Late SAE reports",
             "missing_data": "Missing visits", "anomaly_rate": "ML anomalies",
             "forensic_rate": "Forensic suspicion"}

FILES = {"alerts": "alerts.csv", "deviations": "rule_deviations.csv",
         "escalations": "rule_escalations.csv", "ml_visits": "ml_visit_scores.csv",
         "fz_subjects": "forensics_subject_scores.csv", "fz_sites": "forensics_site_scores.csv",
         "dup_pairs": "forensics_duplicate_pairs.csv", "copy_records": "forensics_copy_records.csv",
         "digit_hist": "forensics_digit_hist.csv", "benford_meta": "forensics_benford_meta.csv",
         "benford_sites": "forensics_benford_sites.csv", "site_risk": "site_risk_ranking.csv",
         "kri": "site_kri_posteriors.csv", "stream_subject": "stream_subject.csv",
         "stream_visit": "stream_visit.csv", "subject_features": "ml_subject_features.csv"}
REQUIRED = ["alerts", "deviations", "site_risk", "kri", "fz_subjects", "fz_sites"]

_NEW_WIDTH_API = "Width" in str(inspect.signature(st.plotly_chart).parameters.get("width"))

def plot(fig, key: str | None = None) -> None:
    style_fig(fig)
    if _NEW_WIDTH_API:
        st.plotly_chart(fig, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df: pd.DataFrame, height: int | None = None, **kw) -> None:
    args = dict(hide_index=True, **kw)
    if height:
        args["height"] = height
    if not _NEW_WIDTH_API:
        args["use_container_width"] = True
    st.dataframe(df, **args)


# ------------------------------------------------------------------ data
def _stamp() -> tuple:
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(PROC.glob("*.csv"))) if PROC.exists() else ()


@st.cache_data(show_spinner="Loading results...")
def load_bundle(stamp: tuple) -> dict:
    return {k: pd.read_csv(PROC / fn) for k, fn in FILES.items() if (PROC / fn).exists()}


@st.cache_data(show_spinner=False)
def load_raw(stamp: tuple) -> dict:
    out = {}
    for name in ("vitals", "subjects"):
        p = RAW / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def load_manifest() -> dict:
    p = PROC / "run_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_settings() -> dict:
    p = ROOT / "config" / "settings.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_truth() -> pd.DataFrame | None:
    p = GT / "site_ground_truth.csv"
    return pd.read_csv(p) if p.exists() else None


# --------------------------------------------------------------- context
@dataclass
class Ctx:
    d: dict
    raw: dict
    cfg: dict
    manifest: dict
    audit: object
    report: object
    reviewer: str
    sites: list
    sevs: list
    truth: pd.DataFrame | None
    reveal: bool
    extras: dict = field(default_factory=dict)

    def filter_summary(self) -> dict:
        return {"sites": self.sites, "severities": self.sevs}


def by_filters(df: pd.DataFrame, ctx: Ctx, site_col: str | None = "site_id",
               sev_col: str | None = "severity") -> pd.DataFrame:
    out = df
    if site_col and site_col in out.columns:
        out = out[out[site_col].isin(ctx.sites)]
    if sev_col and sev_col in out.columns:
        out = out[out[sev_col].isin(ctx.sevs)]
    return out


def csv_download(label: str, df: pd.DataFrame, filename: str, ctx: Ctx, key: str) -> None:
    """Download button that also writes an EXPORT entry to the audit log."""
    def _log() -> None:
        ctx.audit.log_export(filename, len(df), ctx.filter_summary(), ctx.reviewer)
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key, on_click=_log)


def site_summary(ctx: Ctx) -> pd.DataFrame:
    """One row per site: tier, risk index and alert/deviation/forensic counts (the CSV report)."""
    sr = ctx.d["site_risk"][["rank", "site_id", "tier", "risk_index", "ci_lo", "ci_hi", "n_subjects",
                             "breached_kris", "recommendation"]].copy()
    a = ctx.d["alerts"]
    cnt = (a.pivot_table(index="site_id", columns="severity", values="alert_id", aggfunc="count",
                         fill_value=0).reindex(columns=SEVERITIES, fill_value=0)
           .add_prefix("alerts_").reset_index())
    dev = ctx.d["deviations"].groupby("site_id").size().rename("rule_deviations").reset_index()
    fz = (ctx.d["fz_subjects"].query("tier == 'HIGH'").groupby("site_id").size()
          .rename("high_suspicion_subjects").reset_index())
    out = sr.merge(cnt, on="site_id", how="left").merge(dev, on="site_id", how="left") \
            .merge(fz, on="site_id", how="left")
    out["breached_kris"] = out["breached_kris"].fillna("")
    out["recommendation"] = out["recommendation"].fillna("")
    num_cols = out.select_dtypes("number").columns
    out[num_cols] = out[num_cols].fillna(0)
    for c in out.columns:
        if c.startswith("alerts_") or c in ("rule_deviations", "high_suspicion_subjects"):
            out[c] = out[c].astype(int)
    return out.sort_values("rank")


# -------------------------------------------------------------- rendering
def tag(text: str, color: str) -> str:
    return f"<span class='tag' style='background:{color}'>{html.escape(text)}</span>"


def tier_tag(tier: str) -> str:
    return tag(TIER_LABEL.get(tier, tier), TIER_COLOR.get(tier, "#8793A0"))


def render_card(row) -> None:
    """The 'Why flagged' card for one alert."""
    sev = str(row["severity"])
    ev = [e.strip() for e in str(row.get("evidence", "")).split("||") if e.strip()]
    items = "".join(f"<li>{html.escape(e)}</li>" for e in ev)
    ref = f"<div class='meta'>Reference: {html.escape(str(row['reference']))}</div>" \
        if str(row.get("reference", "")) not in ("", "nan") else ""
    nxt = f"<div class='meta'><b>Suggested next step:</b> {html.escape(str(row.get('next_step', '')))}</div>"
    st.markdown(
        f"<div class='card' style='border-left:5px solid {SEV_COLOR.get(sev, '#8793A0')}'>"
        f"{tag(sev, SEV_COLOR.get(sev, '#8793A0'))}{tag(str(row['source']), SOURCE_COLOR.get(row['source'], '#555'))}"
        f"<h4>{html.escape(str(row['headline']))}</h4>"
        f"<p class='why'><b>Why flagged:</b> {html.escape(str(row['summary']))}</p>"
        f"<b>Evidence</b><ul>{items}</ul>{ref}{nxt}</div>", unsafe_allow_html=True)


def chain_badge(rep) -> str:
    n = 9
    bad_at = None if rep.ok else min(n - 1, int((rep.first_bad_seq or 1) / max(rep.n_entries, 1) * (n - 1)))
    parts = []
    for i in range(n):
        color = "#2DD4BF" if (rep.ok or i < bad_at) else ("#FF4D6D" if i == bad_at else "#5B6B80")
        parts.append(f"<span class='blk' style='background:{color};color:{color}'></span>")
        if i < n - 1:
            parts.append("<span class='lnk'></span>")
    head = rep.head_hash[:12] if rep.head_hash else "-"
    label = "Audit chain verified" if rep.ok else "Audit chain BROKEN"
    sub = f"{rep.n_entries:,} entries, head {head}" if rep.ok else f"first problem at entry {rep.first_bad_seq}"
    return (f"<div><b>{label}</b><div class='chain-mini'>{''.join(parts)}</div><div class='small'>{sub}</div></div>")


def goto(page: str, site: str | None = None) -> None:
    """Button callback: jump to another page (optionally focusing a site)."""
    st.session_state["nav"] = page
    if site:
        st.session_state["focus_site"] = site
