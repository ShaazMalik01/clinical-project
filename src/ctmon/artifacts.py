"""Build every file the dashboard needs (and populate the audit log) from one pipeline run.

Needs the Phase 2-7 code and data.  Usage:  python build_dashboard_data.py [--reset-audit]
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .audit import AuditLog, ConfigWatcher, hash_tables
from .data_gen import load_tables, load_yaml
from .explain import build_alert_table, build_visit_frames, save_cards_markdown
from .feedback.streams import build_subject_stream, build_visit_stream
from .pipeline import run_site_risk
from .rules import RuleEngine


def _stream_frame(X: pd.DataFrame, base: pd.Series, y) -> pd.DataFrame:
    df = X.copy()
    df["base"] = base.to_numpy()
    if y is not None:
        df["y_true"] = y
    df.index.name = "alert_id"
    return df.reset_index()


def build_artifacts(root, reset_audit: bool = False, actor: str = "pipeline") -> dict:
    root = Path(root)
    proc, audit_dir = root / "data" / "processed", root / "data" / "audit"
    proc.mkdir(parents=True, exist_ok=True)
    settings = load_yaml(root / "config" / "settings.yaml")
    protocol = load_yaml(root / "config" / "protocol.yaml")
    engine = RuleEngine.from_files(root / "config" / "rules.yaml", root / "config" / "protocol.yaml")
    raw = load_tables(root / "data" / "raw")
    gt_path = root / "data" / "ground_truth" / "anomaly_ground_truth.csv"
    gt = pd.read_csv(gt_path) if gt_path.exists() else None

    res, _counts, (rules, ml, fz) = run_site_risk(raw, settings, protocol, engine)
    frames = build_visit_frames(raw)
    alerts, cards = build_alert_table(raw, rules.deviations, ml, fz, res.site_risk, frames)

    # ---- results for the dashboard
    alerts.to_csv(proc / "alerts.csv", index=False)
    save_cards_markdown(cards, root / "docs" / "why_flagged_examples.md", per_source=3)
    rules.deviations.to_csv(proc / "rule_deviations.csv", index=False)
    rules.escalations.to_csv(proc / "rule_escalations.csv", index=False)
    ml.visit_scores.to_csv(proc / "ml_visit_scores.csv", index=False)
    ml.subject_scores.to_csv(proc / "ml_subject_scores.csv", index=False)
    ml.subject_features.to_csv(proc / "ml_subject_features.csv")
    fz.subject_scores.to_csv(proc / "forensics_subject_scores.csv", index=False)
    fz.site_scores.to_csv(proc / "forensics_site_scores.csv", index=False)
    fz.dup_pairs.to_csv(proc / "forensics_duplicate_pairs.csv", index=False)
    fz.copy_records.to_csv(proc / "forensics_copy_records.csv", index=False)
    fz.digit_hist.rename_axis(columns=None).rename(columns=str).reset_index().to_csv(proc / "forensics_digit_hist.csv", index=False)
    fz.benford_meta.to_csv(proc / "forensics_benford_meta.csv", index=False)
    if fz.benford_sites:
        pd.concat([t.assign(field=f) for f, t in fz.benford_sites.items()], ignore_index=True) \
            .to_csv(proc / "forensics_benford_sites.csv", index=False)
    elif (proc / "forensics_benford_sites.csv").exists():
        (proc / "forensics_benford_sites.csv").unlink()
    res.site_risk.to_csv(proc / "site_risk_ranking.csv", index=False)
    res.kri_table.to_csv(proc / "site_kri_posteriors.csv", index=False)
    res.priors.to_csv(proc / "kri_priors.csv", index=False)

    fb = settings.get("feedback", {})
    Xs, bs, ys, _ = build_subject_stream(ml.subject_scores, ml.subject_features, gt,
                                         fb.get("subject", {}).get("pool_frac", 0.25))
    Xv, bv, yv, _ = build_visit_stream(ml.visit_scores, frames, gt, fb.get("visit", {}).get("pool_frac", 0.06))
    _stream_frame(Xs, bs, ys).to_csv(proc / "stream_subject.csv", index=False)
    _stream_frame(Xv, bv, yv).to_csv(proc / "stream_visit.csv", index=False)

    # ---- audit trail: config state, run start, every flag, every site recommendation
    audit_dir.mkdir(parents=True, exist_ok=True)
    log = AuditLog(audit_dir / "audit_log.jsonl")
    if reset_audit:
        log.archive(audit_dir / "archive")
        log = AuditLog(audit_dir / "audit_log.jsonl")
    cfg_files = {n: root / "config" / f"{n}.yaml" for n in ("settings", "protocol", "rules")}
    ConfigWatcher(log, cfg_files, audit_dir / "config_snapshots").check(actor)
    run_id = "RUN-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    data_sha = hash_tables(raw)
    log.log_run_start(run_id, data_sha, {k: len(v) for k, v in raw.items()}, actor)
    n_flags = log.log_flags(alerts, run_id, actor)
    n_sites = log.log_site_recommendations(res.site_risk, run_id, actor)

    manifest = {"run_id": run_id, "data_sha256": data_sha, "n_subjects": int(len(raw["subjects"])),
                "n_sites": int(len(raw["sites"])), "n_alerts": int(len(alerts)),
                "n_deviations": int(len(rules.deviations)), "audit_flags_logged": n_flags,
                "audit_site_recommendations": n_sites,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (proc / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
