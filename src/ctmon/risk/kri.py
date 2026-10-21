"""Key Risk Indicators (KRIs). Every KRI is 'events out of opportunities'."""
from __future__ import annotations

import numpy as np
import pandas as pd

KRI_ORDER = ["deviation_rate", "critical_deviation_rate", "query_rate", "sae_late",
             "missing_data", "anomaly_rate", "forensic_rate"]

KRI_INFO = {
    "deviation_rate": {
        "label": "Protocol deviation rate",
        "events": "subjects with >=1 major/critical protocol deviation (missed visits excluded)",
        "opportunities": "subjects", "source": "Phase 3 rule engine"},
    "critical_deviation_rate": {
        "label": "Critical deviation rate",
        "events": "subjects with >=1 critical deviation (eligibility, consent, gross dosing error)",
        "opportunities": "subjects", "source": "Phase 3 rule engine"},
    "query_rate": {
        "label": "Data query rate (proxy)",
        "events": "visit records tripping an automated edit check (range / cross-field)",
        "opportunities": "visit records", "source": "Phase 3 data-plausibility rules"},
    "sae_late": {
        "label": "SAE reporting lateness",
        "events": "serious adverse events reported after the deadline",
        "opportunities": "serious adverse events", "source": "adverse_events table"},
    "missing_data": {
        "label": "Missing visits",
        "events": "missed visits", "opportunities": "scheduled visits",
        "source": "visits table"},
    "anomaly_rate": {
        "label": "ML anomaly rate",
        "events": "visits in the global top-2% ML anomaly scores",
        "opportunities": "scored visits", "source": "Phase 4 ML ensemble"},
    "forensic_rate": {
        "label": "Forensic suspicion rate",
        "events": "subjects with HIGH fabrication-suspicion tier",
        "opportunities": "subjects", "source": "Phase 5 forensics"},
}


def _per_site(s: pd.Series, sites: list) -> pd.Series:
    return s.reindex(sites).fillna(0).astype(int)


def build_kri_counts(tables: dict, deviations: pd.DataFrame, visit_scores: pd.DataFrame,
                     forensic_subjects: pd.DataFrame, protocol: dict,
                     cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or {}
    sites = tables["sites"]["site_id"].tolist()
    n_subj = _per_site(tables["subjects"].groupby("site_id").size(), sites)

    dcfg = cfg.get("deviation", {})
    sev = set(dcfg.get("severities", ["major", "critical"]))
    excl = set(dcfg.get("exclude_rules", ["VIS-MISSED"]))

    pdv = deviations[deviations["category"] == "PD"]
    x_dev = _per_site(pdv[pdv["severity"].isin(sev) & ~pdv["rule_id"].isin(excl)]
                      .groupby("site_id")["subject_id"].nunique(), sites)
    x_crit = _per_site(pdv[(pdv["severity"] == "critical") & ~pdv["rule_id"].isin(excl)]
                       .groupby("site_id")["subject_id"].nunique(), sites)

    q = deviations[deviations["category"] == "OUT"][["site_id", "subject_id", "visit_num"]].drop_duplicates()
    n_rec = _per_site(tables["vitals"].groupby("site_id").size(), sites)
    x_q = np.minimum(_per_site(q.groupby("site_id").size(), sites), n_rec)

    ae = tables["adverse_events"]
    ser = ae[ae["is_serious"].astype(bool)].copy()
    deadline = protocol["safety"]["sae_report_deadline_days"]
    ser["late"] = (ser["report_date"] - ser["onset_date"]).dt.days > deadline
    n_sae = _per_site(ser.groupby("site_id").size(), sites)
    x_sae = _per_site(ser[ser["late"]].groupby("site_id").size(), sites)

    vis = tables["visits"]
    n_vis = _per_site(vis.groupby("site_id").size(), sites)
    x_miss = _per_site(vis[vis["status"] == "Missed"].groupby("site_id").size(), sites)

    n_ml = _per_site(visit_scores.groupby("site_id").size(), sites)
    x_ml = _per_site(visit_scores[visit_scores["flag"].astype(bool)].groupby("site_id").size(), sites)

    n_f = _per_site(forensic_subjects.groupby("site_id").size(), sites)
    x_f = _per_site(forensic_subjects[forensic_subjects["tier"] == "HIGH"]
                    .groupby("site_id").size(), sites)

    parts = {"deviation_rate": (x_dev, n_subj), "critical_deviation_rate": (x_crit, n_subj),
             "query_rate": (x_q, n_rec), "sae_late": (x_sae, n_sae),
             "missing_data": (x_miss, n_vis), "anomaly_rate": (x_ml, n_ml),
             "forensic_rate": (x_f, n_f)}
    rows = []
    for k in KRI_ORDER:
        x, n = parts[k]
        for s in sites:
            rows.append({"site_id": s, "kri": k, "events": int(min(x[s], n[s])),
                         "opportunities": int(n[s])})
    return pd.DataFrame(rows)
