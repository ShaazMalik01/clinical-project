"""Build a small, self-contained fake results bundle so the dashboard can be tried (and smoke-tested)
without running the whole pipeline.  Not real trial data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ctmon.audit import AuditLog

SITES = ["S01", "S02", "S03", "S04", "S05", "S06"]
TIER = {"S02": "FULL_AUDIT", "S05": "TARGETED_ONSITE", "S03": "REMOTE_REVIEW"}
KRIS = ["deviation_rate", "critical_deviation_rate", "query_rate", "sae_late", "missing_data",
        "anomaly_rate", "forensic_rate"]


def make_fake_bundle(root, seed: int = 0) -> Path:
    root = Path(root)
    rng = np.random.default_rng(seed)
    for sub in ("config", "data/processed", "data/raw", "data/ground_truth", "data/audit"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "config" / "settings.yaml").write_text(yaml.safe_dump({
        "seed": 42, "risk": {"tiers": {"remote_review": {"index": 0.15, "prob": 0.5},
                                       "targeted_onsite": {"index": 0.3, "prob": 0.7},
                                       "full_audit": {"index": 0.6, "prob": 0.8}}},
        "feedback": {"label_noise": 0.05, "learner": {"C": 0.5, "min_labels": 8, "n0": 20.0},
                     "subject": {"batch": 6}, "visit": {"batch": 6}}}))
    (root / "config" / "protocol.yaml").write_text("trial_id: FAKE\n")
    (root / "config" / "rules.yaml").write_text("rules: []\n")
    P = root / "data" / "processed"

    subs = [f"{s}-{k:03d}" for s in SITES for k in range(1, 13)]
    site_of = {s: s[:3] for s in subs}
    pd.DataFrame({"subject_id": subs, "site_id": [site_of[s] for s in subs]}).to_csv(
        root / "data" / "raw" / "subjects.csv", index=False)
    rows = []
    for s in subs:
        for v in range(1, 9):
            rows.append({"subject_id": s, "site_id": site_of[s], "visit_num": v,
                         "sbp": int(rng.normal(130, 8)), "dbp": int(rng.normal(80, 6)),
                         "hr": int(rng.normal(72, 7))})
    pd.DataFrame(rows).to_csv(root / "data" / "raw" / "vitals.csv", index=False)
    pd.DataFrame({"site_id": SITES, "profile": ["clean", "sloppy_process", "clean", "clean",
                                                "fabrication", "clean"],
                  "is_bad_site": [False, True, False, False, True, False]}).to_csv(
        root / "data" / "ground_truth" / "site_ground_truth.csv", index=False)

    def card(i, src, sev, sid, vn, head):
        return {"alert_id": i, "level": "record", "source": src, "site_id": site_of.get(sid, sid) if sid else "S02",
                "subject_id": sid, "visit_num": vn, "severity": sev, "priority": round(float(rng.random()), 3),
                "headline": head, "summary": f"{src} detector flagged this item.",
                "evidence": "Value is far from typical || Second supporting fact",
                "reference": "ICH E6 4.9.1", "next_step": "Check the source document.",
                "card": f"[{sev.upper()}] {head}"}
    al = []
    for i in range(30):
        sid = subs[int(rng.integers(0, len(subs)))]
        al.append(card(f"RUL-D{i:05d}", "RULE", ["critical", "major", "minor"][i % 3], sid, 1 + i % 8, f"Rule alert {i}"))
    for i in range(30):
        sid = subs[int(rng.integers(0, len(subs)))]
        al.append(card(f"MLV-{sid}-V{1 + i % 8}", "ML", ["major", "minor"][i % 2], sid, 1 + i % 8, f"Unusual visit {i}"))
    for i in range(20):
        al.append(card(f"MLS-{subs[i]}", "ML", "minor", subs[i], None, f"Unusual subject {subs[i]}"))
    for i in range(10):
        al.append(card(f"FOR-{subs[20 + i]}", "FORENSIC", "major", subs[20 + i], None, f"Integrity issue {i}"))
    for s, t in TIER.items():
        a = card(f"SIT-{s}", "RISK", "critical", None, None, f"Site {s}: {t}")
        a["site_id"] = s
        al.append(a)
    alerts = pd.DataFrame(al).drop_duplicates("alert_id")
    alerts.to_csv(P / "alerts.csv", index=False)

    devs = alerts[alerts.source == "RULE"].copy()
    pd.DataFrame({"deviation_id": [f"D{i:05d}" for i in range(len(devs))],
                  "rule_id": ["ELIG-AGE", "DOSE-01", "VIS-WINDOW"] * 10, "rule_name": "Rule name",
                  "category": ["PD", "PD", "OUT"] * 10, "severity": devs.severity.to_numpy(),
                  "subject_id": devs.subject_id.to_numpy(), "site_id": devs.site_id.to_numpy(),
                  "visit_num": devs.visit_num.to_numpy(), "record_id": None, "table": "vitals",
                  "field": "sbp", "observed": "250", "expected": "<= 160", "evidence": "SBP too high",
                  "reference": "ICH E6 4.5.1", "detects": "x"}).to_csv(P / "rule_deviations.csv", index=False)
    pd.DataFrame({"rule_id": ["META-SITE-HOTSPOT"], "rule_name": ["hotspot"], "level": ["site"],
                  "entity_id": ["S02"], "site_id": ["S02"], "score": [0.4], "detail": ["many critical"],
                  "action": ["audit"]}).to_csv(P / "rule_escalations.csv", index=False)

    tiers = ["HIGH" if i % 12 == 0 else ("MEDIUM" if i % 7 == 0 else "LOW") for i in range(len(subs))]
    pd.DataFrame({"subject_id": subs, "site_id": [site_of[s] for s in subs],
                  "suspicion": rng.random(len(subs)).round(3), "primary_pattern": "digit_preference",
                  "tier": tiers, "reasons": "58% of readings end in 0 or 5"}).to_csv(
        P / "forensics_subject_scores.csv", index=False)
    pd.DataFrame({"site_id": SITES, "suspicion": [0.1, 0.3, 0.1, 0.05, 0.95, 0.1],
                  "tier": ["LOW", "LOW", "LOW", "LOW", "HIGH", "LOW"],
                  "reasons": ["", "", "", "", "terminal-digit preference; Benford deviation", ""],
                  "rank": [3, 2, 4, 6, 1, 5]}).to_csv(P / "forensics_site_scores.csv", index=False)
    dh = pd.DataFrame(np.full((6, 10), 0.1), columns=[str(i) for i in range(10)])
    dh.loc[4, ["0", "5"]] = [0.3, 0.3]
    dh.insert(0, "site_id", SITES)
    dh.to_csv(P / "forensics_digit_hist.csv", index=False)
    pd.DataFrame({"field": ["crp", "alt"], "n": [600, 600], "p1": [0.1, 10], "p99": [90, 80],
                  "span_ratio": [900, 8], "suitable": [True, False],
                  "status": ["applied", "not suitable: range too narrow"]}).to_csv(
        P / "forensics_benford_meta.csv", index=False)
    ben = np.log10(1 + 1 / np.arange(1, 10))
    bs = pd.DataFrame([{"field": "crp", "site_id": s, "n": 100, "chi2": 5.0, "p_benford": 0.4,
                        "mad": 0.004, "mad_class": "close", **{f"d{i + 1}": ben[i] for i in range(9)}}
                       for s in SITES])
    bs.loc[bs.site_id == "S05", [f"d{i}" for i in range(1, 10)]] = 1 / 9
    bs.to_csv(P / "forensics_benford_sites.csv", index=False)
    pd.DataFrame({"subject_a": ["S01-001"], "subject_b": ["S02-001"], "pair_score": [0.9],
                  "flagged": [True]}).to_csv(P / "forensics_duplicate_pairs.csv", index=False)
    pd.DataFrame({"subject_id": ["S05-002"], "site_id": ["S05"], "visit_num": [5], "group_size": [3],
                  "source_visit": [4]}).to_csv(P / "forensics_copy_records.csv", index=False)

    idx = [0.05, 0.9, 0.2, 0.04, 0.45, 0.06]
    sr = pd.DataFrame({"rank": 0, "site_id": SITES, "n_subjects": 12, "risk_index": idx,
                       "ci_lo": [max(i - 0.05, 0) for i in idx], "ci_hi": [i + 0.1 for i in idx],
                       "p_ge_remote": 0.5, "p_ge_onsite": 0.3, "p_ge_audit": 0.1, "n_breached": 0,
                       "breached_kris": "", "naive_index": idx, "tier": [TIER.get(s, "ROUTINE") for s in SITES],
                       "recommendation": "Do the recommended action.", "drivers": "Protocol deviations: 20% vs peers 8%",
                       "reason": "risk index above threshold"})
    sr.loc[sr.site_id == "S02", "breached_kris"] = "forensic_rate"
    sr = sr.sort_values("risk_index", ascending=False).reset_index(drop=True)
    sr["rank"] = range(1, 7)
    sr.to_csv(P / "site_risk_ranking.csv", index=False)
    kr = [{"site_id": s, "kri": k, "events": int(rng.integers(0, 6)), "opportunities": 50,
           "raw_rate": 0.05, "post_mean": 0.06, "ci_lo": 0.02, "ci_hi": 0.12, "ref_rate": 0.05,
           "risk_ratio": float(rng.uniform(0.5, 4)), "p_exceed": 0.3, "p_breach": 0.1, "p_outlier": 0.1,
           "data_weight": 0.5, "contribution": 0.01} for s in SITES for k in KRIS]
    pd.DataFrame(kr).to_csv(P / "site_kri_posteriors.csv", index=False)

    for name, ids in (("stream_subject", [f"MLS-{s}" for s in subs[:50]]),
                      ("stream_visit", [f"MLV-{s}-V{1 + i % 8}" for i, s in enumerate(subs[:50])])):
        f = rng.normal(size=(50, 5))
        y = (f[:, 1] + 0.5 * rng.normal(size=50) > 0.6).astype(int)
        df = pd.DataFrame(f, columns=[f"f{i}" for i in range(5)])
        df.insert(0, "alert_id", ids)
        df["base"] = f[:, 0] * 0.2 + y * 0.3 + rng.random(50)
        df["y_true"] = y
        df.to_csv(P / f"{name}.csv", index=False)

    sf = pd.DataFrame(rng.normal(size=(len(subs), 6)), columns=[f"feat{i}" for i in range(6)])
    sf.insert(0, "site_id", [site_of[x] for x in subs])
    sf.insert(0, "subject_id", subs)
    sf.to_csv(P / "ml_subject_features.csv", index=False)

    (P / "run_manifest.json").write_text(json.dumps({"run_id": "RUN-FAKE", "n_subjects": len(subs),
                                                     "n_sites": 6, "data_sha256": "0" * 64}))
    log = AuditLog(root / "data" / "audit" / "audit_log.jsonl")
    log.log_run_start("RUN-FAKE", "0" * 64, {"subjects": len(subs)})
    log.log_flags(alerts.head(15), "RUN-FAKE")
    for i in range(6):
        log.log_decision(f"RUL-D{i:05d}", bool(i % 2), "demo", "", "RULE")
    return root
