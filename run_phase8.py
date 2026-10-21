"""Run Phase 8: build a demo audit log, verify it, then attack copies of it (tamper test).

Uses the outputs of earlier phases (alerts.csv, site_risk_ranking.csv, ...).  The demo log lives in
data/audit/phase8_demo/ and is rebuilt on every run; the dashboard's real log is not touched.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CTMON_ROOT") or Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import yaml

from ctmon.audit import AuditLog, ConfigWatcher, hash_tables, run_tamper_suite
from ctmon.audit.hashchain import now_iso

pd.set_option("display.width", 230, "display.max_colwidth", 80, "display.max_columns", 20)
PROC = ROOT / "data" / "processed"
need = [PROC / "alerts.csv", PROC / "site_risk_ranking.csv"]
if not all(p.exists() for p in need):
    sys.exit("Run Phase 6 and Phase 7 first (they write alerts.csv and site_risk_ranking.csv), "
             "or run build_dashboard_data.py.")

demo = ROOT / "data" / "audit" / "phase8_demo"
shutil.rmtree(demo, ignore_errors=True)
demo.mkdir(parents=True)
log = AuditLog(demo / "audit_log.jsonl")

# 1) configuration: baseline, then a deliberate change on a COPY of settings.yaml
cfg_dir = demo / "cfg"
cfg_dir.mkdir()
src_cfg = ROOT / "config" / "settings.yaml"
cfg_copy = cfg_dir / "settings.yaml"
shutil.copy2(src_cfg, cfg_copy)
watcher = ConfigWatcher(log, {"settings": cfg_copy}, demo / "snapshots")
watcher.check("admin")
cfg = yaml.safe_load(cfg_copy.read_text(encoding="utf-8")) or {}
cfg.setdefault("risk", {}).setdefault("prior", {})["outlier_prob"] = 0.30
cfg_copy.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
change = watcher.check("admin")
print("=== CONFIG CHANGE RECORDED ===")
for c in change["changed"]:
    for d in c["diff"][:3]:
        print(f"  {c['file']}: {d['key']}: {d['old']} -> {d['new']}")

# 2) a run: data fingerprint, every flag, every site recommendation
raw_dir = ROOT / "data" / "raw"
sizes, sha = {}, "n/a (raw data not found)"
if (raw_dir / "subjects.csv").exists():
    tabs = {p.stem: pd.read_csv(p) for p in sorted(raw_dir.glob("*.csv"))}
    sizes, sha = {k: len(v) for k, v in tabs.items()}, hash_tables(tabs)
run_id = "RUN-DEMO"
log.log_run_start(run_id, sha, sizes, "pipeline")
alerts = pd.read_csv(PROC / "alerts.csv")
n_flags = log.log_flags(alerts, run_id, "pipeline")
n_sites = log.log_site_recommendations(pd.read_csv(PROC / "site_risk_ranking.csv"), run_id, "pipeline")
print(f"\nLogged {n_flags} flags and {n_sites} site recommendations.")

# 3) monitor decisions (from the Phase 7 demo feedback log if present, else the first alerts)
fb = PROC / "feedback_log_demo.csv"
if fb.exists():
    for r in pd.read_csv(fb).fillna("").itertuples():
        log.log_decision(r.alert_id, bool(r.is_true_issue), r.reviewer, r.comment, "ML")
else:
    for i, aid in enumerate(alerts.alert_id.head(30)):
        log.log_decision(aid, i % 3 != 0, "demo-monitor", "", "ML")
log.log_model_update({"n_labels": len(log), "learner": "logistic regression", "note": "demo"})
log.log_export("site_summary.csv", 12, {"sites": "all"}, "demo-monitor")
log.checkpoint()

rep = log.verify()
print("\n=== VERIFY (untouched log) ===")
print(rep.summary())
print(log.to_frame().tail(6)[["seq", "ts", "actor", "event", "entity_id", "hash", "prev_hash"]].to_string(index=False))

# 4) one attack in detail: flip a monitor's decision in a copy of the log
dec = next(e for e in log.entries() if e["event"] == "DECISION")
victim = demo / "tampered_copy.jsonl"
shutil.copy2(log.path, victim)
shutil.copy2(log.anchor_path, victim.with_suffix(".anchor.json"))
text = victim.read_text(encoding="utf-8")
flipped = ('"is_true_issue":false' if dec["payload"]["is_true_issue"] else '"is_true_issue":true')
original = ('"is_true_issue":true' if dec["payload"]["is_true_issue"] else '"is_true_issue":false')
victim.write_text(text.replace(original, flipped, 1), encoding="utf-8")
r2 = AuditLog(victim).verify()
print(f"\n=== ONE ATTACK IN DETAIL: flip the monitor's verdict on {dec['entity_id']} ===")
print(r2.summary())
print(r2.to_frame().to_string(index=False))

# 5) the full tamper suite
res = run_tamper_suite(log, demo / "tamper_work")
print(f"\n=== TAMPER SUITE ({res.attrs['n_entries']} entries in the log) ===")
show = res.assign(detected=res.detected.map({True: "detected", False: "NOT detected"}),
                  as_designed=res.as_designed.map({True: "yes", False: "NO"}))
print(show[["scenario", "protection", "detected", "as_designed", "problems", "first_bad_seq"]].to_string())
print("\nEvery scenario behaved as designed:", bool(res.as_designed.all()))
print("Original demo log still verifies after all attacks:", log.verify().ok)

(ROOT / "docs").mkdir(exist_ok=True)
res.to_csv(ROOT / "docs" / "phase8_tamper_results.csv", index=False)
print("\nSaved docs/phase8_tamper_results.csv")
