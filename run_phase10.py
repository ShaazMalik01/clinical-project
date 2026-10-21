"""Run Phase 10: final evaluation against the injected ground truth.

    python run_phase10.py                 # main dataset + 2 untouched replicate datasets, 300 bootstrap draws
    python run_phase10.py --seeds 0       # main dataset only (faster)
    python run_phase10.py --boot 1000     # tighter bootstrap intervals

Needs the Phase 2-7 code and data (data/raw, data/ground_truth).  Takes a few minutes.
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from ctmon.data_gen import load_tables, load_yaml
from ctmon.data_gen.anomaly_injector import AnomalyInjector
from ctmon.data_gen.crp_patch import add_crp
from ctmon.data_gen.generator import generate_clean_trial
from ctmon.evaluation.final_eval import (MAIN_SYSTEMS, Outputs, evaluate_all, sens_forensics_tiers,
                                         sens_ml_contamination, sens_rule_severity, site_level)
from ctmon.evaluation.report import save_report, why_hybrid_text
from ctmon.forensics import run_forensics
from ctmon.ml import run_ml
from ctmon.pipeline import run_site_risk
from ctmon.rules import RuleEngine

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=2, help="number of untouched replicate datasets")
ap.add_argument("--boot", type=int, default=300, help="bootstrap draws")
args = ap.parse_args()

pd.set_option("display.width", 230, "display.max_columns", 30, "display.max_colwidth", 60)
settings = load_yaml(ROOT / "config" / "settings.yaml")
protocol = load_yaml(ROOT / "config" / "protocol.yaml")
engine = RuleEngine.from_files(ROOT / "config" / "rules.yaml", ROOT / "config" / "protocol.yaml")
seed = settings["seed"]


def to_outputs(tables, rules, ml, fz, gt) -> Outputs:
    return Outputs(tables["subjects"][["subject_id", "site_id"]], rules.deviations, ml.visit_scores,
                   ml.subject_scores, fz.subject_scores, gt)


# ============================================================ main dataset
raw = load_tables(ROOT / "data" / "raw")
gt = pd.read_csv(ROOT / "data" / "ground_truth" / "anomaly_ground_truth.csv")
site_gt = pd.read_csv(ROOT / "data" / "ground_truth" / "site_ground_truth.csv")
print("Running rules + ML + forensics + site risk on the main dataset...")
risk, _counts, (rules, ml, fz) = run_site_risk(raw, settings, protocol, engine)
out = to_outputs(raw, rules, ml, fz, gt)
res = evaluate_all(out, n_boot=args.boot, seed=seed)
truth, comp = res["truth"], res["comp"]
p = res["prevalence"]
print(f"\nEvaluated {p['subjects']} subjects; {p['positive']} anomalous "
      f"({', '.join(f'{c} {p[c]}' for c in ['PD', 'OUT', 'FAB', 'SAE'])}).")

print("\n=== 1. OVERALL (any anomaly) ===")
print(res["overall"].round(3).to_string())
print("\nBootstrap 95% intervals:")
print(res["boot_ci"].round(3).to_string())
print("\nPaired difference, hybrid minus each single detector:")
print(res["boot_diff"].round(3).to_string())

print("\n=== 2. PER CATEGORY (precision/F1 against clean subjects) ===")
print(res["category"].round(3).to_string(index=False))
print("\n=== 3. RECALL PER INJECTED ANOMALY TYPE ===")
print(res["subtype"].round(2).to_string())

print("\n=== 4. ABLATION: every combination of detectors ===")
print(res["ablation"].round(3).to_string())
print("\nMarginal value of each component (hybrid minus hybrid-without-it):")
print(res["loo"].round(3).to_string())

print("\n=== 5. DOES THE HYBRID WIN, AND WHY? ===")
for line in why_hybrid_text(res):
    print(line)
by_pat, by_cat, uniq = res["coverage"]
print("\nWhich detectors caught each anomalous subject:")
print(by_pat.round(3).to_string())

print("\n=== 6. SAME WORKLOAD FOR EVERY SYSTEM (review the top b% of subjects) ===")
print(res["budget"].pivot(index="budget", columns="system", values="recall_at_k").round(3).to_string())
print("\nFusion rule:")
print(res["fusion"].round(3).to_string())

print("\n=== 7. SENSITIVITY ===")
sens = {"ml": sens_ml_contamination(out, truth), "forensics": sens_forensics_tiers(out, truth),
        "rules": sens_rule_severity(out, truth), "fusion": res["fusion"]}
h = "hybrid (all three)"
for name, key, cols in (("ML review budget", "ml", ["visit_q", "subject_q"]),
                        ("forensic tier cut-offs", "forensics", ["high_q", "medium_q", "flag_tiers"])):
    d = sens[key][sens[key]["system"] == h]
    print(f"Hybrid F1 across the {name} grid: min {d.f1.min():.3f}, max {d.f1.max():.3f}, "
          f"at the configured setting see the grid below")
    print(d[cols + ["precision", "recall", "f1"]].round(3).to_string(index=False))
print("\nRule severity threshold:")
print(sens["rules"].round(3).to_string(index=False))

print("\n=== 8. SITE LEVEL (rank of each truly bad site; 1 = riskiest) ===")
site_tbl = site_level(site_gt, raw["subjects"], rules.deviations, ml.site_scores, fz.site_scores, risk.site_risk)
print(site_tbl.round(3).to_string(index=False))

# ============================================================ replicate datasets
multiseed = None
if args.seeds > 0:
    reps = []
    for k in range(args.seeds):
        s_rep = 101 * (k + 1)
        print(f"\nReplicate dataset, seed {s_rep} (no tuning on it)...")
        st = copy.deepcopy(settings)
        st["seed"] = s_rep
        clean = generate_clean_trial(st, protocol)
        t, g, sg = AnomalyInjector(clean, protocol, st, seed=s_rep + 1).run()
        clean, t, g, sg = add_crp(clean, t, g, sg, seed=s_rep + 5)
        r_rules = engine.run(t)
        r_ml = run_ml(t, st["ml"], s_rep)
        r_fz = run_forensics(t, protocol, st.get("forensics"), s_rep)
        reps.append(evaluate_all(to_outputs(t, r_rules, r_ml, r_fz, g), n_boot=0))
    rows = []
    for system in MAIN_SYSTEMS:
        row = {"system": system}
        for m in ("precision", "recall", "f1", "auc"):
            vals = [r["overall"].loc[system, m] for r in reps]
            row[f"{m} main"] = round(float(res["overall"].loc[system, m]), 3)
            row[f"{m} replicates"] = f"{np.mean(vals):.3f} +/- {np.std(vals, ddof=1) if len(vals) > 1 else 0:.3f}"
        rows.append(row)
    multiseed = pd.DataFrame(rows).set_index("system")
    print("\n=== 9. MAIN DATASET vs UNTOUCHED REPLICATES ===")
    print(multiseed.to_string())
    best_single = lambda r: r["overall"].drop(index=h)["f1"].max()      # noqa: E731
    wins = sum(r["overall"].loc[h, "f1"] > best_single(r) for r in reps)
    dom = sum(r["overall"].loc[h, "recall"] >= r["overall"]["recall"].max() - 1e-12 for r in reps)
    print(f"\nHybrid F1 beats the best single detector in {wins} of {len(reps)} replicates; "
          f"hybrid recall is the highest in {dom} of {len(reps)}.")

# ============================================================ save
ev = ROOT / "data" / "processed" / "eval"
ev.mkdir(parents=True, exist_ok=True)
for name in ("overall", "boot_ci", "boot_diff", "category", "subtype", "ablation", "loo", "budget", "fusion"):
    res[name].to_csv(ev / f"{name}.csv")
for name, df in sens.items():
    df.to_csv(ev / f"sensitivity_{name}.csv")
site_tbl.to_csv(ev / "site_level.csv", index=False)
save_report(res, sens, site_tbl, multiseed, ROOT, n_sites=int(raw["sites"].shape[0]),
            notes=f"Main dataset seed {seed}; thresholds fixed before evaluation.")
print("\nSaved: data/processed/eval/*.csv, docs/evaluation_report.md, docs/LIMITATIONS.md, "
      "docs/phase10_main.png, docs/phase10_sensitivity.png")
