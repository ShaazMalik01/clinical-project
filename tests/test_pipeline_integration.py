"""End-to-end check of phases 2-7 on a small generated trial (about 20 seconds).

Replaces the per-phase test files: each test below pins one headline property of one phase.
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ctmon.data_gen.anomaly_injector import AnomalyInjector
from ctmon.data_gen.crp_patch import add_crp
from ctmon.data_gen.generator import generate_clean_trial, load_yaml
from ctmon.evaluation.feedback_eval import explanation_fidelity
from ctmon.evaluation.forensics_eval import evaluate_dup_pairs, evaluate_subjects
from ctmon.evaluation.ml_eval import evaluate_visit_level
from ctmon.evaluation.risk_eval import evaluate_site_risk
from ctmon.evaluation.rule_eval import evaluate_rules
from ctmon.explain import build_alert_table, build_visit_frames
from ctmon.pipeline import run_site_risk
from ctmon.rules import RuleEngine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world():
    s = load_yaml(ROOT / "config" / "settings.yaml")
    p = load_yaml(ROOT / "config" / "protocol.yaml")
    s = copy.deepcopy(s)
    s["trial"]["n_sites"], s["trial"]["n_subjects"] = 8, 320
    s["injection"]["counts"] = {k: max(4, v // 2) for k, v in s["injection"]["counts"].items()}
    clean = generate_clean_trial(s, p)
    raw, gt, sg = AnomalyInjector(clean, p, s, seed=s["seed"] + 1).run()
    clean, raw, gt, sg = add_crp(clean, raw, gt, sg, seed=s["seed"] + 5)
    engine = RuleEngine.from_files(ROOT / "config" / "rules.yaml", ROOT / "config" / "protocol.yaml")
    res, _, (rules, ml, fz) = run_site_risk(raw, s, p, engine)
    return dict(s=s, p=p, clean=clean, raw=raw, gt=gt, sg=sg, engine=engine, res=res, rules=rules, ml=ml, fz=fz)


def test_phase2_clean_data_is_reproducible_and_anomalies_are_labelled(world):
    s, p = world["s"], world["p"]
    again = generate_clean_trial(s, p)
    assert again["vitals"].equals(world["clean"]["vitals"])
    assert set(world["gt"].category) == {"PD", "OUT", "FAB", "SAE"}
    assert world["sg"].is_bad_site.sum() == 2


def test_phase3_rules_are_silent_on_clean_data_and_exact_on_covered_anomalies(world):
    eng = world["engine"]
    assert len(eng.run(world["clean"]).deviations) == 0
    per, overall = evaluate_rules(world["rules"].deviations, world["gt"], eng.covered_subtypes())
    r = dict(zip(per["group"], per["recall"]))
    for g in ("ineligible_enrolled", "procedure_before_consent", "visit_out_of_window", "missed_visit",
              "wrong_dose", "late_sae_report", "value_outlier [impossible]"):
        assert r[g] == 1.0, g
    assert overall["precision"] == 1.0
    assert all(r[g] == 0.0 for g in ("copy_forward", "low_variance", "digit_preference"))


def test_phase4_ml_separates_extreme_values(world):
    table, _ = evaluate_visit_level(world["ml"].visit_scores, world["gt"],
                                    world["s"]["ml"]["contamination"]["visit"])
    assert table.loc["ens_prior", "AUC"] > 0.95


def test_phase5_forensics_finds_fabrication_and_is_quiet_on_clean_data(world):
    fz = world["fz"]
    summ, per = evaluate_subjects(fz.subject_scores, world["gt"])
    assert summ["AUC"] > 0.95 and summ["precision_HIGH"] >= 0.9
    assert evaluate_dup_pairs(fz.dup_pairs, world["gt"])["recall"] >= 0.8
    from ctmon.forensics import run_forensics
    quiet = run_forensics(world["clean"], world["p"], world["s"]["forensics"], world["s"]["seed"])
    assert (quiet.subject_scores.tier == "HIGH").sum() == 0 and int(quiet.dup_pairs.flagged.sum()) == 0


def test_phase6_bayesian_risk_ranks_both_bad_sites_first(world):
    m, summ = evaluate_site_risk(world["res"].site_risk, world["sg"])
    assert (m.loc[m.is_bad_site, "rank_by_index"] <= 2).all()
    assert summ["clean_targeted_or_audit"] == 0


def test_phase7_cards_are_complete_and_faithful(world):
    frames = build_visit_frames(world["raw"])
    alerts, cards = build_alert_table(world["raw"], world["rules"].deviations, world["ml"], world["fz"],
                                      world["res"].site_risk, frames)
    assert {"RULE", "ML", "FORENSIC"} <= set(alerts.source) and alerts.alert_id.is_unique
    assert all(c.headline and c.summary and c.evidence for c in cards)
    fid, _ = explanation_fidelity(frames, world["ml"].visit_scores, world["gt"])
    assert fid["n"] > 0 and fid["hit_at_k"] >= 0.9
