import numpy as np
import pandas as pd
import pytest

from ctmon.evaluation.final_eval import (CATEGORIES, COMPONENTS, MAIN_SYSTEMS, Outputs, build_truth,
                                         component_scores, evaluate_all, limitations_markdown, md_table,
                                         prf, retier, sens_fusion, sens_forensics_tiers,
                                         sens_ml_contamination, sens_rule_severity, site_level,
                                         system_frame)


def synthetic_outputs(seed=0, n_sites=8, per_site=50):
    """A toy world in which each detector is good at ONE kind of anomaly (plus a little noise)."""
    rng = np.random.default_rng(seed)
    ids = [f"S{s}-{k:03d}" for s in range(n_sites) for k in range(per_site)]
    site = {i: i.split("-")[0] for i in ids}
    subjects = pd.DataFrame({"subject_id": ids, "site_id": [site[i] for i in ids]})
    perm = list(rng.permutation(ids))
    pd_ids, out_ids, fab_ids, sae_ids, lab_ids = perm[:20], perm[20:36], perm[36:52], perm[52:60], perm[60:65]
    rows = []
    for i in pd_ids:
        rows.append(("PD", "wrong_dose", i, None, 3))
    for i in out_ids:
        rows.append(("OUT", "value_outlier", i, None, 2))
    for k, i in enumerate(fab_ids):
        rows.append(("FAB", ["low_variance", "digit_preference", "copy_forward", "duplicate_subject"][k % 4], i,
                     ids[0] if k % 4 == 3 and ids[0] not in perm[:60] else None, None))
    for i in sae_ids:
        rows.append(("SAE", "late_sae_report", i, None, None))
    for i in lab_ids:
        rows.append(("FAB", "fabricated_labs", i, None, None))
    gt = pd.DataFrame(rows, columns=["category", "subtype", "subject_id", "related_subject_id", "visit_num"])
    gt["anomaly_id"] = [f"A{i}" for i in range(len(gt))]
    gt["description"] = ""

    clean = [i for i in ids if i not in set(perm[:65])]
    fp_rules = list(rng.choice(clean, 6, replace=False))
    dev = [{"subject_id": i, "severity": str(rng.choice(["critical", "major", "minor"])), "category": "PD"}
           for i in pd_ids + sae_ids] + [{"subject_id": i, "severity": "major", "category": "OUT"}
                                          for i in out_ids[:4]] + \
          [{"subject_id": i, "severity": "minor", "category": "PD"} for i in fp_rules]
    deviations = pd.DataFrame(dev)
    deviations["site_id"] = deviations.subject_id.map(site)

    mv = []
    for i in ids:
        for v in range(1, 6):
            e = rng.random() * 0.9
            if i in out_ids and v == 2:
                e = 0.995
            mv.append({"subject_id": i, "site_id": site[i], "visit_num": v, "ensemble": e})
    ml_visits = pd.DataFrame(mv)
    ml_visits["flag"] = ml_visits.ensemble >= ml_visits.ensemble.quantile(0.98)
    se = rng.random(len(ids))
    ml_sub = pd.DataFrame({"subject_id": ids, "ensemble": se})
    ml_sub.loc[ml_sub.subject_id.isin(fab_ids[:5]), "ensemble"] = 0.97
    ml_sub["ensemble"] = ml_sub.ensemble.rank(pct=True)
    ml_sub["flag"] = ml_sub.ensemble >= ml_sub.ensemble.quantile(0.90)

    fz = pd.DataFrame({"subject_id": ids, "suspicion": rng.random(len(ids)) * 0.15,
                       "q_digit": rng.uniform(0.3, 1, len(ids)), "q_lowvar": rng.uniform(0.3, 1, len(ids)),
                       "n_copied": 0, "dup_partner": None})
    fzi = fz.set_index("subject_id")
    for k, i in enumerate(fab_ids):
        fzi.loc[i, "suspicion"] = 0.95
        fzi.loc[i, "q_lowvar" if k % 2 else "q_digit"] = 0.001
    for i in rng.choice(clean, 3, replace=False):
        fzi.loc[i, ["suspicion", "q_digit"]] = [0.6, 0.1]
    fz = fzi.reset_index()
    fz["tier"] = retier(fz, 0.05, 0.20).to_numpy()
    return Outputs(subjects, deviations, ml_visits, ml_sub, fz, gt)


@pytest.fixture(scope="module")
def out():
    return synthetic_outputs()


@pytest.fixture(scope="module")
def res(out):
    return evaluate_all(out, n_boot=120, seed=1)


def test_truth_excludes_labs_only_subjects_and_includes_duplicate_originals(out):
    truth, sub = build_truth(out.gt, out.subjects.subject_id)
    labs_only = set(out.gt[out.gt.subtype == "fabricated_labs"].subject_id)
    assert not (labs_only & set(truth.index))
    assert truth["any"].sum() == 20 + 16 + 16 + 8 or truth["any"].sum() >= 60
    assert set(sub.columns) >= {"wrong_dose", "value_outlier", "late_sae_report"}


def test_prf_matches_hand_calculation():
    m = prf([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert np.isclose(m["precision"], 2 / 3) and np.isclose(m["recall"], 2 / 3) and np.isclose(m["f1"], 2 / 3)


def test_each_detector_wins_its_own_category(res):
    c = res["category"].set_index(["category", "system"])
    assert c.loc[("PD", "rules-only"), "recall"] > 0.9
    assert c.loc[("SAE", "rules-only"), "recall"] > 0.9
    assert c.loc[("OUT", "ML-only"), "recall"] >= 0.9
    assert c.loc[("FAB", "forensics-only"), "recall"] >= 0.9
    assert c.loc[("FAB", "rules-only"), "recall"] < 0.2


def test_hybrid_recall_dominates_and_f1_wins_in_a_complementary_world(res):
    o = res["overall"]
    h = o.loc["hybrid (all three)"]
    for s in ("rules-only", "ML-only", "forensics-only"):
        assert h["recall"] >= o.loc[s, "recall"] and h["f1"] > o.loc[s, "f1"]
    d = res["boot_diff"]
    # vs the two detectors that each miss most categories the gain is clear-cut ...
    assert (d.loc[["ML-only", "forensics-only"], "P(dF1>0)"] > 0.95).all()
    assert (d.loc[["ML-only", "forensics-only"], "dF1_lo"] > 0).all()
    # ... but vs rules-only (which already covers most anomalies with few false alarms) the F1 gain is
    # small and NOT guaranteed significant: the extra recall costs precision. The report must say so.
    assert d.loc["rules-only", "dF1"] > 0


def test_ablation_lattice_is_monotone_in_recall(res):
    a = res["ablation"]
    assert len(a) == 7
    assert a.loc["rules+ML+forensics", "recall"] >= a.loc["rules+ML", "recall"] >= a.loc["rules", "recall"]
    assert (res["loo"]["recall_gain"] >= -1e-12).all()


def test_coverage_counts_add_up(res):
    by_pat, by_cat, unique = res["coverage"]
    assert by_pat["subjects"].sum() == res["prevalence"]["positive"]
    assert np.isclose(by_pat["share"].sum(), 1.0)
    assert unique.sum() <= by_pat["subjects"].sum()


def test_bootstrap_intervals_contain_point_estimates(res):
    ci = res["boot_ci"]
    assert ((ci.f1_lo <= ci.f1 + 1e-9) & (ci.f1 <= ci.f1_hi + 1e-9)).all()


def test_budget_table_is_sane(res):
    b = res["budget"]
    assert b.groupby("system").recall_at_k.apply(lambda s: s.is_monotonic_increasing).all()
    assert ((b.precision_at_k >= 0) & (b.precision_at_k <= 1)).all()


def test_sensitivity_grids_run_and_behave(out, res):
    truth = res["truth"]
    s = sens_ml_contamination(out, truth, visit_grid=(0.005, 0.05), subject_grid=(0.05, 0.20))
    ml = s[s.system == "ML-only"].set_index(["visit_q", "subject_q"])
    assert ml.loc[(0.05, 0.20), "recall"] >= ml.loc[(0.005, 0.05), "recall"]
    f = sens_forensics_tiers(out, truth)
    assert {"HIGH", "HIGH+MEDIUM"} == set(f.flag_tiers)
    r = sens_rule_severity(out, truth)
    ro = r[r.system == "rules-only"].set_index("rules_flag_from")["recall"]
    assert ro["minor"] >= ro["major"] >= ro["critical"]
    fu = sens_fusion(truth, res["comp"])
    assert fu.loc["any detector (OR)", "recall"] >= fu.loc["all 3 agree", "recall"]


def test_retier_rules():
    fz = pd.DataFrame({"subject_id": ["a", "b", "c", "d"], "q_digit": [0.01, 0.15, 0.9, 0.9],
                       "q_lowvar": [0.9, 0.9, 0.9, 0.9], "n_copied": [0, 0, 0, 2], "dup_partner": [None] * 4})
    assert retier(fz, 0.05, 0.20).tolist() == ["HIGH", "MEDIUM", "LOW", "HIGH"]


def test_site_level_ranks_each_bad_site_first_for_the_right_system():
    sg = pd.DataFrame({"site_id": ["A", "B", "C", "D"], "profile": ["clean", "sloppy_process", "fabrication", "clean"]})
    subs = pd.DataFrame({"subject_id": range(40), "site_id": list("ABCD") * 10})
    dev = pd.DataFrame({"site_id": ["B"] * 8 + ["A"], "severity": ["critical"] * 9})
    ml = pd.DataFrame({"site_id": list("ABCD"), "site_score": [1, 1.2, 1.1, 0.9]})
    fz = pd.DataFrame({"site_id": list("ABCD"), "suspicion": [0.1, 0.1, 0.9, 0.05]})
    sr = pd.DataFrame({"site_id": list("ABCD"), "risk_index": [0.1, 0.8, 0.7, 0.05],
                       "tier": ["ROUTINE", "FULL_AUDIT", "FULL_AUDIT", "ROUTINE"]})
    t = site_level(sg, subs, dev, ml, fz, sr).set_index(["bad_site_type", "system"])
    assert t.loc[("sloppy_process", "rules-only"), "rank"] == 1
    assert t.loc[("fabrication", "forensics-only"), "rank"] == 1
    assert t.loc[("fabrication", "rules-only"), "rank"] > 1


def test_markdown_helpers(res):
    md = md_table(res["overall"])
    assert md.startswith("| system |") and "hybrid (all three)" in md
    assert "synthetic" in limitations_markdown(res["prevalence"], 8).lower()
