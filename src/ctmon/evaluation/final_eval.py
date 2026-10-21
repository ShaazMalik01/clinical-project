"""Phase 10: end-to-end evaluation of rules, ML, forensics and the hybrid against injected ground truth.

Unit of evaluation = SUBJECT.  A subject is positive if at least one anomaly was injected on it
(duplicates: both the clone and its original).  Every function takes plain DataFrames, so the
module can be tested without running the pipeline.

Decisions fixed BEFORE looking at any result (so they cannot be tuned to flatter the hybrid):
  * a system flags a subject as: rules = >=1 deviation; ML = subject flagged or any of its visits
    flagged; forensics = HIGH or MEDIUM tier; hybrid = union (OR) of the components it contains
  * ranking score: single systems use their own score; combinations use noisy-OR of the evidence
    values below, tie-broken by the mean of the native scores
  * subjects whose only anomaly is the site-level 'fabricated_labs' pattern are left out of the
    subject-level evaluation (neither positive nor negative); that pattern is scored at site level
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

COMPONENTS = ["rules", "ml", "forensics"]
LABEL = {"rules": "rules", "ml": "ML", "forensics": "forensics"}
CATEGORIES = ["PD", "OUT", "FAB", "SAE"]
CATEGORY_NAME = {"PD": "protocol deviation", "OUT": "value outlier", "FAB": "fabrication", "SAE": "late SAE report"}
MAIN_SYSTEMS = {"rules-only": ("rules",), "ML-only": ("ml",), "forensics-only": ("forensics",),
                "hybrid (all three)": ("rules", "ml", "forensics")}
SEV_RANK = {"minor": 1, "major": 2, "critical": 3}
SEV_SCORE = {"critical": 1.0, "major": 0.66, "minor": 0.33}      # native rule score
SEV_EVID = {"critical": 0.9, "major": 0.7, "minor": 0.4}          # evidence for fusion
ML_EVIDENCE_CAP, ML_EVIDENCE_FLOOR = 0.8, 0.9                     # only the top 10% of ML ranks count


@dataclass
class Outputs:
    subjects: pd.DataFrame        # subject_id, site_id (every subject in the evaluated dataset)
    deviations: pd.DataFrame      # rule engine: subject_id, severity, ...
    ml_visits: pd.DataFrame       # subject_id, visit_num, ensemble, flag
    ml_subjects: pd.DataFrame     # subject_id, ensemble, flag
    fz_subjects: pd.DataFrame     # subject_id, suspicion, tier, q_digit, q_lowvar, n_copied, dup_partner
    gt: pd.DataFrame              # injected anomalies (anomaly_ground_truth.csv)


# ============================================================ helpers
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    cols = [str(c) for c in d.columns]

    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def safe_auc(y, s) -> float:
    y = np.asarray(y, bool)
    return float("nan") if y.sum() == 0 or y.all() else float(roc_auc_score(y, s))


def safe_ap(y, s) -> float:
    y = np.asarray(y, bool)
    return float("nan") if y.sum() == 0 else float(average_precision_score(y, s))


def prf(y, flag) -> dict:
    y, f = np.asarray(y, bool), np.asarray(flag, bool)
    tp, fp, fn = int((y & f).sum()), int((~y & f).sum()), int((y & ~f).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "n_flagged": tp + fp}


def subset_name(subset) -> str:
    return "+".join(LABEL[c] for c in subset)


# ============================================================ ground truth
def build_truth(gt: pd.DataFrame, subject_ids):
    """(category table, subtype table), both indexed by subject_id and aligned."""
    ids = pd.Index(list(subject_ids))
    g = gt[gt["subtype"] != "fabricated_labs"]
    rows = [(r.subject_id, r.category, r.subtype) for r in g.itertuples()]
    rows += [(r.related_subject_id, r.category, r.subtype) for r in g[g["subtype"] == "duplicate_subject"].itertuples()
             if isinstance(r.related_subject_id, str)]
    long = pd.DataFrame(rows, columns=["subject_id", "category", "subtype"]).drop_duplicates()
    long = long[long["subject_id"].isin(set(ids))]
    T = pd.DataFrame({c: ids.isin(set(long.loc[long["category"] == c, "subject_id"])) for c in CATEGORIES}, index=ids)
    T["any"] = T[CATEGORIES].any(axis=1)
    S = pd.DataFrame({s: ids.isin(set(long.loc[long["subtype"] == s, "subject_id"]))
                      for s in sorted(long["subtype"].unique())}, index=ids)
    ambiguous = set(gt.loc[gt["subtype"] == "fabricated_labs", "subject_id"]) - set(long["subject_id"])
    keep = ~ids.isin(ambiguous)
    return T[keep], S[keep]


# ============================================================ component scores
def component_scores(out: Outputs, rule_min_severity: str = "minor", fz_tiers=("HIGH", "MEDIUM"),
                     ml_flag: pd.Series | None = None, fz_tier: pd.Series | None = None) -> pd.DataFrame:
    """Per-subject score, flag and fusion evidence for each of the three detectors."""
    ids = pd.Index(list(out.subjects["subject_id"]))
    comp = pd.DataFrame(index=ids)

    dev = out.deviations
    if len(dev):
        dev = dev[dev["severity"].map(SEV_RANK) >= SEV_RANK[rule_min_severity]]
    if len(dev):
        g = dev.groupby("subject_id")
        sev = g["severity"].agg(lambda s: max(s, key=SEV_RANK.get))
        n = g.size()
        score = (sev.map(SEV_SCORE) * 0.95 + 0.05 * np.minimum(n, 5) / 5).reindex(ids).fillna(0.0)
        evid = sev.map(SEV_EVID).reindex(ids).fillna(0.0)
    else:
        score, evid = pd.Series(0.0, index=ids), pd.Series(0.0, index=ids)
    comp["rules_score"], comp["e_rules"], comp["rules_flag"] = score.to_numpy(), evid.to_numpy(), (score > 0).to_numpy()

    ms = out.ml_subjects.set_index("subject_id")
    vis = out.ml_visits.groupby("subject_id").agg(v=("ensemble", "max"), f=("flag", "max"))
    sub_score = ms["ensemble"].reindex(ids).fillna(0.0)
    ml_score = np.maximum(sub_score.to_numpy(), vis["v"].reindex(ids).fillna(0.0).to_numpy())
    if ml_flag is None:
        flag = (ms["flag"].reindex(ids).fillna(False).astype(bool).to_numpy()
                | vis["f"].reindex(ids).fillna(False).astype(bool).to_numpy())
    else:
        flag = ml_flag.reindex(ids).fillna(False).astype(bool).to_numpy()
    comp["ml_score"], comp["ml_flag"] = ml_score, flag
    comp["e_ml"] = ML_EVIDENCE_CAP * np.clip((ml_score - ML_EVIDENCE_FLOOR) / (1 - ML_EVIDENCE_FLOOR), 0, 1)

    fz = out.fz_subjects.set_index("subject_id")
    tiers = (fz_tier if fz_tier is not None else fz["tier"]).reindex(ids)
    comp["forensics_score"] = fz["suspicion"].reindex(ids).fillna(0.0).to_numpy()
    comp["e_forensics"] = comp["forensics_score"]
    comp["forensics_flag"] = tiers.isin(list(fz_tiers)).to_numpy()
    return comp


def system_frame(comp: pd.DataFrame, subset) -> tuple:
    """(ranking score, boolean flag) of a system made of the given components."""
    subset = tuple(subset)
    flag = np.zeros(len(comp), bool)
    for c in subset:
        flag |= comp[f"{c}_flag"].to_numpy(bool)
    if len(subset) == 1:
        return comp[f"{subset[0]}_score"].to_numpy(float), flag
    keep = np.ones(len(comp))
    for c in subset:
        keep *= 1.0 - comp[f"e_{c}"].to_numpy(float)
    native = np.mean([comp[f"{c}_score"].to_numpy(float) for c in subset], axis=0)
    return 0.999 * (1.0 - keep) + 0.001 * native, flag


def vote_system(comp: pd.DataFrame, k: int = 2) -> tuple:
    """Alternative fusion: flag only when at least k of the three detectors agree."""
    score, _ = system_frame(comp, COMPONENTS)
    votes = sum(comp[f"{c}_flag"].to_numpy(bool).astype(int) for c in COMPONENTS)
    return score, votes >= k


# ============================================================ metric tables
def overall_table(truth: pd.DataFrame, comp: pd.DataFrame, systems: dict | None = None) -> pd.DataFrame:
    y = truth["any"].to_numpy()
    rows = []
    for name, subset in (systems or MAIN_SYSTEMS).items():
        score, flag = system_frame(comp, subset)
        rows.append({"system": name, **prf(y, flag), "auc": safe_auc(y, score), "ap": safe_ap(y, score)})
    return pd.DataFrame(rows).set_index("system")[["n_flagged", "precision", "recall", "f1", "auc", "ap"]]


def category_table(truth: pd.DataFrame, comp: pd.DataFrame, systems: dict | None = None) -> pd.DataFrame:
    """Per anomaly category. Precision/F1 are judged against clean subjects only (flags on subjects that
    carry a different category of anomaly are neither right nor wrong here); AUC separates the category's
    subjects from clean subjects."""
    clean = ~truth["any"].to_numpy()
    rows = []
    for cat in CATEGORIES:
        pos = truth[cat].to_numpy()
        if pos.sum() == 0:
            continue
        m = pos | clean
        for name, subset in (systems or MAIN_SYSTEMS).items():
            score, flag = system_frame(comp, subset)
            tp, fp, fn = int((flag & pos).sum()), int((flag & clean).sum()), int((~flag & pos).sum())
            rows.append({"category": cat, "system": name, "n_pos": int(pos.sum()),
                         "precision": tp / (tp + fp) if tp + fp else float("nan"),
                         "recall": tp / pos.sum(), "f1": 2 * tp / (2 * tp + fp + fn),
                         "auc": safe_auc(pos[m], score[m])})
    return pd.DataFrame(rows)


def subtype_recall(subtypes: pd.DataFrame, comp: pd.DataFrame, systems: dict | None = None) -> pd.DataFrame:
    rows = []
    for st in subtypes.columns:
        pos = subtypes[st].to_numpy()
        if pos.sum() == 0:
            continue
        row = {"subtype": st, "n": int(pos.sum())}
        for name, subset in (systems or MAIN_SYSTEMS).items():
            row[name] = float(system_frame(comp, subset)[1][pos].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("subtype")


def ablation_lattice(truth: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """Every non-empty combination of {rules, ML, forensics}: shows what each component adds or costs."""
    y = truth["any"].to_numpy()
    rows = []
    for k in (1, 2, 3):
        for subset in combinations(COMPONENTS, k):
            score, flag = system_frame(comp, subset)
            row = {"components": subset_name(subset), "k": k, **prf(y, flag), "auc": safe_auc(y, score)}
            for cat in CATEGORIES:
                pos = truth[cat].to_numpy()
                row[f"recall_{cat}"] = float(flag[pos].mean()) if pos.sum() else float("nan")
            rows.append(row)
    cols = ["components", "k", "n_flagged", "precision", "recall", "f1", "auc"] + [f"recall_{c}" for c in CATEGORIES]
    return pd.DataFrame(rows)[cols].set_index("components")


def leave_one_out(abl: pd.DataFrame) -> pd.DataFrame:
    """Marginal value of each component = hybrid minus the hybrid without it."""
    full = abl.loc[subset_name(COMPONENTS)]
    rows = []
    for c in COMPONENTS:
        rest = subset_name([x for x in COMPONENTS if x != c])
        r = abl.loc[rest]
        rows.append({"component": LABEL[c], "recall_gain": full["recall"] - r["recall"],
                     "precision_change": full["precision"] - r["precision"],
                     "f1_gain": full["f1"] - r["f1"], "auc_gain": full["auc"] - r["auc"]})
    return pd.DataFrame(rows).set_index("component")


def coverage_patterns(truth: pd.DataFrame, comp: pd.DataFrame):
    """Which detectors caught each truly anomalous subject? This is the 'why hybrid' evidence."""
    pos = truth["any"].to_numpy()
    flags = {LABEL[c]: comp[f"{c}_flag"].to_numpy(bool)[pos] for c in COMPONENTS}
    pattern = pd.Series(["+".join(n for n in flags if flags[n][i]) or "missed by all" for i in range(int(pos.sum()))],
                        index=truth.index[pos])
    by_pat = pattern.value_counts().rename("subjects").to_frame()
    by_pat["share"] = by_pat["subjects"] / by_pat["subjects"].sum()
    cat = pd.DataFrame({c: pattern[truth.loc[pattern.index, c]].value_counts() for c in CATEGORIES}).fillna(0).astype(int)
    unique = pd.Series({n: int(((pattern == n)).sum()) for n in flags}, name="caught_only_by_this")
    return by_pat, cat, unique


def at_budget(truth: pd.DataFrame, comp: pd.DataFrame, budgets=(0.05, 0.10, 0.15, 0.20), seed: int = 0) -> pd.DataFrame:
    """Fair comparison at equal workload: review the top b% of subjects of each system."""
    y = truth["any"].to_numpy()
    N, P = len(y), int(y.sum())
    rng = np.random.default_rng(seed)
    rows = []
    for name, subset in MAIN_SYSTEMS.items():
        score, _ = system_frame(comp, subset)
        order = np.argsort(-(score + 1e-9 * rng.random(N)), kind="stable")
        for b in budgets:
            k = int(np.ceil(b * N))
            tp = int(y[order[:k]].sum())
            rows.append({"system": name, "budget": b, "k": k, "precision_at_k": tp / k, "recall_at_k": tp / max(P, 1)})
    return pd.DataFrame(rows)


# ============================================================ uncertainty
def bootstrap(truth: pd.DataFrame, comp: pd.DataFrame, n_boot: int = 300, seed: int = 0):
    """Resample subjects with replacement: CIs for F1 and AUC, and PAIRED differences hybrid - single."""
    y = truth["any"].to_numpy()
    N = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(n_boot, N))
    yb = y[idx]
    res: dict = {}
    for name, subset in MAIN_SYSTEMS.items():
        score, flag = system_frame(comp, subset)
        fb, sb = flag[idx], score[idx]
        tp, fp, fn = (yb & fb).sum(1), (~yb & fb).sum(1), (yb & ~fb).sum(1)
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
        auc = np.array([safe_auc(yb[i], sb[i]) for i in range(n_boot)])
        res[name] = {"f1": f1, "auc": auc, "f1_point": prf(y, flag)["f1"], "auc_point": safe_auc(y, score)}
    rows = []
    for name, r in res.items():
        rows.append({"system": name, "f1": r["f1_point"], "f1_lo": np.nanpercentile(r["f1"], 2.5),
                     "f1_hi": np.nanpercentile(r["f1"], 97.5), "auc": r["auc_point"],
                     "auc_lo": np.nanpercentile(r["auc"], 2.5), "auc_hi": np.nanpercentile(r["auc"], 97.5)})
    ci = pd.DataFrame(rows).set_index("system")
    hyb = res["hybrid (all three)"]
    drows = []
    for name, r in res.items():
        if name.startswith("hybrid"):
            continue
        d_f1, d_auc = hyb["f1"] - r["f1"], hyb["auc"] - r["auc"]
        drows.append({"hybrid minus": name, "dF1": hyb["f1_point"] - r["f1_point"],
                      "dF1_lo": np.nanpercentile(d_f1, 2.5), "dF1_hi": np.nanpercentile(d_f1, 97.5),
                      "P(dF1>0)": float(np.nanmean(d_f1 > 0)),
                      "dAUC": hyb["auc_point"] - r["auc_point"], "dAUC_lo": np.nanpercentile(d_auc, 2.5),
                      "dAUC_hi": np.nanpercentile(d_auc, 97.5), "P(dAUC>0)": float(np.nanmean(d_auc > 0))})
    return ci, pd.DataFrame(drows).set_index("hybrid minus")


def roc_points(truth: pd.DataFrame, comp: pd.DataFrame) -> dict:
    y = truth["any"].to_numpy()
    return {n: roc_curve(y, system_frame(comp, s)[0])[:2] for n, s in MAIN_SYSTEMS.items()}


# ============================================================ sensitivity
def sens_ml_contamination(out: Outputs, truth: pd.DataFrame, visit_grid=(0.005, 0.01, 0.02, 0.03, 0.05),
                          subject_grid=(0.05, 0.10, 0.15, 0.20)) -> pd.DataFrame:
    """Re-threshold the stored ensemble scores (no refit): how fragile is the review-budget choice?"""
    y = truth["any"].to_numpy()
    rows = []
    for qv in visit_grid:
        v_thr = out.ml_visits["ensemble"].quantile(1 - qv)
        v_flag = out.ml_visits[out.ml_visits["ensemble"] >= v_thr].groupby("subject_id").size() > 0
        for qs in subject_grid:
            s_thr = out.ml_subjects["ensemble"].quantile(1 - qs)
            s_flag = out.ml_subjects.set_index("subject_id")["ensemble"] >= s_thr
            flag = s_flag.reindex(truth.index).fillna(False) | v_flag.reindex(truth.index).fillna(False)
            comp = component_scores(out, ml_flag=flag).loc[truth.index]
            for name in ("ML-only", "hybrid (all three)"):
                score, f = system_frame(comp, MAIN_SYSTEMS[name])
                rows.append({"visit_q": qv, "subject_q": qs, "system": name, **prf(y, f)})
    return pd.DataFrame(rows)


def retier(fz: pd.DataFrame, high_q: float, medium_q: float) -> pd.Series:
    qmin = np.minimum(fz["q_digit"].fillna(1.0), fz["q_lowvar"].fillna(1.0))
    det = (fz["n_copied"].fillna(0) > 0) | fz["dup_partner"].notna()
    t = np.where(det | (qmin < high_q), "HIGH", np.where(qmin < medium_q, "MEDIUM", "LOW"))
    return pd.Series(t, index=fz["subject_id"].to_numpy())


def sens_forensics_tiers(out: Outputs, truth: pd.DataFrame, high_grid=(0.01, 0.05, 0.10),
                         medium_grid=(0.10, 0.20, 0.30)) -> pd.DataFrame:
    y = truth["any"].to_numpy()
    rows = []
    for hq in high_grid:
        for mq in medium_grid:
            if mq < hq:
                continue
            tier = retier(out.fz_subjects, hq, mq)
            for tiers in (("HIGH",), ("HIGH", "MEDIUM")):
                comp = component_scores(out, fz_tiers=tiers, fz_tier=tier).loc[truth.index]
                for name in ("forensics-only", "hybrid (all three)"):
                    rows.append({"high_q": hq, "medium_q": mq, "flag_tiers": "+".join(tiers), "system": name,
                                 **prf(y, system_frame(comp, MAIN_SYSTEMS[name])[1])})
    return pd.DataFrame(rows)


def sens_rule_severity(out: Outputs, truth: pd.DataFrame) -> pd.DataFrame:
    y = truth["any"].to_numpy()
    rows = []
    for sev in ("minor", "major", "critical"):
        comp = component_scores(out, rule_min_severity=sev).loc[truth.index]
        for name in ("rules-only", "hybrid (all three)"):
            rows.append({"rules_flag_from": sev, "system": name, **prf(y, system_frame(comp, MAIN_SYSTEMS[name])[1])})
    return pd.DataFrame(rows)


def sens_fusion(truth: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """Union (any detector) vs majority vote (>=2 agree) vs unanimous (all 3)."""
    y = truth["any"].to_numpy()
    rows = []
    for label, k in (("any detector (OR)", 1), ("at least 2 agree", 2), ("all 3 agree", 3)):
        _, flag = vote_system(comp, k)
        rows.append({"fusion": label, **prf(y, flag)})
    return pd.DataFrame(rows).set_index("fusion")


# ============================================================ site level
def site_level(site_gt: pd.DataFrame, subjects: pd.DataFrame, deviations: pd.DataFrame, ml_sites: pd.DataFrame,
               fz_sites: pd.DataFrame, site_risk: pd.DataFrame) -> pd.DataFrame:
    n = subjects.groupby("site_id").size()
    crit = (deviations[deviations["severity"] == "critical"].groupby("site_id").size().reindex(n.index).fillna(0) / n)
    sc = {"rules-only": crit,
          "ML-only": ml_sites.set_index("site_id")["site_score"],
          "forensics-only": fz_sites.set_index("site_id")["suspicion"],
          "hybrid (Bayesian risk model)": site_risk.set_index("site_id")["risk_index"]}
    tier = site_risk.set_index("site_id")["tier"]
    rows = []
    for target in ("sloppy_process", "fabrication"):
        ids = site_gt.loc[site_gt["profile"] == target, "site_id"]
        if ids.empty:
            continue
        site = ids.iloc[0]
        for name, s in sc.items():
            s = s.reindex(site_gt["site_id"]).fillna(0.0)
            y = (s.index == site)
            rows.append({"bad_site_type": target, "system": name, "rank": int((s > s[site]).sum()) + 1,
                         "of": len(s), "auc": safe_auc(y, s.to_numpy()),
                         "tier": tier.get(site, "") if name.startswith("hybrid") else ""})
    return pd.DataFrame(rows)


# ============================================================ orchestration
def evaluate_all(out: Outputs, n_boot: int = 300, seed: int = 0) -> dict:
    truth, subtypes = build_truth(out.gt, out.subjects["subject_id"])
    comp = component_scores(out).loc[truth.index]
    res = {"truth": truth, "subtypes": subtypes, "comp": comp,
           "prevalence": {"subjects": len(truth), "positive": int(truth["any"].sum()),
                          **{c: int(truth[c].sum()) for c in CATEGORIES}},
           "overall": overall_table(truth, comp), "category": category_table(truth, comp),
           "subtype": subtype_recall(subtypes, comp), "ablation": ablation_lattice(truth, comp),
           "budget": at_budget(truth, comp, seed=seed), "fusion": sens_fusion(truth, comp)}
    res["loo"] = leave_one_out(res["ablation"])
    res["coverage"] = coverage_patterns(truth, comp)
    if n_boot:
        res["boot_ci"], res["boot_diff"] = bootstrap(truth, comp, n_boot, seed)
    return res


def limitations_markdown(prev: dict, n_sites: int) -> str:
    pos, n = prev["positive"], prev["subjects"]
    return f"""# Limitations (read this before quoting any number)

**1. The data is synthetic.** Every result comes from a simulator we wrote and anomalies we injected.
No real clinical-trial data was used, and nothing here has been validated on a real trial. Real data has
transcription noise, missing values, mixed units, protocol amendments and audit trails that this simulator
does not model.

**2. Circularity.** The same person designed the anomalies and the detectors. Copy-forward, digit
preference and low variability were injected *because* the forensic tests look for them, so forensic recall
is optimistic. Real fabrication is subtler and adapts to known checks. A fair test needs anomalies
written by someone who has not seen the detectors.

**3. Small sample, high prevalence.** {n} evaluated subjects at {n_sites} sites, {pos} of them anomalous
({pos / max(n, 1):.0%}). Real trials have far fewer true problems, so real precision would be much lower
at the same recall. Confidence intervals come from bootstrapping subjects; they do not capture
simulator misspecification. Only two sites are 'bad', so site-level metrics rest on two examples.

**4. Thresholds were set on the main dataset.** Contamination rates, tier cut-offs and fusion rules were
chosen on seed 42. The replicate seeds in the multi-seed table were not used for any tuning, which is why
they are the more honest numbers.

**5. Hybrid design choices are ours.** The evidence weights used to fuse detectors (see the module
docstring) are fixed by judgement, not learned. Union fusion raises recall by construction; the
interesting question is what it costs in precision, which the ablation table shows.

**6. Subject-level evaluation hides record-level errors.** A subject counts as detected if the detector
flags the subject, even if it points at a different visit than the injected anomaly.

**7. The monitor in the feedback loop is simulated.** Real reviewers disagree, tire, and are biased by the
queue order.

**8. Not a validated GxP system.** A real deployment needs computerised-system validation (GAMP 5),
change control for model updates, access control, and electronic-signature controls beyond this hash-chained
log.

**9. A flag is a reason to look, not a finding of misconduct.** Statistical suspicion must go through
human review and documented follow-up before any conclusion about a site or person.
"""
