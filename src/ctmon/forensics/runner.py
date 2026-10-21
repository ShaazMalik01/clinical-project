"""Orchestrates all forensic tests and produces subject- and site-level suspicion scores."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .benford import benford_by_site, benford_suitability
from .common import plausible_vitals
from .copy_forward import detect_copy_forward
from .digit_pref import subject_digit_preference
from .duplicates import find_near_duplicates
from .scoring import score_sites, score_subjects
from .variability import score_low_variance, subject_variability

DEFAULTS = {
    "plausible_filter": True,
    "digit": {"fields": ["sbp", "dbp", "hr"], "min_readings": 12, "null_rate": "auto"},
    "variance": {"fields": ["sbp", "dbp", "hr"], "min_visits": 8, "tail_df": 6},
    "copy_forward": {"fields": ["sbp", "dbp", "hr", "weight_kg", "temp_c"]},
    "duplicates": {"age_tol": 1, "height_tol_cm": 1.0, "bmi_tol": 0.5, "min_shared_visits": 4,
                   "similarity_threshold": 0.75, "ref_pairs": 3000,
                   "lab_fields": ["alt", "creatinine", "glucose", "hemoglobin", "crp"]},
    "benford": {"candidate_fields": ["alt", "creatinine", "glucose", "hemoglobin", "crp"],
                "min_n": 300, "min_span_ratio": 100, "aggregate": "first", "mc_sims": 20000},
    "scoring": {"s0": 1.3, "tau": 2.0, "deterministic_evidence": 6.0, "high_q": 0.05,
                "medium_q": 0.20, "subject_nominal_p": 0.01},
}


def _merge(base: dict, over: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass
class ForensicsResult:
    subject_scores: pd.DataFrame
    site_scores: pd.DataFrame
    variability: pd.DataFrame
    digit_subject: pd.DataFrame
    digit_hist: pd.DataFrame
    copy_records: pd.DataFrame
    dup_pairs: pd.DataFrame
    benford_meta: pd.DataFrame
    benford_sites: dict
    null_digit_rate: float


def run_benford(tables: dict, cfg: dict, seed: int):
    labs = tables["labs"]
    meta, tabs = [], {}
    for f in cfg["candidate_fields"]:
        if f not in labs.columns:
            continue
        s = benford_suitability(labs[f], cfg["min_n"], cfg["min_span_ratio"])
        status = f"not suitable: {s['reason']}"
        if s["suitable"]:
            df = labs[["site_id", "subject_id", "visit_num", f]].rename(columns={f: "value"})
            site = benford_by_site(df, cfg["mc_sims"], seed, cfg["aggregate"])
            if len(site) and float(site["p_benford"].median()) >= 0.05:
                tabs[f], status = site, "applied"
            else:
                status = "skipped: most sites reject Benford, so a deviation would not be informative"
        meta.append({"field": f, **{k: s[k] for k in ("n", "p1", "p99", "span_ratio", "suitable")},
                     "status": status})
    site_p = None
    if tabs:
        allp = pd.concat([t.set_index("site_id")["p_benford"] for t in tabs.values()], axis=1)
        site_p = (allp.min(axis=1) * len(tabs)).clip(upper=1.0)
    return site_p, tabs, pd.DataFrame(meta)


def run_forensics(tables: dict, protocol: dict | None = None, cfg: dict | None = None,
                  seed: int = 42) -> ForensicsResult:
    c = _merge(DEFAULTS, cfg)
    d, v, cf, du, bf, sc = c["digit"], c["variance"], c["copy_forward"], c["duplicates"], c["benford"], c["scoring"]

    fields = sorted(set(d["fields"]) | set(v["fields"]))
    vit = (plausible_vitals(tables["vitals"], protocol, fields)
           if protocol and c["plausible_filter"] else tables["vitals"].copy())

    digit_subj, p0, digit_site, hist = subject_digit_preference(vit, d["fields"], d["min_readings"], d["null_rate"])
    var = score_low_variance(subject_variability(vit, v["fields"], v["min_visits"]),
                             v["fields"], tail_df=v["tail_df"])
    copy_rec, copy_subj = detect_copy_forward(tables["vitals"], cf["fields"])
    dup_pairs, dup_subj = find_near_duplicates(tables, protocol, du, seed)
    ben_p, ben_tabs, ben_meta = run_benford(tables, bf, seed)

    subj = score_subjects(tables["subjects"], digit_subj, var, copy_subj, dup_subj, p0, sc)
    site = score_sites(tables["subjects"], subj, digit_site, ben_p, sc)
    return ForensicsResult(subj, site, var, digit_subj, hist, copy_rec, dup_pairs,
                           ben_meta, ben_tabs, p0)
