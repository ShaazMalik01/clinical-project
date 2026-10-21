"""End-to-end helpers (used by run scripts and by the dashboard builder)."""
from __future__ import annotations

from .forensics import run_forensics
from .ml import run_ml
from .risk import build_kri_counts, score_sites


def run_detection(tables: dict, settings: dict, protocol: dict, engine):
    """Run all three detection layers: rules (P3), ML (P4), forensics (P5)."""
    rules = engine.run(tables)
    ml = run_ml(tables, settings["ml"], settings["seed"])
    fz = run_forensics(tables, protocol, settings.get("forensics"), settings["seed"])
    return rules, ml, fz


def run_site_risk(tables: dict, settings: dict, protocol: dict, engine):
    """Detection -> KRIs -> Bayesian site risk. Returns (RiskResult, KRI counts, detection parts)."""
    rules, ml, fz = run_detection(tables, settings, protocol, engine)
    rcfg = settings.get("risk")
    counts = build_kri_counts(tables, rules.deviations, ml.visit_scores, fz.subject_scores,
                              protocol, (rcfg or {}).get("kri"))
    return score_sites(counts, rcfg, settings["seed"]), counts, (rules, ml, fz)
