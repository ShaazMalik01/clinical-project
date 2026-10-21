"""Robust z-scores and plain-English wording for features (shared by cards, evaluation, feedback)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..ml.features import build_labs_view, build_vitals_view

IQR_TO_SD = 1.349

NAMES = {"sbp": "Systolic BP", "dbp": "Diastolic BP", "hr": "Heart rate",
         "weight_kg": "Body weight", "temp_c": "Body temperature", "alt": "ALT",
         "creatinine": "Creatinine", "glucose": "Glucose", "hemoglobin": "Hemoglobin"}
UNITS = {"sbp": "mmHg", "dbp": "mmHg", "hr": "bpm", "weight_kg": "kg", "temp_c": "°C",
         "alt": "U/L", "creatinine": "mg/dL", "glucose": "mg/dL", "hemoglobin": "g/dL"}

SUBJECT_LABELS = {
    "age": ("Age", "years", "raw"), "bmi": ("BMI", "kg/m²", "raw"),
    "median_sbp": ("Typical systolic BP", "mmHg", "raw"),
    "median_hr": ("Typical heart rate", "bpm", "raw"),
    "lsd_sbp": ("Visit-to-visit systolic BP variability", "", "lsd"),
    "lsd_dbp": ("Visit-to-visit diastolic BP variability", "", "lsd"),
    "lsd_hr": ("Visit-to-visit heart-rate variability", "", "lsd"),
    "lsd_weight": ("Visit-to-visit weight variability", "", "lsd"),
    "lsd_temp": ("Visit-to-visit temperature variability", "", "lsd"),
    "mean_abs_visit_dev": ("Average distance from the scheduled visit day", "days", "raw"),
    "ae_per_visit": ("Adverse events per visit", "", "raw"),
    "mean_ae_delay": ("Average adverse-event reporting delay", "days", "raw"),
    "consent_gap_days": ("Days from consent to screening visit", "days", "raw"),
    "screening_sbp": ("Screening systolic BP", "mmHg", "raw"),
    "screening_creat": ("Screening creatinine", "mg/dL", "raw"),
    "log_median_alt": ("Typical ALT", "U/L", "exp"),
}


def robust_stats(X: pd.DataFrame) -> pd.DataFrame:
    med = X.median()
    sd = ((X.quantile(0.75) - X.quantile(0.25)) / IQR_TO_SD).replace(0, np.nan)
    sd = sd.fillna(X.std(ddof=0)).replace(0, 1.0).fillna(1.0)
    return pd.DataFrame({"median": med, "sd": sd})


def zscores(X: pd.DataFrame):
    st = robust_stats(X)
    return (X - st["median"]) / st["sd"], st


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def parse_feature(f: str):
    kind = "level"
    if f.startswith("d_"):
        kind, f = "dev", f[2:]
    elif f.startswith("step_"):
        kind, f = "step", f[5:]
    log = f.startswith("log_")
    return kind, (f[4:] if log else f), log


def describe_visit_feature(f: str, x: float, z: float, med: float) -> str:
    kind, base, log = parse_feature(f)
    name = NAMES.get(base, base)
    unit = UNITS.get(base, "")
    u = f" {unit}" if unit else ""
    if kind == "level":
        val, typ = (np.exp(x), np.exp(med)) if log else (x, med)
        return f"{name} = {_fmt(val)}{u} (typical across the trial: {_fmt(typ)}; {z:+.1f} robust SD)"
    if kind == "dev":
        if log:
            return (f"{name} is {np.exp(x):.1f}x this subject's own typical value "
                    f"({z:+.1f} robust SD vs. the usual spread)")
        return (f"{name} is {x:+.1f}{u} away from this subject's own typical value "
                f"({z:+.1f} robust SD)")
    return f"{name} changed by {x:+.1f}{u} since the previous visit ({z:+.1f} robust SD)"


def describe_subject_feature(f: str, x: float, z: float, med: float) -> str:
    label, unit, mode = SUBJECT_LABELS.get(f, (f, "", "raw"))
    u = f" {unit}" if unit else ""
    if mode == "lsd":
        ratio = float(np.exp(x - med))
        how = f"only {ratio:.0%} of" if ratio < 1 else f"{ratio:.1f}x"
        return f"{label} is {how} a typical subject's ({z:+.1f} robust SD)"
    if mode == "exp":
        return f"{label} = {_fmt(np.exp(x))}{u} (typical {_fmt(np.exp(med))}; {z:+.1f} robust SD)"
    return f"{label} = {_fmt(x)}{u} (typical {_fmt(med)}; {z:+.1f} robust SD)"


@dataclass
class VisitFrame:
    keys: pd.DataFrame
    feats: pd.DataFrame
    z: pd.DataFrame
    stats: pd.DataFrame
    index: dict
    groupmax: pd.DataFrame


def _group_max(z: pd.DataFrame) -> pd.DataFrame:
    a = z.abs()
    lvl = [c for c in a.columns if not c.startswith(("d_", "step_"))]
    dev = [c for c in a.columns if c.startswith("d_")]
    stp = [c for c in a.columns if c.startswith("step_")]
    return pd.DataFrame({"z_level": a[lvl].max(axis=1), "z_dev": a[dev].max(axis=1),
                         "z_step": a[stp].max(axis=1) if stp else 0.0}, index=a.index)


def build_visit_frames(tables: dict) -> dict:
    out = {}
    for view, (keys, feats) in (("vitals", build_vitals_view(tables)),
                                ("labs", build_labs_view(tables))):
        keys, feats = keys.reset_index(drop=True), feats.reset_index(drop=True)
        z, st = zscores(feats)
        index = {(s, int(v)): i for i, (s, v) in enumerate(zip(keys["subject_id"], keys["visit_num"]))}
        out[view] = VisitFrame(keys, feats, z, st, index, _group_max(z))
    return out


def visit_z_features(frames: dict, vs: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for view, fr in frames.items():
        sub = vs[vs["view"] == view]
        if sub.empty:
            continue
        pos = [fr.index[(s, int(v))] for s, v in zip(sub["subject_id"], sub["visit_num"])]
        g = fr.groupmax.iloc[pos].reset_index(drop=True)
        g.index = sub.index
        parts.append(g)
    return pd.concat(parts).reindex(vs.index).fillna(0.0)


def top_visit_features(fr: VisitFrame, pos: int, k: int = 3) -> list:
    z, x = fr.z.iloc[pos], fr.feats.iloc[pos]
    groups: dict = {}
    for f in z.index:
        kind, base, _ = parse_feature(f)
        groups.setdefault(base, []).append((f, kind))
    rows = []
    for base, fl in groups.items():
        best = max(fl, key=lambda t: abs(z[t[0]]))[0]
        lvl = next((f for f, kd in fl if kd == "level"), None)
        shown = lvl if (lvl is not None and abs(z[lvl]) >= 3.0) else best
        rows.append((abs(float(z[best])), base, shown))
    rows.sort(reverse=True)
    out = []
    for _, base, f in rows[:k]:
        out.append({"feature": f, "base": base, "kind": parse_feature(f)[0], "z": float(z[f]),
                    "text": describe_visit_feature(f, float(x[f]), float(z[f]),
                                                   float(fr.stats.at[f, "median"]))})
    return out
