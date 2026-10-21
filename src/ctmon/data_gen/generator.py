"""Synthetic clinical-trial generator: produces CLEAN, protocol-compliant data.

Tables (SDTM-inspired): sites, subjects, consent, visits, vitals, labs,
dosing, adverse_events.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

TABLE_NAMES = ["sites", "subjects", "consent", "visits", "vitals",
               "labs", "dosing", "adverse_events"]

DATE_COLS = {
    "subjects": ["baseline_date"],
    "consent": ["consent_date"],
    "visits": ["planned_date", "actual_date"],
    "vitals": ["date"],
    "labs": ["date"],
    "dosing": ["date"],
    "adverse_events": ["onset_date", "report_date"],
}

COUNTRIES = ["India", "USA", "Germany", "Brazil", "Japan", "UK",
             "Spain", "Poland", "Canada", "Australia", "South Africa", "Mexico"]
SURNAMES = ["Sharma", "Miller", "Schmidt", "Silva", "Tanaka", "Evans",
            "Garcia", "Nowak", "Tremblay", "Clarke", "Naidoo", "Hernandez"]
AE_TERMS = ["Headache", "Dizziness", "Nausea", "Fatigue", "Cough",
            "Peripheral edema", "Palpitations", "Constipation", "Rash", "Back pain"]
SERIOUS_AE_TERMS = ["Myocardial infarction", "Stroke", "Hypertensive crisis",
                    "Acute kidney injury", "Syncope with injury"]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clip(x: float, bounds: list[float]) -> float:
    return float(min(max(x, bounds[0]), bounds[1]))


def _make_sites(rng: np.random.Generator, n_sites: int, n_subjects: int) -> pd.DataFrame:
    """Unequal site sizes that sum exactly to n_subjects (min 15 per site)."""
    min_n = 15
    w = rng.uniform(0.6, 1.8, n_sites)
    sizes = np.maximum(np.floor(w / w.sum() * n_subjects).astype(int), min_n)
    i = 0
    while sizes.sum() != n_subjects:
        j = i % n_sites
        if sizes.sum() < n_subjects:
            sizes[j] += 1
        elif sizes[j] > min_n:
            sizes[j] -= 1
        i += 1
    return pd.DataFrame({
        "site_id": [f"S{j + 1:02d}" for j in range(n_sites)],
        "country": [COUNTRIES[j % len(COUNTRIES)] for j in range(n_sites)],
        "pi_name": [f"Dr. {SURNAMES[j % len(SURNAMES)]}" for j in range(n_sites)],
        "planned_n": sizes,
    })


def generate_clean_trial(settings: dict, protocol: dict,
                         seed: int | None = None) -> dict[str, pd.DataFrame]:
    """Generate a fully protocol-compliant trial. Same seed -> same data."""
    seed = settings["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    t = settings["trial"]

    sites = _make_sites(rng, t["n_sites"], t["n_subjects"])
    visits_cfg = protocol["visits"]
    n_visits = len(visits_cfg)
    lab_visits = set(protocol["lab_visits"])
    dose_visits = set(protocol["dosing_visits"])
    arms = list(protocol["arms"])
    arm_p = np.array(protocol["arm_weights"], dtype=float)
    arm_p = arm_p / arm_p.sum()
    effect = protocol["simulation"]["sbp_effect_mmhg"]
    R = protocol["plausible_ranges"]
    start = pd.Timestamp(t["start_date"])

    # small, persistent site-level calibration differences (realistic)
    bias = {s: {"sbp": rng.normal(0, 1.5), "hr": rng.normal(0, 1.0),
                "alt": float(np.exp(rng.normal(0, 0.05)))}
            for s in sites.site_id}

    subj, cons, vis, vit, lab, dos, aes = [], [], [], [], [], [], []
    ae_counter = 0

    for site in sites.itertuples(index=False):
        b = bias[site.site_id]
        for k in range(1, int(site.planned_n) + 1):
            sid = f"{site.site_id}-{k:03d}"

            # ---- subject-level latent characteristics (correlated) ----
            age = int(np.clip(round(rng.normal(52, 11)), 18, 75))
            sex = "M" if rng.random() < 0.52 else "F"
            height = rng.normal(172, 7) if sex == "M" else rng.normal(160, 6.5)
            bmi = float(np.clip(rng.normal(26.5, 3.8), 19, 36))
            weight0 = bmi * (height / 100) ** 2
            arm = str(rng.choice(arms, p=arm_p))
            anchor = start + pd.Timedelta(days=int(rng.integers(0, t["enrollment_days"])))
            discontinued = rng.random() < t["dropout_rate"]
            last_visit = int(rng.integers(3, 12)) if discontinued else n_visits

            sbp0 = float(np.clip(128 + 0.25 * (age - 50) + 0.9 * (bmi - 26)
                                 + rng.normal(0, 8), 100, 150))
            dbp0 = 0.55 * sbp0 + 12 + rng.normal(0, 3)
            hr0 = float(np.clip(rng.normal(72, 8), 52, 100))
            temp0 = rng.normal(36.6, 0.15)
            creat0 = float(np.clip((0.95 if sex == "M" else 0.78)
                                   + 0.003 * (age - 50) + rng.normal(0, 0.1), 0.55, 1.3))
            alt0 = float(np.clip(np.exp(rng.normal(np.log(24), 0.35)), 9, 80))
            glu0 = float(np.clip(92 + 1.5 * (bmi - 26) + rng.normal(0, 8), 72, 125))
            hgb0 = float(np.clip(rng.normal(14.6 if sex == "M" else 13.2, 0.8), 11.5, 17))

            ar = 0.0
            last_actual = anchor
            screening_actual = None

            # ---- visit loop ----
            for v in visits_cfg:
                vn = v["num"]
                if vn > last_visit:
                    break
                w = v["window"]
                offset = int(np.clip(round(rng.normal(0, 1.3)), -w, w))
                planned = anchor + pd.Timedelta(days=v["day"])
                actual = planned + pd.Timedelta(days=offset)
                last_actual = actual
                if vn == 1:
                    screening_actual = actual

                vis.append({"subject_id": sid, "site_id": site.site_id,
                            "visit_num": vn, "visit_name": v["name"],
                            "planned_date": planned, "actual_date": actual,
                            "status": "Completed"})

                ramp = min(1.0, max(0, v["day"]) / 56.0)
                season = 2.0 * np.cos(2 * np.pi * (actual.dayofyear - 15) / 365.0)
                ar = 0.5 * ar + rng.normal(0, 3.5)
                eff = effect[arm]

                sbp = sbp0 + eff * ramp + season + b["sbp"] + ar
                dbp = dbp0 + 0.6 * eff * ramp + 0.5 * season + 0.5 * ar + rng.normal(0, 2.0)
                hr = hr0 + b["hr"] + rng.normal(0, 3.5)
                wt = weight0 + rng.normal(0, 0.5)
                temp = temp0 + rng.normal(0, 0.15)
                if vn == 1:
                    sbp = min(sbp, 158.0)   # must satisfy eligibility at screening

                vit.append({"subject_id": sid, "site_id": site.site_id,
                            "visit_num": vn, "date": actual,
                            "sbp": int(round(_clip(sbp, R["sbp"]))),
                            "dbp": int(round(_clip(dbp, R["dbp"]))),
                            "hr": int(round(_clip(hr, R["hr"]))),
                            "weight_kg": round(_clip(wt, R["weight_kg"]), 1),
                            "temp_c": round(_clip(temp, R["temp_c"]), 1)})

                if vn in lab_visits:
                    alt = alt0 * b["alt"] * np.exp(rng.normal(0, 0.12))
                    creat = creat0 + rng.normal(0, 0.04)
                    if vn == 1:
                        creat = min(creat, 1.4)
                    lab.append({"subject_id": sid, "site_id": site.site_id,
                                "visit_num": vn, "date": actual,
                                "alt": round(_clip(alt, R["alt"]), 1),
                                "creatinine": round(_clip(creat, R["creatinine"]), 2),
                                "glucose": round(_clip(glu0 + rng.normal(0, 5), R["glucose"]), 1),
                                "hemoglobin": round(_clip(hgb0 + rng.normal(0, 0.4), R["hemoglobin"]), 1)})

                if vn in dose_visits:
                    dos.append({"subject_id": sid, "site_id": site.site_id,
                                "visit_num": vn, "date": actual, "arm": arm,
                                "dose_mg": int(protocol["arms"][arm]["dose_mg"])})

            # ---- consent (given on/before the screening procedures) ----
            consent_date = screening_actual - pd.Timedelta(days=int(rng.integers(0, 3)))
            cons.append({"subject_id": sid, "site_id": site.site_id,
                         "consent_date": consent_date,
                         "icf_version": "v2.0" if consent_date >= pd.Timestamp("2025-04-01") else "v1.0"})

            subj.append({"subject_id": sid, "site_id": site.site_id, "age": age,
                         "sex": sex, "height_cm": round(float(height), 1),
                         "bmi": round(bmi, 1), "arm": arm, "baseline_date": anchor,
                         "status": "Discontinued" if discontinued else "Completed",
                         "last_visit_num": last_visit})

            # ---- adverse events (SAEs reported within the deadline) ----
            n_ae = int(rng.poisson(t["ae_rate_per_subject"] * last_visit / n_visits))
            span = max(1, (last_actual - anchor).days)
            for _ in range(n_ae):
                ae_counter += 1
                serious = bool(rng.random() < t["serious_ae_prob"])
                onset = anchor + pd.Timedelta(days=int(rng.integers(0, span + 1)))
                delay = int(rng.integers(0, protocol["safety"]["sae_report_deadline_days"] + 1)) \
                    if serious else int(rng.integers(0, 8))
                aes.append({"ae_id": f"AE-{ae_counter:05d}", "subject_id": sid,
                            "site_id": site.site_id,
                            "term": str(rng.choice(SERIOUS_AE_TERMS if serious else AE_TERMS)),
                            "severity": "Severe" if serious else str(
                                rng.choice(["Mild", "Moderate", "Severe"], p=[0.6, 0.3, 0.1])),
                            "is_serious": serious, "onset_date": onset,
                            "report_date": onset + pd.Timedelta(days=delay)})

    tables = {"sites": sites, "subjects": pd.DataFrame(subj),
              "consent": pd.DataFrame(cons), "visits": pd.DataFrame(vis),
              "vitals": pd.DataFrame(vit), "labs": pd.DataFrame(lab),
              "dosing": pd.DataFrame(dos), "adverse_events": pd.DataFrame(aes)}
    for name, df in tables.items():
        for c in DATE_COLS.get(name, []):
            df[c] = pd.to_datetime(df[c])
    return tables


def save_tables(tables: dict[str, pd.DataFrame], folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(folder / f"{name}.csv", index=False)


def load_tables(folder: str | Path) -> dict[str, pd.DataFrame]:
    folder = Path(folder)
    return {n: pd.read_csv(folder / f"{n}.csv", parse_dates=DATE_COLS.get(n, []))
            for n in TABLE_NAMES}
