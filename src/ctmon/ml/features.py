"""Feature engineering at three levels: visit, subject, site."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["subject_id", "visit_num"]
VITAL_COLS = ["sbp", "dbp", "hr", "weight_kg", "temp_c"]
LAB_COLS = ["log_alt", "creatinine", "glucose", "hemoglobin"]

SPREAD_FLOOR = {"sbp": 0.25, "dbp": 0.25, "hr": 0.25, "weight_kg": 0.05, "temp_c": 0.02}
SBP_CLIP = (70.0, 250.0)
CREAT_CLIP = (0.2, 15.0)

SUBJECT_MODEL_COLS = [
    "age", "bmi", "median_sbp", "median_hr",
    "lsd_sbp", "lsd_dbp", "lsd_hr", "lsd_weight", "lsd_temp",
    "mean_abs_visit_dev", "ae_per_visit", "mean_ae_delay",
    "consent_gap_days", "screening_sbp", "screening_creat", "log_median_alt",
]

SITE_AGG = {
    "lsd_sbp": "mean", "lsd_dbp": "mean", "lsd_hr": "mean",
    "mean_abs_visit_dev": "mean", "miss_rate": "mean",
    "ae_per_visit": "mean", "mean_ae_delay": "mean", "consent_gap_days": "mean",
}


def _within_subject_dev(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    med = df.groupby("subject_id")[cols].transform("median")
    dev = df[cols] - med
    dev.columns = [f"d_{c}" for c in cols]
    return dev


def build_vitals_view(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = tables["vitals"].sort_values(KEYS).reset_index(drop=True)
    df = v[["subject_id", "site_id", "visit_num"] + VITAL_COLS].copy()
    df[VITAL_COLS] = df[VITAL_COLS].astype(float)
    dev = _within_subject_dev(df, VITAL_COLS)
    step = df.groupby("subject_id")[["sbp", "hr"]].diff().fillna(0.0)
    step.columns = ["step_sbp", "step_hr"]
    feats = pd.concat([df[VITAL_COLS], dev, step], axis=1)
    return df[["subject_id", "site_id", "visit_num"]], feats


def build_labs_view(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = tables["labs"].sort_values(KEYS).reset_index(drop=True)
    df = lab[["subject_id", "site_id", "visit_num", "alt", "creatinine", "glucose", "hemoglobin"]].copy()
    df["log_alt"] = np.log(df["alt"].astype(float).clip(lower=1e-3))
    for c in ("creatinine", "glucose", "hemoglobin"):
        df[c] = df[c].astype(float)
    dev = _within_subject_dev(df, LAB_COLS)
    feats = pd.concat([df[LAB_COLS], dev], axis=1)
    return df[["subject_id", "site_id", "visit_num"]], feats


def build_subject_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    subs = tables["subjects"].set_index("subject_id")
    vit, vis, labs = tables["vitals"], tables["visits"], tables["labs"]
    ae, con = tables["adverse_events"], tables["consent"].set_index("subject_id")
    idx = subs.index

    out = pd.DataFrame(index=idx)
    out["site_id"] = subs["site_id"]
    out["age"] = subs["age"].astype(float)
    out["bmi"] = subs["bmi"].astype(float)

    g = vit.groupby("subject_id")
    out["median_sbp"] = g["sbp"].median()
    out["median_hr"] = g["hr"].median()
    med = g[VITAL_COLS].transform("median")
    mad = (vit[VITAL_COLS] - med).abs().groupby(vit["subject_id"]).median() * 1.4826
    names = {"sbp": "sbp", "dbp": "dbp", "hr": "hr", "weight_kg": "weight", "temp_c": "temp"}
    for col, short in names.items():
        out[f"lsd_{short}"] = np.log(mad[col] + SPREAD_FLOOR[col])

    done = vis[vis["status"] == "Completed"].copy()
    done["dev"] = (done["actual_date"] - done["planned_date"]).dt.days.abs()
    n_done = done.groupby("subject_id").size().reindex(idx).fillna(0)
    n_miss = vis[vis["status"] == "Missed"].groupby("subject_id").size().reindex(idx).fillna(0)
    out["mean_abs_visit_dev"] = done.groupby("subject_id")["dev"].mean()
    out["miss_rate"] = n_miss / (n_done + n_miss).clip(lower=1)

    a = (ae.assign(delay=(ae["report_date"] - ae["onset_date"]).dt.days)
           .groupby("subject_id").agg(n_ae=("ae_id", "count"), delay=("delay", "mean")))
    out["ae_per_visit"] = a["n_ae"].reindex(idx).fillna(0) / n_done.clip(lower=1)
    out["mean_ae_delay"] = a["delay"].reindex(idx)

    scr = vis[(vis["visit_num"] == 1) & (vis["status"] == "Completed")].set_index("subject_id")["actual_date"]
    out["consent_gap_days"] = (scr - con["consent_date"]).dt.days
    out["screening_sbp"] = (vit[vit["visit_num"] == 1].set_index("subject_id")["sbp"]
                            .astype(float).clip(*SBP_CLIP))
    out["screening_creat"] = (labs[labs["visit_num"] == 1].set_index("subject_id")["creatinine"]
                              .astype(float).clip(*CREAT_CLIP))
    out["log_median_alt"] = np.log(labs.groupby("subject_id")["alt"].median().clip(lower=1e-3))

    num = [c for c in out.columns if c != "site_id"]
    out[num] = out[num].fillna(out[num].median())
    return out


def build_site_features(sf: pd.DataFrame,
                        subject_flag: pd.Series | None = None,
                        visit_flags: pd.DataFrame | None = None) -> pd.DataFrame:
    g = sf.groupby("site_id")
    site = g.agg(**{c: (c, how) for c, how in SITE_AGG.items()})
    site.insert(0, "n_subjects", g.size())
    if subject_flag is not None:
        site["pct_subjects_flagged"] = subject_flag.astype(float).groupby(sf["site_id"]).mean()
    if visit_flags is not None:
        site["pct_visits_flagged"] = visit_flags.groupby("site_id")["flag"].mean()
    return site


def score_sites(site_feats: pd.DataFrame, z_clip: float = 5.0) -> pd.DataFrame:
    X = site_feats.drop(columns=["n_subjects"], errors="ignore").astype(float)
    med = X.median()
    mad = (X - med).abs().median() * 1.4826
    scale = pd.concat([mad, 0.5 * X.std(ddof=0)], axis=1).max(axis=1).replace(0, 1.0)
    z = (X - med) / scale
    score = np.sqrt((z.clip(-z_clip, z_clip) ** 2).mean(axis=1))

    def drivers(row: pd.Series) -> str:
        top = row.abs().nlargest(2).index
        return ", ".join(f"{c}({row[c]:+.1f})" for c in top)

    out = pd.DataFrame({"site_id": X.index,
                        "n_subjects": site_feats["n_subjects"].to_numpy(),
                        "site_score": score.to_numpy().round(3),
                        "top_drivers": z.apply(drivers, axis=1).to_numpy()})
    out["rank"] = out["site_score"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("rank").reset_index(drop=True)
