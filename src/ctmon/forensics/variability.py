"""Variance-too-low test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .common import neglog10, robust_z


def subject_variability(vitals: pd.DataFrame, fields, min_visits: int = 8) -> pd.DataFrame:
    rows = []
    v = vitals.sort_values(["subject_id", "visit_num"])
    for sid, g in v.groupby("subject_id", sort=False):
        rec = {"subject_id": sid, "site_id": g["site_id"].iloc[0]}
        for f in fields:
            x = g[f].to_numpy(float)
            t = g["visit_num"].to_numpy(float)
            m = np.isfinite(x)
            x, t = x[m], t[m]
            sd = np.nan
            if len(x) >= min_visits:
                A = np.column_stack([np.ones(len(x)), t])
                coef = np.linalg.lstsq(A, x, rcond=None)[0]
                r = x - A @ coef
                sd = float(np.sqrt(r @ r / (len(x) - 2)))
            rec[f"sd_{f}"] = sd
        rows.append(rec)
    return pd.DataFrame(rows)


def score_low_variance(var: pd.DataFrame, fields, floor_sd: float = 0.05,
                       tail_df: int = 6) -> pd.DataFrame:
    out = var.copy()
    z_cols = []
    for f in fields:
        sd = out[f"sd_{f}"].clip(lower=floor_sd)
        out[f"ratio_{f}"] = sd / np.nanmedian(sd.to_numpy())
        out[f"z_{f}"] = robust_z(np.log(sd.to_numpy() ** 2))
        z_cols.append(f"z_{f}")
    valid = out[z_cols].notna().all(axis=1).to_numpy()
    z = np.zeros(len(out))
    if valid.any():
        z[valid] = robust_z(out.loc[valid, z_cols].sum(axis=1).to_numpy())
    out["z_lowvar"] = z
    out["p_lowvar"] = np.where(valid, stats.t.cdf(z, df=tail_df), 1.0)
    out["s_lowvar"] = neglog10(out["p_lowvar"])
    out["var_ratio"] = np.exp(np.log(out[[f"ratio_{f}" for f in fields]]).mean(axis=1))
    return out
