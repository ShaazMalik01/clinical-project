"""Terminal-digit preference: genuine BP/HR readings have ~uniform last digits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .common import neglog10


def last_digit(x) -> np.ndarray:
    return np.abs(np.round(np.asarray(x, dtype=float))).astype(int) % 10


def subject_digit_preference(vitals: pd.DataFrame, fields, min_readings: int = 12,
                             null_rate="auto"):
    long = (vitals[["subject_id", "site_id", *fields]]
            .melt(id_vars=["subject_id", "site_id"], var_name="field", value_name="value")
            .dropna(subset=["value"]))
    long["last"] = last_digit(long["value"].to_numpy(float))
    long["round5"] = long["last"].isin([0, 5])

    site = long.groupby("site_id")["round5"].agg(k="sum", n="count").reset_index()
    site["rate"] = site["k"] / site["n"]
    p0 = float(np.median(site["rate"])) if null_rate == "auto" else float(null_rate)
    p0 = float(np.clip(p0, 0.01, 0.99))

    g = long.groupby("subject_id")["round5"].agg(n_readings="count", n_round5="sum")
    p = stats.binom.sf(g["n_round5"].to_numpy() - 1, g["n_readings"].to_numpy(), p0)
    p = np.where(g["n_readings"].to_numpy() < min_readings, 1.0, p)
    g["pct_round5"] = g["n_round5"] / g["n_readings"]
    g["p_digit"] = p
    g["s_digit"] = neglog10(p)

    hist = (pd.crosstab(long["site_id"], long["last"], normalize="index")
              .reindex(columns=range(10), fill_value=0.0))
    return g.reset_index(), p0, site, hist
