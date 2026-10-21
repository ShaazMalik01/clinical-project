"""Benford's law with an applicability gate (chi-square with Monte-Carlo p-value + Nigrini's MAD)."""
from __future__ import annotations

import numpy as np
import pandas as pd

BENFORD = np.log10(1.0 + 1.0 / np.arange(1, 10))


def first_digit(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    e = np.floor(np.log10(x))
    return np.clip(np.floor(x / np.power(10.0, e)).astype(int), 1, 9)


def mad_class(mad: float) -> str:
    if mad <= 0.006:
        return "close"
    if mad <= 0.012:
        return "acceptable"
    if mad <= 0.015:
        return "marginal"
    return "nonconformity"


def benford_test(x, sims: int = 20000, seed: int = 0) -> dict:
    d = first_digit(x)
    n = len(d)
    if n == 0:
        return {"n": 0, "chi2": np.nan, "p_value": np.nan, "mad": np.nan,
                "mad_class": "n/a", "props": np.full(9, np.nan)}
    counts = np.bincount(d, minlength=10)[1:10]
    exp = BENFORD * n
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    mad = float(np.mean(np.abs(counts / n - BENFORD)))
    sim = np.random.default_rng(seed).multinomial(n, BENFORD, size=sims)
    sim_chi = (((sim - exp) ** 2) / exp).sum(axis=1)
    p = float((1 + np.sum(sim_chi >= chi2 - 1e-12)) / (sims + 1))
    return {"n": n, "chi2": chi2, "p_value": p, "mad": mad,
            "mad_class": mad_class(mad), "props": counts / n}


def benford_suitability(x, min_n: int = 300, min_span_ratio: float = 100.0) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    pos = x[x > 0]
    out = {"n": int(len(pos)), "p1": np.nan, "p99": np.nan, "span_ratio": np.nan,
           "suitable": False, "reason": "no positive values"}
    if len(pos) < 2:
        return out
    p1, p99 = np.percentile(pos, [1, 99])
    span = float(p99 / p1) if p1 > 0 else float("inf")
    out.update(p1=float(p1), p99=float(p99), span_ratio=span)
    if len(pos) / len(x) < 0.99:
        out["reason"] = "contains zero/negative values"
    elif len(pos) < min_n:
        out["reason"] = f"too few values ({len(pos)} < {min_n})"
    elif span < min_span_ratio:
        out["reason"] = f"range too narrow (P99/P1 = {span:.1f} < {min_span_ratio:g})"
    else:
        out.update(suitable=True, reason="ok")
    return out


def benford_by_site(df: pd.DataFrame, sims: int = 20000, seed: int = 0,
                    aggregate: str = "first") -> pd.DataFrame:
    d = df.dropna(subset=["value"])
    d = d[d["value"] > 0]
    if aggregate == "first":
        d = d.sort_values(["subject_id", "visit_num"]).groupby("subject_id", as_index=False).first()
    elif aggregate == "median":
        d = d.groupby(["subject_id", "site_id"], as_index=False)["value"].median()
    rows = []
    for k, (site, g) in enumerate(d.groupby("site_id")):
        r = benford_test(g["value"].to_numpy(), sims, seed + k)
        rows.append({"site_id": site, "n": r["n"], "chi2": r["chi2"], "p_benford": r["p_value"],
                     "mad": r["mad"], "mad_class": r["mad_class"],
                     **{f"d{i + 1}": r["props"][i] for i in range(9)}})
    return pd.DataFrame(rows)
