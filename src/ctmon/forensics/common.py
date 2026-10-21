"""Shared statistics helpers for the forensics package."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def neglog10(p) -> np.ndarray:
    return -np.log10(np.clip(np.asarray(p, dtype=float), 1e-300, 1.0))


def evidence(s, s0: float = 1.3, tau: float = 2.0) -> np.ndarray:
    return 1.0 - np.exp(-np.maximum(np.asarray(s, dtype=float) - s0, 0.0) / tau)


def noisy_or(E: pd.DataFrame) -> pd.Series:
    return 1.0 - (1.0 - E).prod(axis=1)


def bh_qvalues(p) -> np.ndarray:
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def robust_z(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / max(mad, 1e-9)


def plausible_vitals(vitals: pd.DataFrame, protocol: dict, fields) -> pd.DataFrame:
    v = vitals.copy()
    for f in fields:
        lo, hi = protocol["plausible_ranges"][f]
        v[f] = v[f].astype(float)
        v.loc[(v[f] < lo) | (v[f] > hi), f] = np.nan
    return v


def site_vs_rest(k: pd.Series, n: pd.Series) -> pd.Series:
    K, N = int(k.sum()), int(n.sum())
    out = {}
    for s in k.index:
        ks, ns = int(k[s]), int(n[s])
        if ns == 0:
            out[s] = 1.0
            continue
        table = [[ks, ns - ks], [K - ks, (N - ns) - (K - ks)]]
        out[s] = float(stats.fisher_exact(table, alternative="greater")[1])
    return pd.Series(out)
