"""Ensemble: percentile-rank normalisation + weighted average."""
from __future__ import annotations

import pandas as pd

DETECTORS = ["iforest", "mahal", "lof"]
PRIOR_WEIGHTS = {"iforest": 0.40, "mahal": 0.35, "lof": 0.25}
ENSEMBLE_COLS = ["ens_equal", "ens_prior", "ens_consensus", "ens_max"]


def to_ranks(raw: pd.DataFrame) -> pd.DataFrame:
    return raw[DETECTORS].rank(method="average", pct=True)


def normalize(w: dict[str, float]) -> dict[str, float]:
    s = float(sum(w.values()))
    if s <= 0:
        raise ValueError("weights must have a positive sum")
    return {k: float(v) / s for k, v in w.items()}


def consensus_weights(ranks: pd.DataFrame, floor: float = 0.05) -> dict[str, float]:
    corr = ranks[DETECTORS].corr()
    agree = (corr.sum() - 1.0) / (len(DETECTORS) - 1)
    return normalize(agree.clip(lower=floor).to_dict())


def combine(ranks: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    w = normalize(weights)
    return sum(ranks[k] * v for k, v in w.items())


def resolve_weights(spec, ranks: pd.DataFrame, prior: dict | None = None) -> dict[str, float]:
    if isinstance(spec, dict):
        return normalize(spec)
    if spec == "prior":
        return normalize(prior or PRIOR_WEIGHTS)
    if spec == "equal":
        return {k: 1.0 / len(DETECTORS) for k in DETECTORS}
    if spec == "consensus":
        return consensus_weights(ranks)
    raise ValueError(f"unknown weights spec: {spec!r}")


def score_variants(ranks: pd.DataFrame, spec="prior", prior: dict | None = None):
    out = ranks.copy()
    out["ens_equal"] = combine(ranks, resolve_weights("equal", ranks))
    out["ens_prior"] = combine(ranks, resolve_weights("prior", ranks, prior))
    out["ens_consensus"] = combine(ranks, consensus_weights(ranks))
    out["ens_max"] = ranks.max(axis=1)
    used = resolve_weights(spec, ranks, prior)
    out["ensemble"] = combine(ranks, used)
    return out, used


def vote_counts(ranks: pd.DataFrame, contamination: float) -> pd.Series:
    return (ranks[DETECTORS] >= 1.0 - contamination).sum(axis=1).astype(int)
