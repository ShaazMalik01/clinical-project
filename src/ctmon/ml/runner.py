"""Orchestrates feature building -> detectors -> ensemble at visit, subject and site level."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .detectors import DetectorConfig, DetectorSuite, top_features
from .ensemble import (DETECTORS, ENSEMBLE_COLS, combine, score_variants, to_ranks,
                       vote_counts)
from .features import (KEYS, SUBJECT_MODEL_COLS, build_labs_view, build_site_features,
                       build_subject_features, build_vitals_view, score_sites)


@dataclass
class MLResult:
    visit_scores: pd.DataFrame
    subject_scores: pd.DataFrame
    site_scores: pd.DataFrame
    subject_features: pd.DataFrame
    weights: dict
    rank_views: dict


def run_ml(tables: dict[str, pd.DataFrame], ml_cfg: dict, seed: int = 42) -> MLResult:
    cfg = DetectorConfig(
        n_estimators=ml_cfg.get("iforest", {}).get("n_estimators", 300),
        lof_neighbors=ml_cfg.get("lof", {}).get("n_neighbors", 35),
        cov_support=ml_cfg.get("mahalanobis", {}).get("support_fraction", 0.75),
        seed=seed)
    cont = ml_cfg.get("contamination", {"visit": 0.02, "subject": 0.10})
    spec = ml_cfg.get("weights", "prior")
    prior = ml_cfg.get("prior_weights")
    weights, rank_views, parts = {}, {}, []

    for name, (keys, feats) in {"vitals": build_vitals_view(tables),
                                "labs": build_labs_view(tables)}.items():
        suite = DetectorSuite(cfg)
        ranks = to_ranks(suite.fit_score(feats))
        variants, used = score_variants(ranks, spec, prior)
        part = pd.concat([keys.reset_index(drop=True), variants.reset_index(drop=True)], axis=1)
        part["votes"] = vote_counts(ranks, cont["visit"]).to_numpy()
        part["mahal_p"] = suite.p_values_
        part["top_features"] = top_features(feats, 3).to_numpy()
        part["view"] = name
        parts.append(part)
        weights[name] = used
        rank_views[name] = (keys.reset_index(drop=True), ranks.reset_index(drop=True))

    allv = pd.concat(parts, ignore_index=True)
    score_cols = DETECTORS + ENSEMBLE_COLS + ["ensemble"]
    best = allv.sort_values("ensemble", ascending=False).drop_duplicates(KEYS)
    mx = allv.groupby(KEYS)[score_cols].max().reset_index()
    visit = best.drop(columns=score_cols).merge(mx, on=KEYS)
    visit = visit.sort_values("ensemble", ascending=False).reset_index(drop=True)
    visit["flag"] = visit["ensemble"] >= visit["ensemble"].quantile(1 - cont["visit"])

    sf = build_subject_features(tables)
    X = sf[SUBJECT_MODEL_COLS]
    suite = DetectorSuite(cfg)
    ranks = to_ranks(suite.fit_score(X))
    variants, used = score_variants(ranks, spec, prior)
    weights["subject"] = used
    ss = variants.copy()
    ss.insert(0, "site_id", sf["site_id"])
    ss["votes"] = vote_counts(ranks, cont["subject"])
    ss["mahal_p"] = suite.p_values_
    ss["top_features"] = top_features(X, 3)
    ss["flag"] = ss["ensemble"] >= ss["ensemble"].quantile(1 - cont["subject"])
    ss = ss.rename_axis("subject_id").reset_index()

    site_feats = build_site_features(sf, subject_flag=ss.set_index("subject_id")["flag"],
                                     visit_flags=visit[["site_id", "flag"]])
    site = score_sites(site_feats, ml_cfg.get("site", {}).get("z_clip", 5.0))

    return MLResult(visit, ss, site, sf, weights, rank_views)


def visit_scores_for_weights(res: MLResult, weights: dict[str, float]) -> pd.DataFrame:
    parts = []
    for keys, ranks in res.rank_views.values():
        p = keys[KEYS].copy()
        p["ensemble"] = combine(ranks, weights).to_numpy()
        parts.append(p)
    return pd.concat(parts).groupby(KEYS, as_index=False)["ensemble"].max()
