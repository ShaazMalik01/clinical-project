"""Evaluate ML detectors against injected ground truth."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..ml.features import KEYS
from ..ml.runner import MLResult, visit_scores_for_weights
from .rule_eval import _group_label

METHODS = ["iforest", "lof", "mahal", "ens_equal", "ens_prior", "ens_consensus", "ens_max"]
VISIT_SUBTYPES = ["value_outlier", "copy_forward"]
SUBJECT_TARGETS = ["low_variance", "digit_preference", "duplicate_subject", "copy_forward",
                   "ineligible_enrolled", "procedure_before_consent"]


def _safe_auc(y, s) -> tuple[float, float]:
    y = np.asarray(y, dtype=bool)
    if y.sum() == 0 or y.all():
        return float("nan"), float("nan")
    return roc_auc_score(y, s), average_precision_score(y, s)


def _visit_groups(gt: pd.DataFrame) -> pd.DataFrame:
    g = gt[gt["subtype"].isin(VISIT_SUBTYPES)].copy()
    g["group"] = _group_label(g)
    g["visit_num"] = g["visit_num"].astype(int)
    return g[KEYS + ["group"]].drop_duplicates(KEYS)


def _outlier_y(scores: pd.DataFrame, gt: pd.DataFrame) -> np.ndarray:
    s = scores[KEYS].merge(_visit_groups(gt), on=KEYS, how="left")
    return s["group"].fillna("normal").str.startswith("value_outlier").to_numpy()


def outlier_ap(scores: pd.DataFrame, gt: pd.DataFrame, col: str = "ensemble") -> float:
    return _safe_auc(_outlier_y(scores, gt), scores[col])[1]


def evaluate_visit_level(visit_scores: pd.DataFrame, gt: pd.DataFrame, contamination: float):
    s = visit_scores.merge(_visit_groups(gt), on=KEYS, how="left")
    s["group"] = s["group"].fillna("normal")
    y = s["group"].str.startswith("value_outlier").to_numpy()
    anomalous = (s["group"] != "normal").to_numpy()
    rows, rec = [], {}
    for m in METHODS:
        sc = s[m]
        flag = (sc >= sc.quantile(1 - contamination)).to_numpy()
        auc, ap = _safe_auc(y, sc)
        tp = int((flag & y).sum())
        rows.append({"method": m, "AUC": auc, "AP": ap,
                     "precision@thr": tp / max(int(flag.sum()), 1),
                     "recall@thr": tp / max(int(y.sum()), 1)})
        rec[m] = pd.Series(flag[anomalous], index=s.loc[anomalous, "group"].to_numpy()
                           ).groupby(level=0).mean()
    table = pd.DataFrame(rows).set_index("method").round(3)
    table["n_positive"] = int(y.sum())
    return table, pd.DataFrame(rec).round(3)


def _subject_labels(gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in gt[gt["subtype"].isin(SUBJECT_TARGETS)].itertuples():
        rows.append((r.subject_id, r.subtype))
        if r.subtype == "duplicate_subject" and isinstance(r.related_subject_id, str):
            rows.append((r.related_subject_id, "duplicate_subject"))
    return pd.DataFrame(rows, columns=["subject_id", "subtype"]).drop_duplicates()


def evaluate_subject_level(subject_scores: pd.DataFrame, gt: pd.DataFrame, contamination: float):
    lab = _subject_labels(gt)
    y = subject_scores["subject_id"].isin(set(lab["subject_id"])).to_numpy()
    rows, rec = [], {}
    for m in METHODS:
        sc = subject_scores[m]
        flag = sc >= sc.quantile(1 - contamination)
        auc, ap = _safe_auc(y, sc)
        tp = int((flag.to_numpy() & y).sum())
        rows.append({"method": m, "AUC": auc, "AP": ap,
                     "precision@thr": tp / max(int(flag.sum()), 1),
                     "recall@thr": tp / max(int(y.sum()), 1)})
        flagged = set(subject_scores.loc[flag, "subject_id"])
        rec[m] = lab.assign(hit=lab["subject_id"].isin(flagged)).groupby("subtype")["hit"].mean()
    table = pd.DataFrame(rows).set_index("method").round(3)
    table["n_positive"] = int(y.sum())
    return table, pd.DataFrame(rec).round(3)


def evaluate_sites(site_scores: pd.DataFrame, site_gt: pd.DataFrame):
    m = site_scores.merge(site_gt[["site_id", "profile", "is_bad_site"]], on="site_id")
    m = m.sort_values("site_score", ascending=False).reset_index(drop=True)
    m["rank"] = np.arange(1, len(m) + 1)
    auc = _safe_auc(m["is_bad_site"].astype(bool), m["site_score"])[0]
    return m, auc


def weight_sensitivity(res: MLResult, gt: pd.DataFrame, step: float = 0.1) -> pd.DataFrame:
    n = int(round(1 / step))
    rows = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            w = {"iforest": i * step, "mahal": j * step, "lof": (n - i - j) * step}
            sc = visit_scores_for_weights(res, w)
            rows.append({"w_iforest": round(w["iforest"], 2), "w_mahal": round(w["mahal"], 2),
                         "w_lof": round(w["lof"], 2), "AP": outlier_ap(sc, gt)})
    return pd.DataFrame(rows).sort_values("AP", ascending=False).reset_index(drop=True)
