"""Evaluate forensic tests against the injected ground truth."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TARGETS = ["copy_forward", "low_variance", "digit_preference", "duplicate_subject"]


def rank_metrics(y, score) -> tuple[float, float]:
    y = np.asarray(y, dtype=bool)
    if y.sum() == 0 or y.all():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def subject_labels(gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in gt[gt["subtype"].isin(TARGETS)].itertuples():
        rows.append((r.subject_id, r.subtype))
        if r.subtype == "duplicate_subject" and isinstance(r.related_subject_id, str):
            rows.append((r.related_subject_id, "duplicate_subject"))
    return pd.DataFrame(rows, columns=["subject_id", "subtype"]).drop_duplicates()


def evaluate_subjects(sub: pd.DataFrame, gt: pd.DataFrame, score_col: str = "suspicion"):
    lab = subject_labels(gt)
    pos = set(lab["subject_id"])
    y = sub["subject_id"].isin(pos).to_numpy()
    auc, ap = rank_metrics(y, sub[score_col])
    rows = []
    for t in TARGETS:
        ids = set(lab.loc[lab["subtype"] == t, "subject_id"])
        s = sub[sub["subject_id"].isin(ids)]
        rows.append({"pattern": t, "n_injected": len(ids),
                     "recall_HIGH": float((s["tier"] == "HIGH").mean()) if len(s) else np.nan,
                     "recall_MEDIUM+": float(s["tier"].isin(["HIGH", "MEDIUM"]).mean()) if len(s) else np.nan})
    high = sub[sub["tier"] == "HIGH"]
    summary = {"AUC": auc, "AP": ap, "n_positive": int(y.sum()), "n_HIGH": int(len(high)),
               "precision_HIGH": float(high["subject_id"].isin(pos).mean()) if len(high) else np.nan}
    return summary, pd.DataFrame(rows).round(3)


def evaluate_copy_records(copy_rec: pd.DataFrame, gt: pd.DataFrame) -> dict:
    g = gt[gt["subtype"] == "copy_forward"]
    gt_keys = {(r.subject_id, int(r.visit_num)) for r in g.itertuples()}
    src_keys = {(sid, int(grp["visit_num"].astype(int).min()) - 1) for sid, grp in g.groupby("subject_id")}
    det = {(r.subject_id, int(r.visit_num)) for r in copy_rec.itertuples()}
    return {"n_injected_records": len(gt_keys), "n_flagged_records": len(det),
            "recall": len(gt_keys & det) / len(gt_keys) if gt_keys else np.nan,
            "precision": len(det & (gt_keys | src_keys)) / len(det) if det else np.nan}


def evaluate_dup_pairs(pairs: pd.DataFrame, gt: pd.DataFrame) -> dict:
    g = gt[gt["subtype"] == "duplicate_subject"]
    truth = {frozenset((r.subject_id, r.related_subject_id)) for r in g.itertuples()}
    flagged = pairs[pairs["flagged"].astype(bool)] if len(pairs) else pairs
    det = {frozenset((a, b)) for a, b in zip(flagged["subject_a"], flagged["subject_b"])}
    return {"n_true_pairs": len(truth), "n_flagged_pairs": len(det),
            "recall": len(truth & det) / len(truth) if truth else np.nan,
            "precision": len(truth & det) / len(det) if det else np.nan}


def evaluate_sites(site_scores: pd.DataFrame, site_gt: pd.DataFrame):
    m = site_scores.merge(site_gt[["site_id", "profile"]], on="site_id")
    auc, _ = rank_metrics((m["profile"] == "fabrication").to_numpy(), m["suspicion"])
    m = m.sort_values("suspicion", ascending=False).reset_index(drop=True)
    m["rank"] = np.arange(1, len(m) + 1)
    fab_rank = int(m.index[m["profile"] == "fabrication"][0]) + 1
    return m, {"AUC_fabrication_site": auc, "rank_of_fabrication_site": fab_rank}
