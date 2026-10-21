"""Evaluate rule-engine output against the injected ground truth."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _group_label(gt: pd.DataFrame) -> pd.Series:
    """Split value outliers into 'impossible' (rules should catch) vs 'plausible' (need ML)."""
    g = gt["subtype"].copy()
    is_out = gt["subtype"] == "value_outlier"
    impossible = gt["description"].str.contains("physically impossible", na=False)
    g[is_out & impossible] = "value_outlier [impossible]"
    g[is_out & ~impossible] = "value_outlier [plausible]"
    return g


def evaluate_rules(dev: pd.DataFrame, gt: pd.DataFrame, covered_subtypes: set[str]):
    d = dev[["deviation_id", "subject_id", "visit_num", "record_id", "detects"]].copy()
    d["subtype"] = d["detects"].fillna("").astype(str).str.split("|")
    d = d.explode("subtype")
    d = d[d["subtype"].astype(bool)]

    g = gt[["anomaly_id", "subtype", "subject_id", "visit_num", "record_id"]]
    m = d.merge(g, on=["subject_id", "subtype"], suffixes=("_d", "_g"))
    if len(m):
        vd = m["visit_num_d"].to_numpy(dtype="float64", na_value=np.nan)
        vg = m["visit_num_g"].to_numpy(dtype="float64", na_value=np.nan)
        rid_d = m["record_id_d"].fillna("").astype(str)
        rid_g = m["record_id_g"].fillna("").astype(str)
        ok = (np.isnan(vd) | np.isnan(vg) | (vd == vg)) & ((rid_g == "") | (rid_g == rid_d)).to_numpy()
        m = m[ok]

    hit_gt = set(m["anomaly_id"])
    tp_dev = set(m["deviation_id"])

    gt = gt.copy()
    gt["detected"] = gt["anomaly_id"].isin(hit_gt)
    gt["group"] = _group_label(gt)
    covered = {c for c in covered_subtypes}

    per = (gt.groupby(["category", "group"])
             .agg(injected=("anomaly_id", "count"), detected=("detected", "sum"))
             .reset_index())
    per["recall"] = (per["detected"] / per["injected"]).round(3)
    per["rule_covered"] = per["group"].map(
        lambda x: x.split(" ")[0] in covered)
    per = per.sort_values(["rule_covered", "category", "group"], ascending=[False, True, True])

    cov_mask = gt["subtype"].isin(covered)
    overall = {
        "n_deviations": int(len(dev)),
        "precision": round(len(tp_dev) / len(dev), 3) if len(dev) else float("nan"),
        "recall_rule_covered": round(float(gt.loc[cov_mask, "detected"].mean()), 3),
        "recall_all_injected": round(float(gt["detected"].mean()), 3),
        "n_injected": int(len(gt)),
        "false_positive_ids": sorted(set(dev["deviation_id"]) - tp_dev),
    }
    return per.reset_index(drop=True), overall
