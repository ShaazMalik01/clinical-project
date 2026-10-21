"""Is the explanation FAITHFUL? Check cards against the field we actually corrupted."""
from __future__ import annotations

import pandas as pd

from ..explain.zscores import top_visit_features


def explanation_fidelity(frames: dict, vs: pd.DataFrame, gt: pd.DataFrame, k: int = 3):
    view_of = {(s, int(v)): w for s, v, w in zip(vs["subject_id"], vs["visit_num"], vs["view"])}
    rows = []
    for r in gt[gt["subtype"] == "value_outlier"].itertuples():
        key = (r.subject_id, int(r.visit_num))
        view = view_of.get(key)
        if view is None or key not in frames[view].index:
            continue
        top = top_visit_features(frames[view], frames[view].index[key], k)
        bases = [t["base"] for t in top]
        rows.append({"subject_id": r.subject_id, "visit_num": int(r.visit_num), "field": r.field,
                     "top_feature": bases[0], "hit_at_1": bases[0] == r.field,
                     "hit_at_k": r.field in bases})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n": 0, "hit_at_1": float("nan"), "hit_at_k": float("nan")}, df
    return {"n": len(df), "hit_at_1": float(df["hit_at_1"].mean()), "hit_at_k": float(df["hit_at_k"].mean())}, df
