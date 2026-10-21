"""Copy-forward: identical vital-sign vectors at different visits of the same subject."""
from __future__ import annotations

import pandas as pd


def detect_copy_forward(vitals: pd.DataFrame, fields):
    fields = list(fields)
    v = vitals.dropna(subset=fields).copy()
    key = v[fields[0]].round(2).astype(str)
    for f in fields[1:]:
        key = key + "|" + v[f].round(2).astype(str)
    v["_key"] = key
    grp = v.groupby(["subject_id", "_key"])["visit_num"]
    v["group_size"] = grp.transform("size")
    v["source_visit"] = grp.transform("min")

    rec = v[v["group_size"] >= 2][["subject_id", "site_id", "visit_num", "group_size",
                                   "source_visit"] + fields].copy()
    rec["is_copy"] = rec["visit_num"] > rec["source_visit"]
    rec = rec.sort_values(["subject_id", "visit_num"]).reset_index(drop=True)

    if rec.empty:
        subj = pd.DataFrame(columns=["subject_id", "n_copied", "max_group"])
    else:
        subj = (rec.groupby("subject_id")
                   .agg(n_copied=("is_copy", "sum"), max_group=("group_size", "max"))
                   .reset_index())
    return rec, subj
