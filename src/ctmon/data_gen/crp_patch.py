"""Additive patch: adds a Benford-suitable lab (CRP, mg/L) to an existing dataset.

Genuine CRP is log-normal with a large between-subject spread (spans >2 orders of magnitude, follows
Benford's law). The 'fabrication' site invents its CRP values with ~uniform first digits. Existing
columns/rows are untouched; idempotent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _genuine_crp(rng, labs: pd.DataFrame, mu: float, s_subj: float, s_vis: float) -> np.ndarray:
    ids = np.sort(labs["subject_id"].unique())
    base = pd.Series(rng.normal(mu, s_subj, len(ids)), index=ids)
    x = np.exp(labs["subject_id"].map(base).to_numpy() + rng.normal(0, s_vis, len(labs)))
    return np.round(np.clip(x, 0.05, 500.0), 2)


def _fabricate(rng, n: int) -> np.ndarray:
    d = rng.integers(1, 10, size=n)
    e = rng.choice([-1, 0, 1], size=n, p=[0.10, 0.60, 0.30])
    return np.round((d + rng.random(n)) * 10.0 ** e, 2)


def add_crp(clean: dict, raw: dict, gt: pd.DataFrame, site_gt: pd.DataFrame, seed: int,
            mu: float = float(np.log(2.5)), sigma_subject: float = 1.3,
            sigma_visit: float = 0.3, fab_fraction: float = 1.0):
    rng = np.random.default_rng(seed)
    clean = {k: v.copy() for k, v in clean.items()}
    raw = {k: v.copy() for k, v in raw.items()}
    gt = gt[gt["subtype"] != "fabricated_labs"].copy()
    for t in (clean, raw):
        if "crp" in t["labs"].columns:
            t["labs"] = t["labs"].drop(columns="crp")

    clean["labs"]["crp"] = _genuine_crp(rng, clean["labs"], mu, sigma_subject, sigma_visit)
    key = clean["labs"].set_index(["subject_id", "visit_num"])["crp"]

    rl = raw["labs"].copy()
    rl["crp"] = key.reindex(pd.MultiIndex.from_frame(rl[["subject_id", "visit_num"]])).to_numpy()

    for r in gt[gt["subtype"] == "duplicate_subject"].itertuples():
        m = (rl["subject_id"] == r.subject_id).to_numpy()
        if not m.any():
            continue
        mi = pd.MultiIndex.from_arrays([[r.related_subject_id] * int(m.sum()),
                                        rl.loc[m, "visit_num"].to_numpy()])
        src = key.reindex(mi).to_numpy()
        rl.loc[m, "crp"] = np.round(src * (1 + rng.normal(0, 0.01, int(m.sum()))), 2)

    fab = site_gt.loc[site_gt["profile"] == "fabrication", "site_id"]
    if len(fab):
        fs = fab.iloc[0]
        subs = np.sort(rl.loc[rl["site_id"] == fs, "subject_id"].unique())
        chosen = rng.choice(subs, size=max(1, int(round(fab_fraction * len(subs)))), replace=False)
        m = rl["subject_id"].isin(chosen)
        rl.loc[m, "crp"] = _fabricate(rng, int(m.sum()))

        next_id = int(gt["anomaly_id"].str[1:].astype(int).max()) + 1
        rows = []
        for sid in chosen:
            row = {c: None for c in gt.columns}
            row.update(anomaly_id=f"A{next_id:04d}", category="FAB", subtype="fabricated_labs",
                       severity="major", table="labs", subject_id=sid, site_id=fs, field="crp",
                       injected_value="invented values (uniform first digits)",
                       granularity="subject", in_bad_site=True,
                       description="CRP values invented: first digits ~uniform instead of Benford")
            rows.append(row)
            next_id += 1
        gt = pd.concat([gt, pd.DataFrame(rows, columns=gt.columns)], ignore_index=True)

    raw["labs"] = rl
    site_gt = site_gt.copy()
    site_gt["n_injected_anomalies"] = (site_gt["site_id"].map(gt.groupby("site_id").size())
                                       .fillna(0).astype(int))
    return clean, raw, gt, site_gt
