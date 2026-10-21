"""Near-duplicate subjects (same patient enrolled twice, possibly at different sites)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import plausible_vitals

VIT_FIELDS = ["sbp", "dbp", "hr"]
PAIR_COLS = ["subject_a", "subject_b", "site_a", "site_b", "cross_site", "n_shared_visits",
             "vit_mad", "lab_rel", "v_sim", "l_sim", "pair_score", "flagged"]


def _dense(df: pd.DataFrame, index: dict, fields: list, n_visits: int) -> np.ndarray:
    arr = np.full((len(index), n_visits, len(fields)), np.nan)
    si = df["subject_id"].map(index)
    ok = si.notna().to_numpy()
    s = si[ok].astype(int).to_numpy()
    v = df.loc[ok, "visit_num"].astype(int).to_numpy() - 1
    for k, f in enumerate(fields):
        arr[s, v, k] = df.loc[ok, f].to_numpy(float)
    return arr


def _pair_metrics(V, L, i, j):
    dv = np.abs(V[i] - V[j])
    fin = np.isfinite(dv)
    n_shared = int(fin.any(axis=1).sum())
    vit_mad = float(dv[fin].mean()) if fin.any() else np.nan
    lab_rel = np.nan
    if L is not None:
        a, b = L[i], L[j]
        rel = np.abs(a - b) / ((np.abs(a) + np.abs(b)) / 2.0 + 1e-9)
        ok = np.isfinite(rel)
        if ok.sum() >= 2:
            lab_rel = float(np.median(rel[ok]))
    return n_shared, vit_mad, lab_rel


def _candidates(subs: pd.DataFrame, cfg: dict):
    sex, age = subs["sex"].to_numpy(), subs["age"].to_numpy(float)
    h, b = subs["height_cm"].to_numpy(float), subs["bmi"].to_numpy(float)
    ii, jj = [], []
    for s in np.unique(sex):
        idx = np.where(sex == s)[0]
        a, hh, bb = age[idx], h[idx], b[idx]
        m = ((np.abs(a[:, None] - a[None, :]) <= cfg["age_tol"])
             & (np.abs(hh[:, None] - hh[None, :]) <= cfg["height_tol_cm"])
             & (np.abs(bb[:, None] - bb[None, :]) <= cfg["bmi_tol"]))
        r, c = np.where(np.triu(m, 1))
        ii.append(idx[r])
        jj.append(idx[c])
    return np.concatenate(ii), np.concatenate(jj)


def find_near_duplicates(tables: dict, protocol, cfg: dict, seed: int = 42):
    subs = tables["subjects"].reset_index(drop=True)
    ids = subs["subject_id"].to_numpy()
    index = {s: i for i, s in enumerate(ids)}
    vit = plausible_vitals(tables["vitals"], protocol, VIT_FIELDS) if protocol else tables["vitals"]
    labs = tables["labs"]
    n_visits = int(max(vit["visit_num"].max(), labs["visit_num"].max()))
    V = _dense(vit, index, VIT_FIELDS, n_visits)
    lab_fields = [f for f in cfg["lab_fields"] if f in labs.columns]
    L = _dense(labs, index, lab_fields, n_visits) if lab_fields else None

    rng = np.random.default_rng(seed)
    refs = [_pair_metrics(V, L, i, j) for i, j in rng.integers(0, len(ids), size=(cfg["ref_pairs"], 2))
            if i != j]
    rv = [r[1] for r in refs if r[0] >= cfg["min_shared_visits"] and np.isfinite(r[1])]
    rl = [r[2] for r in refs if np.isfinite(r[2])]
    ref_v = max(float(np.median(rv)) if rv else 1.0, 1e-6)
    ref_l = max(float(np.median(rl)) if rl else 1.0, 1e-6)

    ci, cj = _candidates(subs, cfg)
    rows = []
    for i, j in zip(ci, cj):
        n_sh, vm, lr = _pair_metrics(V, L, i, j)
        v_sim = float(np.clip(1 - vm / ref_v, 0, 1)) if np.isfinite(vm) else np.nan
        l_sim = float(np.clip(1 - lr / ref_l, 0, 1)) if np.isfinite(lr) else np.nan
        score = v_sim if np.isnan(l_sim) else min(v_sim, l_sim)
        rows.append({"subject_a": ids[i], "subject_b": ids[j],
                     "site_a": subs.at[i, "site_id"], "site_b": subs.at[j, "site_id"],
                     "cross_site": subs.at[i, "site_id"] != subs.at[j, "site_id"],
                     "n_shared_visits": n_sh, "vit_mad": vm, "lab_rel": lr,
                     "v_sim": v_sim, "l_sim": l_sim, "pair_score": score,
                     "flagged": bool(n_sh >= cfg["min_shared_visits"] and np.isfinite(score)
                                     and score >= cfg["similarity_threshold"])})
    pairs = pd.DataFrame(rows, columns=PAIR_COLS)
    if len(pairs):
        pairs = pairs.sort_values("pair_score", ascending=False).reset_index(drop=True)

    flagged = pairs[pairs["flagged"]] if len(pairs) else pairs
    sub_rows = []
    for r in flagged.itertuples():
        sub_rows += [(r.subject_a, r.subject_b, r.pair_score), (r.subject_b, r.subject_a, r.pair_score)]
    subj = pd.DataFrame(sub_rows, columns=["subject_id", "dup_partner", "dup_score"])
    subj = subj.sort_values("dup_score", ascending=False).drop_duplicates("subject_id")
    return pairs, subj
