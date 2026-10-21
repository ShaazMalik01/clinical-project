"""Evidence -> suspicion score -> tier -> plain-language interpretation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import bh_qvalues, evidence, neglog10, noisy_or, site_vs_rest

SITE_TESTS = ["digit", "lowvar", "copy", "dup", "benford"]


def _subject_reason(r, p0: float, cfg: dict) -> str:
    out = []
    if r.n_copied > 0:
        out.append(f"{int(r.n_copied)} visit(s) have vitals identical to an earlier visit (copy-forward)")
    if pd.notna(r.dup_partner):
        out.append(f"near-duplicate of {r.dup_partner} (similarity {r.dup_score:.2f})")
    if r.q_digit < cfg["medium_q"]:
        out.append(f"{r.pct_round5:.0%} of {int(r.n_readings)} BP/HR readings end in 0 or 5 "
                   f"(peers ~{p0:.0%}): terminal-digit preference")
    if r.q_lowvar < cfg["medium_q"]:
        out.append(f"visit-to-visit variability is only {r.var_ratio:.0%} of a typical subject's "
                   f"(z = {r.z_lowvar:.1f})")
    return "; ".join(out)


def score_subjects(subjects, digit, var, copy_subj, dup_subj, p0: float, cfg: dict) -> pd.DataFrame:
    df = subjects[["subject_id", "site_id"]].copy()
    df = df.merge(digit[["subject_id", "n_readings", "pct_round5", "p_digit"]], on="subject_id", how="left")
    df = df.merge(var[["subject_id", "var_ratio", "z_lowvar", "p_lowvar"]], on="subject_id", how="left")
    df = df.merge(copy_subj[["subject_id", "n_copied"]], on="subject_id", how="left")
    df = df.merge(dup_subj[["subject_id", "dup_partner", "dup_score"]], on="subject_id", how="left")
    df["p_digit"] = df["p_digit"].fillna(1.0)
    df["p_lowvar"] = df["p_lowvar"].fillna(1.0)
    df["z_lowvar"] = df["z_lowvar"].fillna(0.0)
    df["var_ratio"] = df["var_ratio"].fillna(1.0)
    df["pct_round5"] = df["pct_round5"].fillna(0.0)
    df["n_readings"] = df["n_readings"].fillna(0).astype(int)
    df["n_copied"] = df["n_copied"].fillna(0).astype(int)

    df["q_digit"] = bh_qvalues(df["p_digit"].to_numpy())
    df["q_lowvar"] = bh_qvalues(df["p_lowvar"].to_numpy())

    det = float(cfg["deterministic_evidence"])
    s = pd.DataFrame({
        "digit_preference": neglog10(df["p_digit"]),
        "low_variance": neglog10(df["p_lowvar"]),
        "copy_forward": np.where(df["n_copied"] > 0, det, 0.0),
        "duplicate": np.where(df["dup_partner"].notna(), det, 0.0)}, index=df.index)
    E = pd.DataFrame(evidence(s.to_numpy(), cfg["s0"], cfg["tau"]), index=df.index, columns=s.columns)
    df["suspicion"] = noisy_or(E).round(4)
    df["primary_pattern"] = np.where(E.max(axis=1) > 0, E.idxmax(axis=1), "none")

    qmin = np.minimum(df["q_digit"], df["q_lowvar"])
    deterministic = (df["n_copied"] > 0) | df["dup_partner"].notna()
    df["tier"] = np.where(deterministic | (qmin < cfg["high_q"]), "HIGH",
                          np.where(qmin < cfg["medium_q"], "MEDIUM", "LOW"))
    df["reasons"] = df.apply(lambda r: _subject_reason(r, p0, cfg), axis=1)
    return df


def score_sites(subjects, subj: pd.DataFrame, digit_site: pd.DataFrame, benford_p, cfg: dict) -> pd.DataFrame:
    n = subjects.groupby("site_id").size().rename("n_subjects")
    idx = n.index

    def per_site(mask) -> pd.Series:
        return mask.groupby(subj["site_id"]).sum().reindex(idx).fillna(0).astype(int)

    k_low = per_site(subj["p_lowvar"] < cfg["subject_nominal_p"])
    k_copy = per_site(subj["n_copied"] > 0)
    k_dup = per_site(subj["dup_partner"].notna())
    ds = digit_site.set_index("site_id").reindex(idx).fillna(0)

    p = pd.DataFrame(index=idx)
    p["digit"] = site_vs_rest(ds["k"], ds["n"]).reindex(idx)
    p["lowvar"] = site_vs_rest(k_low, n).reindex(idx)
    p["copy"] = site_vs_rest(k_copy, n).reindex(idx)
    p["dup"] = site_vs_rest(k_dup, n).reindex(idx)
    p["benford"] = benford_p.reindex(idx) if benford_p is not None else np.nan

    P = p.fillna(1.0)
    S = pd.DataFrame(neglog10(P.to_numpy()), index=idx, columns=P.columns)
    E = pd.DataFrame(evidence(S.to_numpy(), cfg["s0"], cfg["tau"]), index=idx, columns=P.columns)
    Q = pd.DataFrame({c: bh_qvalues(P[c].to_numpy()) for c in P.columns}, index=idx)
    qmin = Q.min(axis=1)

    K, N = float(ds["k"].sum()), float(ds["n"].sum())
    rate = (ds["k"] / ds["n"].replace(0, np.nan)).fillna(0)
    rate_rest = ((K - ds["k"]) / (N - ds["n"]).replace(0, np.nan)).fillna(0)

    reasons = []
    for s in idx:
        r = []
        if P.at[s, "digit"] < 0.05:
            r.append(f"terminal-digit: {rate[s]:.1%} of BP/HR readings end in 0/5 vs "
                     f"{rate_rest[s]:.1%} elsewhere (p={P.at[s, 'digit']:.1e})")
        for t, k, label in (("lowvar", k_low, "abnormally low variability"),
                            ("copy", k_copy, "copy-forward vitals"),
                            ("dup", k_dup, "near-duplicate records")):
            if P.at[s, t] < 0.05:
                r.append(f"{int(k[s])}/{int(n[s])} subjects with {label} vs "
                         f"{int(k.sum() - k[s])}/{int(n.sum() - n[s])} elsewhere (p={P.at[s, t]:.1e})")
        if P.at[s, "benford"] < 0.05:
            r.append(f"first-digit distribution of lab values deviates from Benford (p={P.at[s, 'benford']:.1e})")
        reasons.append("; ".join(r))

    out = pd.DataFrame({
        "site_id": idx, "n_subjects": n.to_numpy(),
        "digit_rate": rate.to_numpy().round(4), "digit_rate_rest": rate_rest.to_numpy().round(4),
        "k_lowvar": k_low.to_numpy(), "k_copy": k_copy.to_numpy(), "k_dup": k_dup.to_numpy()})
    for t in SITE_TESTS:
        out[f"p_{t}"] = p[t].to_numpy()
    out["suspicion"] = noisy_or(E).round(4).to_numpy()
    out["top_test"] = np.where(E.max(axis=1) > 0, E.idxmax(axis=1), "none")
    out["tier"] = np.where(qmin < cfg["high_q"], "HIGH",
                           np.where(qmin < cfg["medium_q"], "MEDIUM", "LOW"))
    out["reasons"] = reasons
    out["rank"] = out["suspicion"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("rank").reset_index(drop=True)
