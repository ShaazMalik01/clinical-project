"""Evaluate site risk vs ground truth, plus a simulation study of small-site fairness."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..risk.bayesian_site_risk import fit_prior, posterior_draws


def _auc(y, s) -> float:
    y = np.asarray(y, dtype=bool)
    if y.sum() == 0 or y.all():
        return float("nan")
    return float(roc_auc_score(y, s))


def evaluate_site_risk(site_risk: pd.DataFrame, site_gt: pd.DataFrame):
    m = site_risk.merge(site_gt[["site_id", "profile", "is_bad_site"]], on="site_id", how="left")
    m["is_bad_site"] = m["is_bad_site"].fillna(False).astype(bool)
    m["rank_by_index"] = m["risk_index"].rank(ascending=False, method="min").astype(int)
    m["rank_by_naive"] = m["naive_index"].rank(ascending=False, method="min").astype(int)

    def pick(profile, col):
        sub = m[m["profile"] == profile]
        return sub[col].iloc[0] if len(sub) else None

    clean = m[~m["is_bad_site"]]
    summary = {
        "AUC_bayes_index": _auc(m["is_bad_site"], m["risk_index"]),
        "AUC_naive_index": _auc(m["is_bad_site"], m["naive_index"]),
        "rank_sloppy_bayes": pick("sloppy_process", "rank_by_index"),
        "rank_sloppy_naive": pick("sloppy_process", "rank_by_naive"),
        "rank_fabrication_bayes": pick("fabrication", "rank_by_index"),
        "rank_fabrication_naive": pick("fabrication", "rank_by_naive"),
        "tier_sloppy": pick("sloppy_process", "tier"),
        "tier_fabrication": pick("fabrication", "tier"),
        "clean_sites": int(len(clean)),
        "clean_above_routine": int((clean["tier"] != "ROUTINE").sum()),
        "clean_targeted_or_audit": int(clean["tier"].isin(["TARGETED_ONSITE", "FULL_AUDIT"]).sum()),
    }
    return m, summary


def weight_sensitivity(res, site_gt: pd.DataFrame, n_draws: int = 2000,
                       concentration: float = 30.0, seed: int = 0) -> dict:
    E = res.excess_mean
    w0 = np.array([res.weights[k] for k in E.columns])
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.maximum(concentration * w0, 1e-3), size=n_draws)
    eri = W @ E.to_numpy().T
    ranks = (-eri).argsort(axis=1).argsort(axis=1) + 1
    bad = site_gt.set_index("site_id").reindex(E.index)["is_bad_site"].fillna(False).to_numpy(bool)
    nb = int(bad.sum())
    top = ranks <= max(nb, 1)
    return {"n_bad_sites": nb,
            "p_all_bad_in_top": float(top[:, bad].all(axis=1).mean()) if nb else float("nan"),
            "p_site_in_top": pd.Series(top[:, bad].mean(axis=0), index=E.index[bad]),
            "median_rank": pd.Series(np.median(ranks[:, bad], axis=0), index=E.index[bad])}


def simulate_small_site_study(n_reps: int = 150, seed: int = 0, sizes=None, mu0: float = 0.08,
                              kappa0: float = 60.0, bad_theta: float = 0.30, n_bad: int = 2,
                              small_max: int = 15, prior_cfg: dict | None = None,
                              n_draws: int = 1500) -> dict:
    sizes = (np.array([8, 10, 12, 40, 50, 60, 70, 80, 90, 250, 300, 350], dtype=int)
             if sizes is None else np.asarray(sizes, dtype=int))
    S = len(sizes)
    small = sizes <= small_max
    pc = {"outlier_prob": 0.15, "kappa_min": 4.0, "kappa_cap_factor": 1.0}
    pc.update(prior_cfg or {})
    rng = np.random.default_rng(seed)

    auc_n, auc_b, fa_n, fa_b = [], [], [], []
    se_n_small, se_b_small, se_n_clean, se_b_clean = [], [], [], []
    cov_clean, cov_clean_small, cov_bad = [], [], []
    for _ in range(n_reps):
        theta = rng.beta(mu0 * kappa0, (1 - mu0) * kappa0, S)
        bad = np.zeros(S, dtype=bool)
        bad[rng.choice(S, n_bad, replace=False)] = True
        theta[bad] = bad_theta
        x = rng.binomial(sizes, theta)

        naive = x / sizes
        fit = fit_prior(x, sizes, pc["outlier_prob"], pc["kappa_min"], pc["kappa_cap_factor"])
        th, _ = posterior_draws(x, sizes, fit, pc["outlier_prob"], n_draws, rng)
        pm = th.mean(axis=0)
        lo, hi = np.percentile(th, [2.5, 97.5], axis=0)

        jit = 1e-9 * rng.random(S)
        auc_n.append(_auc(bad, naive))
        auc_b.append(_auc(bad, pm))
        for score, store in ((naive + jit, fa_n), (pm + jit, fa_b)):
            top2 = np.argsort(-score)[:2]
            store.append(bool((small[top2] & ~bad[top2]).any()))

        clean = ~bad
        se_n_clean.extend(((naive - theta) ** 2)[clean])
        se_b_clean.extend(((pm - theta) ** 2)[clean])
        se_n_small.extend(((naive - theta) ** 2)[clean & small])
        se_b_small.extend(((pm - theta) ** 2)[clean & small])
        covered = (lo <= theta) & (theta <= hi)
        cov_clean.extend(covered[clean])
        cov_clean_small.extend(covered[clean & small])
        cov_bad.extend(covered[bad])

    return {"n_reps": n_reps,
            "AUC_naive": float(np.nanmean(auc_n)), "AUC_bayes": float(np.nanmean(auc_b)),
            "false_alarm_naive": float(np.mean(fa_n)), "false_alarm_bayes": float(np.mean(fa_b)),
            "RMSE_naive_small": float(np.sqrt(np.mean(se_n_small))),
            "RMSE_bayes_small": float(np.sqrt(np.mean(se_b_small))),
            "RMSE_naive_clean": float(np.sqrt(np.mean(se_n_clean))),
            "RMSE_bayes_clean": float(np.sqrt(np.mean(se_b_clean))),
            "coverage_clean": float(np.mean(cov_clean)),
            "coverage_clean_small": float(np.mean(cov_clean_small)),
            "coverage_bad": float(np.mean(cov_bad))}
