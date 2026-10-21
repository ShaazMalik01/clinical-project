"""Bayesian site risk: robust empirical-Bayes Beta-Binomial with an outlier component.

Per KRI:  theta_s ~ (1-eps) Beta(mu*kappa, (1-mu)*kappa) + eps Beta(1,1)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit, logit

from .kri import KRI_INFO, KRI_ORDER

DEFAULTS = {
    "weights": {"deviation_rate": 0.15, "critical_deviation_rate": 0.20, "sae_late": 0.15,
                "forensic_rate": 0.20, "anomaly_rate": 0.10, "missing_data": 0.10,
                "query_rate": 0.10},
    "floors": {"deviation_rate": 0.05, "critical_deviation_rate": 0.02, "query_rate": 0.01,
               "sae_late": 0.10, "missing_data": 0.01, "anomaly_rate": 0.02,
               "forensic_rate": 0.04},
    "critical_class": ["critical_deviation_rate", "sae_late", "forensic_rate"],
    "kri": {"deviation": {"severities": ["major", "critical"], "exclude_rules": ["VIS-MISSED"]}},
    "prior": {"outlier_prob": 0.15, "kappa_min": 4.0, "kappa_cap_factor": 1.0},
    "posterior": {"mc_draws": 8000, "ci_level": 0.95, "tolerance": 1.5, "excess_cap": 5.0},
    "tiers": {"remote_review": {"index": 0.15, "prob": 0.50},
              "targeted_onsite": {"index": 0.30, "prob": 0.70},
              "full_audit": {"index": 0.60, "prob": 0.80}},
    "triggers": {"mult": 3.0, "prob": 0.95, "min_kris_breached": 2},
}

TIER_ORDER = ["ROUTINE", "REMOTE_REVIEW", "TARGETED_ONSITE", "FULL_AUDIT"]
ACTIONS = {
    "ROUTINE": "Continue standard centralized monitoring; no extra action.",
    "REMOTE_REVIEW": "Remote review: raise data queries, request source documents for flagged "
                     "records, re-assess at the next monitoring cycle.",
    "TARGETED_ONSITE": "Targeted on-site visit focused on the driver KRIs (SDV of flagged "
                       "subjects, consent/eligibility files, SAE reporting process).",
    "FULL_AUDIT": "For-cause audit: full on-site audit (source-data verification, consent "
                  "forms, SAE chain, data-integrity review); consider pausing enrolment "
                  "until the review is complete.",
}


def _merge(base: dict, over: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def fit_prior(x, n, outlier_prob: float = 0.15, kappa_min: float = 4.0,
              kappa_cap_factor: float = 1.0, mu_bounds=(1e-4, 0.9)) -> dict:
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    ok = n > 0
    if not ok.any():
        mu, kappa = 0.1, float(kappa_min)
        return {"mu": mu, "kappa": kappa, "alpha": mu * kappa, "beta": (1 - mu) * kappa,
                "kappa_max": kappa, "n_sites": 0}
    xv, nv = np.minimum(x[ok], n[ok]), n[ok]
    kappa_max = max(float(kappa_min), float(kappa_cap_factor) * float(np.median(nv)))
    eps = float(np.clip(outlier_prob, 1e-9, 0.9))
    B = betaln(xv + 1.0, nv - xv + 1.0)
    log_keep, log_out = np.log1p(-eps), np.log(eps)

    def nll(t):
        mu, kappa = expit(t[0]), np.exp(t[1])
        a, b = mu * kappa, (1.0 - mu) * kappa
        A = betaln(xv + a, nv - xv + b) - betaln(a, b)
        val = -np.logaddexp(log_keep + A, log_out + B).sum()
        return float(val) if np.isfinite(val) else 1e12

    lo = np.log(kappa_min)
    hi = max(np.log(kappa_max), lo + 1e-6)
    mu0 = float(np.clip(np.median(xv / nv), 2 * mu_bounds[0], 0.5))
    bounds = [(logit(mu_bounds[0]), logit(mu_bounds[1])), (lo, hi)]
    best = None
    starts = np.unique(np.clip([2 * kappa_min, np.sqrt(kappa_min * kappa_max), kappa_max],
                               kappa_min, kappa_max))
    for k0 in starts:
        r = minimize(nll, x0=[logit(mu0), np.log(k0)], method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    mu, kappa = float(expit(best.x[0])), float(np.exp(best.x[1]))
    return {"mu": mu, "kappa": kappa, "alpha": mu * kappa, "beta": (1 - mu) * kappa,
            "kappa_max": kappa_max, "n_sites": int(ok.sum())}


def posterior_draws(x, n, prior: dict, outlier_prob: float, n_draws: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = np.minimum(np.asarray(x, dtype=float), np.asarray(n, dtype=float))
    n = np.asarray(n, dtype=float)
    a, b = prior["alpha"], prior["beta"]
    eps = float(np.clip(outlier_prob, 1e-12, 1 - 1e-12))
    A = betaln(x + a, n - x + b) - betaln(a, b)
    B = betaln(x + 1.0, n - x + 1.0)
    r = expit(np.log(eps) - np.log1p(-eps) + B - A)
    r = np.where(n > 0, r, 0.0)
    S = len(x)
    th_norm = rng.beta(a + x, b + n - x, size=(n_draws, S))
    th_out = rng.beta(1.0 + x, 1.0 + n - x, size=(n_draws, S))
    use_out = rng.random((n_draws, S)) < r
    return np.where(use_out, th_out, th_norm), r


@dataclass
class RiskResult:
    site_risk: pd.DataFrame
    kri_table: pd.DataFrame
    priors: pd.DataFrame
    excess_mean: pd.DataFrame
    weights: dict
    cfg: dict


def score_sites(counts: pd.DataFrame, cfg: dict | None = None, seed: int = 42) -> RiskResult:
    c = _merge(DEFAULTS, cfg)
    pri, post, tiers, trig = c["prior"], c["posterior"], c["tiers"], c["triggers"]
    kris = [k for k in KRI_ORDER if k in set(counts["kri"])]
    sites = list(dict.fromkeys(counts["site_id"]))
    X = (counts.pivot(index="site_id", columns="kri", values="events")
         .reindex(index=sites, columns=kris).fillna(0).to_numpy(float))
    N = (counts.pivot(index="site_id", columns="kri", values="opportunities")
         .reindex(index=sites, columns=kris).fillna(0).to_numpy(float))
    S, K = X.shape

    w = np.array([c["weights"].get(k, 0.0) for k in kris], dtype=float)
    w = np.full(K, 1.0 / K) if w.sum() <= 0 else w / w.sum()

    rng = np.random.default_rng(seed)
    M = int(post["mc_draws"])
    theta = np.empty((M, S, K))
    resp = np.empty((S, K))
    ref = np.empty(K)
    kappa = np.empty(K)
    prior_rows = []
    for j, k in enumerate(kris):
        fit = fit_prior(X[:, j], N[:, j], pri["outlier_prob"], pri["kappa_min"],
                        pri["kappa_cap_factor"])
        th, r = posterior_draws(X[:, j], N[:, j], fit, pri["outlier_prob"], M, rng)
        theta[:, :, j], resp[:, j] = th, r
        floor = float(c["floors"].get(k, 0.0))
        ref[j], kappa[j] = max(fit["mu"], floor), fit["kappa"]
        prior_rows.append({"kri": k, "mu_peers": fit["mu"], "kappa": fit["kappa"],
                           "kappa_max": fit["kappa_max"], "floor": floor, "ref_rate": ref[j],
                           "tolerance_rate": post["tolerance"] * ref[j],
                           "problem_sites": int((r > 0.5).sum()), "weight": w[j]})

    tol, cap = post["tolerance"], post["excess_cap"]
    rho = theta / (tol * ref)[None, None, :]
    excess = np.clip(rho - 1.0, 0.0, cap - 1.0)
    eri = (excess * w[None, None, :]).sum(axis=2)
    contrib = (excess * w[None, None, :]).mean(axis=0)
    excess_mean = excess.mean(axis=0)

    q = [100 * (1 - post["ci_level"]) / 2, 100 * (1 + post["ci_level"]) / 2]
    mean_th = theta.mean(axis=0)
    th_ci = np.percentile(theta, q, axis=0)
    p_exceed = (rho > 1.0).mean(axis=0)
    p_breach = (theta > trig["mult"] * ref[None, None, :]).mean(axis=0)

    eri_mean = eri.mean(axis=0)
    eri_ci = np.percentile(eri, q, axis=0)
    p_remote = (eri >= tiers["remote_review"]["index"]).mean(axis=0)
    p_onsite = (eri >= tiers["targeted_onsite"]["index"]).mean(axis=0)
    p_audit = (eri >= tiers["full_audit"]["index"]).mean(axis=0)

    raw = np.divide(X, N, out=np.zeros_like(X), where=N > 0)
    naive = np.clip(raw / (tol * ref)[None, :] - 1.0, 0.0, cap - 1.0) @ w

    crit = set(c["critical_class"])
    dev_idx = kris.index("deviation_rate") if "deviation_rate" in kris else None
    rows = []
    for i, s in enumerate(sites):
        breached = [kris[j] for j in range(K) if p_breach[i, j] >= trig["prob"]]
        crit_hit = [k for k in breached if k in crit]
        multi = len(breached) >= trig["min_kris_breached"]
        if crit_hit or multi or p_audit[i] >= tiers["full_audit"]["prob"]:
            tier = "FULL_AUDIT"
        elif p_onsite[i] >= tiers["targeted_onsite"]["prob"]:
            tier = "TARGETED_ONSITE"
        elif p_remote[i] >= tiers["remote_review"]["prob"]:
            tier = "REMOTE_REVIEW"
        else:
            tier = "ROUTINE"

        order = np.argsort(-contrib[i])[:3]
        drivers = "; ".join(
            f"{KRI_INFO[kris[j]]['label']}: {mean_th[i, j]:.1%} vs peers {ref[j]:.1%}"
            for j in order if contrib[i, j] > 0.005) or "none"
        bits = []
        if crit_hit:
            bits.append("critical-class KRI far above peers (" + ", ".join(crit_hit) + ")")
        elif multi:
            bits.append("several KRIs far above peers (" + ", ".join(breached) + ")")
        bits.append(f"risk index {eri_mean[i]:.2f} [95% CI {eri_ci[0, i]:.2f}-{eri_ci[1, i]:.2f}]")
        bits.append(f"P(index >= {tiers['full_audit']['index']:.2f}) = {p_audit[i]:.2f}")
        rows.append({
            "site_id": s, "n_subjects": int(N[i, dev_idx]) if dev_idx is not None else np.nan,
            "risk_index": round(float(eri_mean[i]), 4), "ci_lo": round(float(eri_ci[0, i]), 4),
            "ci_hi": round(float(eri_ci[1, i]), 4), "p_ge_remote": round(float(p_remote[i]), 3),
            "p_ge_onsite": round(float(p_onsite[i]), 3), "p_ge_audit": round(float(p_audit[i]), 3),
            "n_breached": len(breached), "breached_kris": ", ".join(breached),
            "naive_index": round(float(naive[i]), 4), "tier": tier,
            "recommendation": ACTIONS[tier], "drivers": drivers, "reason": "; ".join(bits)})

    site_risk = pd.DataFrame(rows)
    site_risk["tier_rank"] = site_risk["tier"].map({t: r for r, t in enumerate(TIER_ORDER)})
    site_risk = (site_risk.sort_values(["tier_rank", "risk_index"], ascending=False)
                 .drop(columns="tier_rank").reset_index(drop=True))
    site_risk["rank"] = np.arange(1, len(site_risk) + 1)

    kt = []
    for i, s in enumerate(sites):
        for j, k in enumerate(kris):
            kt.append({"site_id": s, "kri": k, "events": int(X[i, j]), "opportunities": int(N[i, j]),
                       "raw_rate": float(raw[i, j]), "post_mean": float(mean_th[i, j]),
                       "ci_lo": float(th_ci[0, i, j]), "ci_hi": float(th_ci[1, i, j]),
                       "ref_rate": float(ref[j]), "risk_ratio": float(mean_th[i, j] / ref[j]),
                       "p_exceed": float(p_exceed[i, j]), "p_breach": float(p_breach[i, j]),
                       "p_outlier": float(resp[i, j]),
                       "data_weight": float(N[i, j] / (N[i, j] + kappa[j])),
                       "contribution": float(contrib[i, j])})
    return RiskResult(site_risk, pd.DataFrame(kt), pd.DataFrame(prior_rows),
                      pd.DataFrame(excess_mean, index=sites, columns=kris),
                      dict(zip(kris, w)), c)
