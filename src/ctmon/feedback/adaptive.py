"""Human-in-the-loop learning from monitor feedback (Phase 7 module, reproduced for the dashboard tests)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

PRIOR_DETECTOR_WEIGHTS = {"iforest": 0.40, "mahal": 0.35, "lof": 0.25}


class FeedbackLog:
    COLS = ["alert_id", "is_true_issue", "reviewer", "timestamp", "comment"]

    def __init__(self):
        self._rows: list = []

    def add(self, alert_id: str, is_true_issue: bool, reviewer: str = "monitor",
            comment: str = "", timestamp: str | None = None) -> None:
        ts = timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._rows.append({"alert_id": alert_id, "is_true_issue": bool(is_true_issue),
                           "reviewer": reviewer, "timestamp": ts, "comment": comment})

    def __len__(self) -> int:
        return len(self._rows)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=self.COLS)

    def labels(self) -> dict:
        return {r["alert_id"]: r["is_true_issue"] for r in self._rows}

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)

    @classmethod
    def load(cls, path) -> "FeedbackLog":
        log = cls()
        for r in pd.read_csv(path).fillna("").to_dict("records"):
            log.add(r["alert_id"], bool(r["is_true_issue"]), r["reviewer"], r["comment"], r["timestamp"])
        return log


@dataclass
class LearnerConfig:
    C: float = 0.5
    min_labels: int = 8
    n0: float = 20.0
    review_cost: float = 1.0
    miss_cost: float = 4.0

    @property
    def p_star(self) -> float:
        return self.review_cost / self.miss_cost


class FeedbackLearner:
    def __init__(self, X: pd.DataFrame, base_score, cfg: LearnerConfig | None = None):
        self.cfg = cfg or LearnerConfig()
        self.cols = list(X.columns)
        self.ids = list(X.index)
        self.Z = StandardScaler().fit_transform(X.to_numpy(float))
        self.base = np.asarray(base_score, dtype=float)
        self.base_rank = pd.Series(self.base).rank(pct=True).to_numpy()
        self.model = None
        self.n_labeled = 0

    def fit(self, idx, y) -> bool:
        idx, y = np.asarray(idx, dtype=int), np.asarray(y, dtype=int)
        self.n_labeled, self.model = len(y), None
        if len(y) < self.cfg.min_labels or len(np.unique(y)) < 2:
            return False
        self.model = LogisticRegression(C=self.cfg.C, max_iter=2000).fit(self.Z[idx], y)
        return True

    def fit_from_log(self, log: FeedbackLog) -> bool:
        lab = log.labels()
        pos = [i for i, a in enumerate(self.ids) if a in lab]
        return self.fit(pos, [int(lab[self.ids[i]]) for i in pos])

    def proba(self):
        return None if self.model is None else self.model.predict_proba(self.Z)[:, 1]

    def score(self) -> np.ndarray:
        p = self.proba()
        if p is None:
            return self.base_rank.copy()
        a = self.n_labeled / (self.n_labeled + self.cfg.n0)
        return (1 - a) * self.base_rank + a * pd.Series(p).rank(pct=True).to_numpy()

    def coefficients(self):
        return None if self.model is None else pd.Series(self.model.coef_[0], index=self.cols)


def implied_detector_weights(coefs, prior: dict | None = None) -> dict:
    prior = prior or PRIOR_DETECTOR_WEIGHTS
    if coefs is None:
        return dict(prior)
    pos = {k: max(float(coefs.get(k, 0.0)), 0.0) for k in prior}
    s = sum(pos.values())
    return dict(prior) if s <= 1e-9 else {k: v / s for k, v in pos.items()}


def _ap(y, s) -> float:
    y = np.asarray(y, dtype=int)
    return float("nan") if len(y) == 0 or y.sum() == 0 else float(average_precision_score(y, s))


def static_curve(y, base, rounds: int, batch: int) -> pd.DataFrame:
    y, base = np.asarray(y, int), np.asarray(base, float)
    order = np.argsort(-base, kind="stable")
    reviewed, rows = np.zeros(len(y), bool), []
    for r in range(1, rounds + 1):
        pick = order[(r - 1) * batch: r * batch]
        if len(pick) == 0:
            break
        reviewed[pick] = True
        rem = ~reviewed
        rows.append({"round": r, "reviewed": int(reviewed.sum()), "found": int(y[reviewed].sum()),
                     "found_sd": 0.0, "batch_precision": float(y[pick].mean()),
                     "ap_remaining": _ap(y[rem], base[rem])})
    df = pd.DataFrame(rows)
    df.insert(0, "strategy", "static")
    return df


def simulate_feedback(X, y_true, base_score, rounds: int = 8, batch: int = 15, strategy: str = "top",
                      explore: float = 0.3, label_noise: float = 0.05,
                      cfg: LearnerConfig | None = None, seed: int = 0):
    cfg = cfg or LearnerConfig()
    y = np.asarray(y_true, int)
    rng = np.random.default_rng(seed)
    learner = FeedbackLearner(X, base_score, cfg)
    reviewed = np.zeros(len(y), bool)
    lab_idx, lab_y, rows = [], [], []
    for r in range(1, rounds + 1):
        avail = np.flatnonzero(~reviewed)
        if len(avail) == 0:
            break
        b = min(batch, len(avail))
        order = avail[np.argsort(-learner.score()[avail], kind="stable")]
        n_exp = int(round(explore * b)) if (strategy == "mixed" and learner.model is not None) else 0
        pick = list(order[: b - n_exp])
        if n_exp:
            pool = order[b - n_exp: max(b, len(order) // 2)]
            pick += list(rng.choice(pool, size=n_exp, replace=False))
        pick = np.array(pick, dtype=int)
        truth = y[pick]
        noisy = np.where(rng.random(len(pick)) < label_noise, 1 - truth, truth)
        reviewed[pick] = True
        lab_idx += list(pick)
        lab_y += list(noisy)
        learner.fit(lab_idx, lab_y)
        rem = ~reviewed
        rows.append({"round": r, "reviewed": int(reviewed.sum()), "found": int(y[reviewed].sum()),
                     "batch_precision": float(truth.mean()),
                     "ap_remaining": _ap(y[rem], learner.score()[rem])})
    final: dict = {}
    p = learner.proba()
    if p is not None:
        rem = np.flatnonzero(~reviewed)
        flag = rem[p[rem] >= cfg.p_star]
        k = len(flag)
        pos_rem = max(int(y[rem].sum()), 1)
        stat = rem[np.argsort(-np.asarray(base_score, float)[rem], kind="stable")[:k]]
        final = {"remaining_alerts": len(rem), "remaining_true": int(y[rem].sum()), "flagged_k": k,
                 "learned_precision": float(y[flag].mean()) if k else float("nan"),
                 "learned_recall": float(y[flag].sum()) / pos_rem,
                 "static_precision": float(y[stat].mean()) if k else float("nan"),
                 "static_recall": float(y[stat].sum()) / pos_rem}
    return pd.DataFrame(rows), final, learner


def run_feedback_experiment(X, y, base, rounds: int = 8, batch: int = 15, n_sims: int = 20,
                            label_noise: float = 0.05, explore: float = 0.3,
                            cfg: LearnerConfig | None = None, seed: int = 0) -> dict:
    y = np.asarray(y, int)
    n, P = len(y), int(y.sum())
    curves = [static_curve(y, base, rounds, batch)]
    finals, coefs = [], None
    for strat in ("top", "mixed"):
        runs = []
        for s in range(n_sims):
            df, fin, lrn = simulate_feedback(X, y, base, rounds, batch, strat, explore,
                                             label_noise, cfg, seed + s)
            df["sim"] = s
            runs.append(df)
            finals.append({"strategy": f"feedback-{strat}", **fin})
            if strat == "top" and s == 0:
                coefs = lrn.coefficients()
        g = (pd.concat(runs).groupby("round")
             .agg(reviewed=("reviewed", "first"), found=("found", "mean"), found_sd=("found", "std"),
                  batch_precision=("batch_precision", "mean"), ap_remaining=("ap_remaining", "mean"))
             .reset_index())
        g.insert(0, "strategy", f"feedback-{strat}")
        curves.append(g)
    oracle = pd.DataFrame({"strategy": "oracle", "round": range(1, rounds + 1)})
    oracle["reviewed"] = np.minimum(oracle["round"] * batch, n)
    oracle["found"] = np.minimum(oracle["reviewed"], P)
    oracle["found_sd"], oracle["batch_precision"], oracle["ap_remaining"] = 0.0, np.nan, np.nan
    curves.append(oracle)
    final = pd.DataFrame(finals).groupby("strategy").mean(numeric_only=True)
    return {"curves": pd.concat(curves, ignore_index=True), "final": final,
            "coefficients": coefs, "n_pool": n, "n_true": P}


class RuleFeedbackTracker:
    def __init__(self, prior=(1.0, 1.0)):
        self.prior, self.tp, self.fp = prior, {}, {}

    def update(self, rule_id: str, is_true: bool) -> None:
        d = self.tp if is_true else self.fp
        d[rule_id] = d.get(rule_id, 0) + 1

    def update_many(self, rule_ids, is_true) -> None:
        for r, t in zip(rule_ids, is_true):
            self.update(r, bool(t))

    def summary(self, min_precision: float = 0.8, conf: float = 0.9) -> pd.DataFrame:
        rows = []
        for r in sorted(set(self.tp) | set(self.fp)):
            tp, fp = self.tp.get(r, 0), self.fp.get(r, 0)
            a, b = self.prior[0] + tp, self.prior[1] + fp
            below = float(stats.beta.cdf(min_precision, a, b))
            rows.append({"rule_id": r, "reviewed": tp + fp, "confirmed": tp, "rejected": fp,
                         "precision_mean": a / (a + b),
                         "ci_lo": float(stats.beta.ppf(0.025, a, b)),
                         "ci_hi": float(stats.beta.ppf(0.975, a, b)),
                         "p_precision_below": below,
                         "action": "REVIEW RULE (too many false alarms)" if below >= conf else "keep"})
        return pd.DataFrame(rows)
