"""The three unsupervised detectors. All return scores where HIGHER = MORE ANOMALOUS."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


@dataclass
class DetectorConfig:
    n_estimators: int = 300
    lof_neighbors: int = 35
    cov_support: float = 0.75
    seed: int = 42


def robust_scale(X: pd.DataFrame) -> np.ndarray:
    return RobustScaler().fit_transform(X.to_numpy(dtype=float))


def mahalanobis_scores(Z: np.ndarray, support: float, seed: int) -> tuple[np.ndarray, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            est, method = MinCovDet(support_fraction=support, random_state=seed).fit(Z), "MCD"
        except Exception:
            est, method = EmpiricalCovariance().fit(Z), "empirical"
        d2 = est.mahalanobis(Z)
    return np.nan_to_num(d2, nan=0.0, posinf=1e12), method


class DetectorSuite:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self.mahal_method = "MCD"
        self.p_values_: np.ndarray | None = None

    def fit_score(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = robust_scale(X)
        raw = pd.DataFrame(index=X.index)
        raw["iforest"] = self._iforest(Z)
        raw["lof"] = self._lof(Z)
        d2, self.mahal_method = mahalanobis_scores(Z, self.cfg.cov_support, self.cfg.seed)
        raw["mahal"] = d2
        self.p_values_ = stats.chi2.sf(d2, df=Z.shape[1])
        return raw

    def _iforest(self, Z: np.ndarray) -> np.ndarray:
        m = IsolationForest(n_estimators=self.cfg.n_estimators, contamination="auto",
                            random_state=self.cfg.seed, n_jobs=-1).fit(Z)
        return -m.score_samples(Z)

    def _lof(self, Z: np.ndarray) -> np.ndarray:
        k = int(min(self.cfg.lof_neighbors, max(2, len(Z) - 1)))
        lof = LocalOutlierFactor(n_neighbors=k, n_jobs=-1)
        lof.fit_predict(Z)
        return np.nan_to_num(-lof.negative_outlier_factor_, nan=1.0, posinf=1e6)


def top_features(X: pd.DataFrame, k: int = 3) -> pd.Series:
    Z = robust_scale(X)
    names = np.array(X.columns)
    order = np.argsort(-np.abs(Z), axis=1)[:, :k]
    text = [", ".join(f"{names[j]}({Z[i, j]:+.1f})" for j in order[i]) for i in range(len(X))]
    return pd.Series(text, index=X.index)
