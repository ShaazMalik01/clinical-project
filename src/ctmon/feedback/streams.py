"""Turn detector output into alert 'streams' (features + unsupervised score + truth) for feedback learning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.ml_eval import _subject_labels
from ..explain.zscores import visit_z_features, zscores
from ..ml.features import SUBJECT_MODEL_COLS


def visit_truth(pool: pd.DataFrame, gt: pd.DataFrame) -> np.ndarray:
    g = gt[gt["visit_num"].notna()]
    keys = {(r.subject_id, int(r.visit_num)) for r in g.itertuples()}
    return np.array([(s, int(v)) in keys for s, v in zip(pool["subject_id"], pool["visit_num"])], dtype=int)


def subject_truth(pool: pd.DataFrame, gt: pd.DataFrame) -> np.ndarray:
    ids = set(_subject_labels(gt)["subject_id"])
    return pool["subject_id"].isin(ids).astype(int).to_numpy()


def build_visit_stream(vs: pd.DataFrame, frames: dict, gt: pd.DataFrame | None = None,
                       pool_frac: float = 0.06):
    n = max(20, int(round(pool_frac * len(vs))))
    pool = vs.sort_values("ensemble", ascending=False).head(n).reset_index(drop=True)
    zf = visit_z_features(frames, pool)
    nlp = np.minimum(-np.log10(np.clip(np.nan_to_num(pool["mahal_p"].to_numpy(float), nan=1.0),
                                       1e-300, 1.0)), 50.0) / 10.0
    ids = [f"MLV-{s}-V{int(v)}" for s, v in zip(pool["subject_id"], pool["visit_num"])]
    X = pd.DataFrame({"iforest": pool["iforest"].to_numpy(float), "mahal": pool["mahal"].to_numpy(float),
                      "lof": pool["lof"].to_numpy(float), "mahal_nlp": nlp,
                      "z_level": np.log1p(zf["z_level"].to_numpy()),
                      "z_dev": np.log1p(zf["z_dev"].to_numpy()),
                      "z_step": np.log1p(zf["z_step"].to_numpy()),
                      "is_labs": (pool["view"] == "labs").astype(float).to_numpy()}, index=ids)
    base = pd.Series(pool["ensemble"].to_numpy(float), index=ids)
    y = visit_truth(pool, gt) if gt is not None else None
    return X, base, y, pool


def build_subject_stream(ss: pd.DataFrame, sf: pd.DataFrame, gt: pd.DataFrame | None = None,
                         pool_frac: float = 0.25, z_clip: float = 6.0):
    n = max(20, int(round(pool_frac * len(ss))))
    pool = ss.sort_values("ensemble", ascending=False).head(n).reset_index(drop=True)
    z, _ = zscores(sf[SUBJECT_MODEL_COLS])
    Z = z.loc[pool["subject_id"]].clip(-z_clip, z_clip).reset_index(drop=True)
    Z.columns = [f"z_{c}" for c in Z.columns]
    R = pool[["iforest", "mahal", "lof"]].reset_index(drop=True)
    X = pd.concat([R, Z], axis=1)
    X.index = [f"MLS-{s}" for s in pool["subject_id"]]
    base = pd.Series(pool["ensemble"].to_numpy(float), index=X.index)
    y = subject_truth(pool, gt) if gt is not None else None
    return X, base, y, pool
