"""Cycle 10: TreeSHAP interpretability for saved RF + XGB models."""
from __future__ import annotations
import numpy as np, pandas as pd

# Feature-family classification (prefix-based). Order matters: check specific first.
def feature_family(name: str) -> str:
    n = name.lower()
    if n.startswith("miss_") or n.endswith("_missing") or "_isna" in n or "_mask" in n:
        return "missing_flag"
    if n.startswith("y_lag") or n.startswith("y_roll") or n.startswith("y_trend") or n.startswith("y_diff"):
        return "autoregressive"
    if n.startswith("sin_") or n.startswith("cos_") or n in ("quarter", "q", "qtr") or "quarter" in n:
        return "seasonal"
    if any(n.startswith(p) for p in ("rainfall", "t2m", "precip", "temp")):
        return "covariate"
    if n in ("lat", "lon", "latitude", "longitude", "welldepth", "well_depth", "elevation") or n.startswith("static_"):
        return "static"
    return "other"

def model_feature_names(model, kind: str) -> list[str]:
    """Canonical feature order the model was trained with."""
    if kind == "rf":
        return list(getattr(model, "feature_names_in_"))
    # xgboost
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    fn = booster.feature_names
    if fn is None:
        raise ValueError("XGB booster has no feature_names; cannot align SHAP.")
    return list(fn)

def mean_abs_shap(shap_vals: np.ndarray) -> np.ndarray:
    # shap_vals shape (n_rows, n_feat); RF regressor returns a single array.
    return np.abs(shap_vals).mean(axis=0)

def family_shares(feat_names, mabs) -> dict:
    fam = {}
    for f, v in zip(feat_names, mabs):
        fam[feature_family(f)] = fam.get(feature_family(f), 0.0) + float(v)
    tot = sum(fam.values()) or 1.0
    return {k: v / tot for k, v in sorted(fam.items(), key=lambda kv: -kv[1])}

def spearman(a, b) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
