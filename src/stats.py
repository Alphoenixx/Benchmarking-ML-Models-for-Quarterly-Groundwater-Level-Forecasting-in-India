"""Cycle 9: statistical-significance + stratified-robustness helpers.
Operates purely on the unified predictions table (no retraining)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def per_well_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(station,split,h,model) MSE/MAE/RMSE and point count."""
    d = df.copy()
    e = d["y"] - d["yhat"]
    d["se"] = e ** 2
    d["ae"] = e.abs()
    g = d.groupby(["station_id", "split", "h", "model"], observed=True)
    out = g.agg(mse=("se", "mean"), mae=("ae", "mean"), n=("se", "size")).reset_index()
    out["rmse"] = np.sqrt(out["mse"])
    return out

def wilcoxon_pair(pw, split, h, model_a, model_b, metric="rmse", min_n=2):
    """Paired Wilcoxon signed-rank of metric[a]-metric[b] across wells where
    BOTH models have >= min_n test points. Negative median => a is better."""
    a = pw[(pw.split == split) & (pw.h == h) & (pw.model == model_a) & (pw.n >= min_n)]
    b = pw[(pw.split == split) & (pw.h == h) & (pw.model == model_b) & (pw.n >= min_n)]
    m = a[["station_id", metric]].merge(
        b[["station_id", metric]], on="station_id", suffixes=("_a", "_b"))
    diff = (m[metric + "_a"] - m[metric + "_b"]).to_numpy()
    n = int(len(diff))
    res = {"split": split, "h": int(h), "model_a": model_a, "model_b": model_b,
           "metric": metric, "n_wells": n,
           "median_diff": float(np.median(diff)) if n else np.nan,
           "mean_diff": float(np.mean(diff)) if n else np.nan,
           "a_better_frac": float(np.mean(diff < 0)) if n else np.nan}
    nz = diff[diff != 0]
    if len(nz) >= 10:
        w = stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
        res["wilcoxon_stat"] = float(w.statistic)
        res["p_value"] = float(w.pvalue)
    else:
        res["wilcoxon_stat"] = np.nan
        res["p_value"] = np.nan
    return res

def diebold_mariano(df, split, h, model_a, model_b, loss="se"):
    """Pooled Diebold-Mariano with HLN small-sample correction.
    d = L(a)-L(b); negative mean => a better. Lags up to h-1 for overlap."""
    key = ["station_id", "period"]
    a = df[(df.split == split) & (df.h == h) & (df.model == model_a)][key + ["y", "yhat"]]
    b = df[(df.split == split) & (df.h == h) & (df.model == model_b)][key + ["y", "yhat"]]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    ea = (m.y_a - m.yhat_a).to_numpy()
    eb = (m.y_b - m.yhat_b).to_numpy()
    d = ea ** 2 - eb ** 2 if loss == "se" else np.abs(ea) - np.abs(eb)
    T = int(len(d))
    if T < 8:
        return {"split": split, "h": int(h), "model_a": model_a, "model_b": model_b,
                "loss": loss, "n_points": T, "mean_loss_diff": np.nan,
                "dm_stat": np.nan, "dm_hln": np.nan, "p_value": np.nan}
    dbar = float(np.mean(d))
    lag = max(int(h) - 1, 0)
    s = float(np.var(d, ddof=0))
    for k in range(1, lag + 1):
        s += 2.0 * float(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
    var_dbar = s / T
    dm = dbar / np.sqrt(var_dbar) if var_dbar > 0 else np.nan
    hh = int(h)
    corr = np.sqrt(max((T + 1 - 2 * hh + hh * (hh - 1) / T) / T, 1e-12))
    dm_hln = dm * corr if np.isfinite(dm) else np.nan
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=T - 1)) if np.isfinite(dm_hln) else np.nan
    return {"split": split, "h": int(h), "model_a": model_a, "model_b": model_b,
            "loss": loss, "n_points": T, "mean_loss_diff": dbar,
            "dm_stat": float(dm) if np.isfinite(dm) else np.nan,
            "dm_hln": float(dm_hln) if np.isfinite(dm_hln) else np.nan,
            "p_value": float(p) if np.isfinite(p) else np.nan}

def binom_winrate(pw, split, h, model, ref, min_n=2):
    """Binomial test that model beats ref on per-well RMSE in >50% of wells."""
    a = pw[(pw.split == split) & (pw.h == h) & (pw.model == model) & (pw.n >= min_n)]
    b = pw[(pw.split == split) & (pw.h == h) & (pw.model == ref) & (pw.n >= min_n)]
    m = a[["station_id", "rmse"]].merge(
        b[["station_id", "rmse"]], on="station_id", suffixes=("_m", "_r"))
    d = (m.rmse_m - m.rmse_r).to_numpy()
    d = d[d != 0]
    wins, n = int(np.sum(d < 0)), int(len(d))
    if n == 0:
        return {"split": split, "h": int(h), "model": model, "ref": ref,
                "n_wells": 0, "wins": 0, "win_rate": np.nan, "p_value": np.nan}
    bt = stats.binomtest(wins, n, 0.5, alternative="two-sided")
    return {"split": split, "h": int(h), "model": model, "ref": ref,
            "n_wells": n, "wins": wins, "win_rate": wins / n, "p_value": float(bt.pvalue)}
