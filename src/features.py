import numpy as np
import pandas as pd
from src.dataset import reindex_well

TARGET_LAGS = [1, 2, 3, 4, 5, 6, 8]
COVS = ["rainfall", "t2m", "t2m_max", "t2m_min"]
COV_LAGS = [0, 1, 2, 4]   # 0 = value at the origin quarter
EXTRA_COV_LAGS = [0, 1, 2]   # engineered climate lags (kept short to limit dimensionality)

def build_well(g, horizons=(1, 2, 3, 4), extra_covs=None, extra_cov_lags=EXTRA_COV_LAGS):
    g = reindex_well(g).reset_index(drop=True)
    y = g["target"].to_numpy(dtype=float)
    yr = g["datetime"].dt.year.to_numpy()
    q = g["datetime"].dt.quarter.to_numpy()
    n = len(g)
    sid = g["station_id"].iloc[0]
    lat = g["lat"].ffill().bfill().to_numpy()
    lon = g["lon"].ffill().bfill().to_numpy()
    wd = g["wellDepth"].ffill().bfill().to_numpy()
    cov = {c: g[c].to_numpy(dtype=float) for c in COVS}
    
    if extra_covs:
        for c in extra_covs:
            if c in g.columns:
                cov[c] = g[c].to_numpy(dtype=float)

    recs = []
    for o in range(n):
        if np.isnan(y[o]):
            continue  # origin value must exist
        f = {"station_id": sid, "lat": lat[o], "lon": lon[o], "wellDepth": wd[o]}
        for L in TARGET_LAGS:
            f[f"y_lag{L}"] = y[o - L] if o - L >= 0 else np.nan
        win = y[max(0, o - 3):o + 1]; win = win[~np.isnan(win)]
        f["y_roll4_mean"] = win.mean() if len(win) else np.nan
        f["y_roll4_std"] = win.std() if len(win) > 1 else np.nan
        f["y_trend4"] = (y[o] - y[o - 4]) if (o - 4 >= 0 and not np.isnan(y[o - 4])) else np.nan
        f["sin_q"] = np.sin(2 * np.pi * q[o] / 4)
        f["cos_q"] = np.cos(2 * np.pi * q[o] / 4)
        
        for c in COVS:
            for L in COV_LAGS:
                f[f"{c}_lag{L}"] = cov[c][o - L] if o - L >= 0 else np.nan
                
        if extra_covs:
            for c in extra_covs:
                if c in cov:
                    for L in extra_cov_lags:
                        f[f"{c}_lag{L}"] = cov[c][o - L] if o - L >= 0 else np.nan
                        
        for h in horizons:
            f[f"target_h{h}"] = y[o + h] if o + h < n else np.nan
            f[f"target_h{h}_year"] = int(yr[o + h]) if o + h < n else -1
            f[f"target_h{h}_period"] = str(g["datetime"].iloc[o + h].date()) if o + h < n else ""
        recs.append(f)
    return recs
