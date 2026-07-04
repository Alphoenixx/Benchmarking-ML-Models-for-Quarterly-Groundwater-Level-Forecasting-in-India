import numpy as np
import pandas as pd
from src.config import PROC

IGP = {"lat_min": 27.5, "lat_max": 32.5, "lon_min": 73.5, "lon_max": 81.0}
M = 4  # seasonal period (quarters/year)

def load_panel():
    return pd.read_parquet(PROC / "quarterly_panel.parquet")

def split_of_year(y):
    if y <= 2018: return "train"
    if y <= 2020: return "val"
    if y <= 2023: return "test"
    return "exclude"

def cohort(panel, min_q=16, region=None):
    p = panel
    if region == "igp":
        p = p[p.lat.between(IGP["lat_min"], IGP["lat_max"]) &
              p.lon.between(IGP["lon_min"], IGP["lon_max"])]
    q = p.groupby("station_id").size()
    keep = q[q >= min_q].index
    return p[p.station_id.isin(keep)].copy()

def reindex_well(g):
    """Reindex one well to a continuous quarter-start grid; missing quarters -> NaN target."""
    sid = g["station_id"].iloc[0]
    g = g.sort_values("datetime").set_index("datetime")
    full = pd.date_range(g.index.min(), g.index.max(), freq="QS")
    g = g.reindex(full)
    g["station_id"] = sid
    g.index.name = "datetime"
    return g.reset_index()

def seasonal_scale(panel):
    """Per-well in-sample (train) seasonal-naive MAE — the MASE denominator. Must match Cycle 4."""
    out = {}
    for sid, g in panel.groupby("station_id"):
        gg = reindex_well(g).reset_index(drop=True)
        y = gg["target"].to_numpy(dtype=float)
        yr = gg["datetime"].dt.year.to_numpy()
        tr = yr <= 2018
        errs = [abs(y[j] - y[j - M]) for j in range(M, len(y))
                if tr[j] and not np.isnan(y[j]) and not np.isnan(y[j - M])]
        out[sid] = float(np.mean(errs)) if errs else np.nan
    return out

