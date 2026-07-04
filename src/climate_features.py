"""Cycle 11: leakage-safe engineered climate features (quarterly cadence)."""
import numpy as np
import pandas as pd

# engineered climate columns added to the quarterly panel (pre-lagging)
ENGINEERED_CLIMATE = [
    "rain_cum2", "rain_cum4", "rain_cum8",   # trailing cumulative rainfall (causal)
    "rain_anom",                             # rainfall seasonal anomaly (train clim)
    "spi4", "spi8",                          # standardized precip index (z-score proxy)
    "t2m_anom",                              # temperature seasonal anomaly (train clim)
]

TRAIN_MAX_YEAR = 2018   # matches the locked split (train <= 2018)
MIN_TRAIN_OBS  = 3      # per well+quarter before falling back to national stats

def _quarter(dt: pd.Series) -> pd.Series:
    return dt.dt.quarter.astype(int)

def _seasonal_stats(df: pd.DataFrame, col: str):
    """TRAIN-only per (well, quarter) mean/std, with national per-quarter fallback.
    Returns two dicts keyed for a fast map."""
    tr = df[df["datetime"].dt.year <= TRAIN_MAX_YEAR].copy()
    tr["q"] = _quarter(tr["datetime"])
    # national per-quarter fallback
    nat = tr.groupby("q")[col].agg(["mean", "std"])
    nat_mean = nat["mean"].to_dict()
    nat_std = nat["std"].replace(0, np.nan).to_dict()
    # per well+quarter
    g = tr.groupby(["station_id", "q"])[col].agg(["mean", "std", "count"])
    wm, ws = {}, {}
    for (sid, q), row in g.iterrows():
        wm[(sid, q)] = row["mean"] if row["count"] >= MIN_TRAIN_OBS else nat_mean.get(q, np.nan)
        s = row["std"] if (row["count"] >= MIN_TRAIN_OBS and pd.notna(row["std"]) and row["std"] > 1e-6) else nat_std.get(q, np.nan)
        ws[(sid, q)] = s
    return wm, ws, nat_mean, nat_std

def add_climate_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Adds ENGINEERED_CLIMATE columns to a quarterly panel.
    Requires columns: station_id, datetime (datetime64), rainfall, t2m.
    Cumulative sums are strictly causal (current + past quarters, per well, time-sorted).
    Anomalies / SPI z-scores use TRAIN-only per-well+quarter climatology."""
    df = panel.sort_values(["station_id", "datetime"]).copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    q = _quarter(df["datetime"])
    grp = df.groupby("station_id", group_keys=False)
    # trailing cumulative rainfall (causal rolling sums; min_periods=1)
    df["rain_cum2"] = grp["rainfall"].apply(lambda s: s.rolling(2, min_periods=1).sum())
    df["rain_cum4"] = grp["rainfall"].apply(lambda s: s.rolling(4, min_periods=1).sum())
    df["rain_cum8"] = grp["rainfall"].apply(lambda s: s.rolling(8, min_periods=1).sum())
    
    # rainfall seasonal anomaly
    rm, rs, _, _ = _seasonal_stats(df, "rainfall")
    key = list(zip(df["station_id"], q))
    df["rain_anom"] = df["rainfall"].to_numpy() - np.array([rm.get(k, np.nan) for k in key])
    
    # SPI proxy = z-score of trailing accumulation vs TRAIN clim of that accumulation
    for win, name in [("rain_cum4", "spi4"), ("rain_cum8", "spi8")]:
        m, s, _, _ = _seasonal_stats(df, win)
        mu = np.array([m.get(k, np.nan) for k in key])
        sd = np.array([s.get(k, np.nan) for k in key])
        df[name] = (df[win].to_numpy() - mu) / sd
        
    # temperature seasonal anomaly
    tm, ts, _, _ = _seasonal_stats(df, "t2m")
    df["t2m_anom"] = df["t2m"].to_numpy() - np.array([tm.get(k, np.nan) for k in key])
    
    # clean up non-finite (division / fallback gaps) -> 0.0, and add a coverage flag
    for c in ENGINEERED_CLIMATE:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df
