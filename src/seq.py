"""Cycle 8 — sequence sample construction for global DL models.
Univariate (target + seasonal + static geo), left-padded windows with a validity
mask; eligibility identical to Chronos (>=MIN_CTX finite points up to the origin)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.dataset import reindex_well, split_of_year

MIN_CTX = 4     # match Chronos exactly
L = 12          # lookback window length (quarters)

def _qenc(ts):
    q = pd.Timestamp(ts).quarter
    a = 2.0 * np.pi * (q - 1) / 4.0
    return np.sin(a), np.cos(a)

def per_well_norm_stats(panel):
    """Per-well target mean/std over TRAIN period (year<=2018), finite only."""
    stats = {}
    for sid, g in panel.groupby("station_id"):
        d = pd.to_datetime(g["datetime"])
        y = g["target"].to_numpy(dtype=float)
        m = np.isfinite(y) & (d.dt.year.to_numpy() <= 2018)
        if m.sum() >= 2:
            mu, sd = float(np.nanmean(y[m])), float(np.nanstd(y[m]))
        elif np.isfinite(y).sum() >= 1:
            mu, sd = float(np.nanmean(y)), float(np.nanstd(y))
        else:
            mu, sd = 0.0, 1.0
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        stats[sid] = (mu, sd)
    return stats

def static_norm_stats(panel):
    s = panel.groupby("station_id")[["lat", "lon", "wellDepth"]].first()
    s["wellDepth"] = s["wellDepth"].fillna(s["wellDepth"].median())
    mu = s.mean()
    sd = s.std().replace(0, 1.0)
    return mu, sd, s

def build_samples(panel, ids, horizons, ynorm, smu, ssd, svals, scale_map):
    """Build DL samples for the given wells across ALL splits.
    Returns Xs[N,L,4], Xt[N,3], h_idx[N], y_norm[N], y_raw[N], meta(DataFrame)."""
    ids = set(ids)
    Xs, Xt, hh, yn, yr, meta = [], [], [], [], [], []
    for sid, g in panel.groupby("station_id"):
        if sid not in ids:
            continue
        grid = reindex_well(g)
        y = grid["target"].to_numpy(dtype=float)
        dates = pd.to_datetime(grid["datetime"]).to_numpy()
        yi = pd.Series(y).interpolate(limit_direction="both").to_numpy()
        n = len(y)
        mu, sd = ynorm[sid]
        st = ((svals.loc[sid] - smu) / ssd).to_numpy(dtype=float)
        sc = float(scale_map.get(sid, np.nan))
        sinq = np.empty(n); cosq = np.empty(n)
        for t in range(n):
            sinq[t], cosq[t] = _qenc(dates[t])
        for j in range(n):
            if not np.isfinite(y[j]):
                continue
            yr_j = pd.Timestamp(dates[j]).year
            sp = split_of_year(yr_j)
            if sp not in ("train", "val", "test"):
                continue
            for h in horizons:
                o = j - h
                if o < 0:
                    continue
                if np.isfinite(y[:o + 1]).sum() < MIN_CTX:
                    continue
                lo = max(0, o - L + 1)
                win = np.arange(lo, o + 1)
                seq = np.zeros((L, 4), dtype=np.float32)
                start = L - len(win)
                seq[start:, 0] = (yi[win] - mu) / sd
                seq[start:, 1] = sinq[win]
                seq[start:, 2] = cosq[win]
                seq[start:, 3] = 1.0
                Xs.append(seq)
                Xt.append(st.astype(np.float32))
                hh.append(h - 1)
                yn.append((y[j] - mu) / sd)
                yr.append(y[j])
                meta.append((sid, sp, h,
                             pd.Timestamp(dates[j]).date().isoformat(), sc, mu, sd))
    meta = pd.DataFrame(meta, columns=["station_id", "split", "h", "period",
                                       "scale", "mu", "sd"])
    return (np.asarray(Xs, dtype=np.float32), np.asarray(Xt, dtype=np.float32),
            np.asarray(hh, dtype=np.int64), np.asarray(yn, dtype=np.float32),
            np.asarray(yr, dtype=np.float32), meta)
