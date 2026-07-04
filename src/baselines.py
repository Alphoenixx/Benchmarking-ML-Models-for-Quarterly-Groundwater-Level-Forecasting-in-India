import numpy as np
from src.dataset import M, split_of_year

def per_well_eval(g, horizons=(1, 2, 3, 4), include_period=False):
    """g: single reindexed well (continuous quarterly grid)."""
    g = g.reset_index(drop=True)
    y = g["target"].to_numpy(dtype=float)
    yr = g["datetime"].dt.year.to_numpy()
    q = g["datetime"].dt.quarter.to_numpy()
    sid = g["station_id"].iloc[0]
    train = np.array([split_of_year(a) == "train" for a in yr])

    # per-well quarterly climatology (train only)
    clim = {}
    for qq in (1, 2, 3, 4):
        v = y[(q == qq) & train]
        v = v[~np.isnan(v)]
        clim[qq] = v.mean() if len(v) else np.nan

    # MASE scale = in-sample seasonal-naive MAE on train
    sn = [abs(y[j] - y[j - M]) for j in range(M, len(y))
          if train[j] and not np.isnan(y[j]) and not np.isnan(y[j - M])]
    scale = float(np.mean(sn)) if sn else np.nan

    rows = []
    for h in horizons:
        for j in range(len(y)):
            sp = split_of_year(yr[j])
            if sp in ("train", "exclude"):
                continue
            if np.isnan(y[j]):
                continue
            preds = {}
            if j - h >= 0 and not np.isnan(y[j - h]):
                preds["persistence"] = y[j - h]
            if j - M >= 0 and not np.isnan(y[j - M]):
                preds["seasonal_naive"] = y[j - M]
            if not np.isnan(clim.get(q[j], np.nan)):
                preds["climatology"] = clim[q[j]]
            for name, yhat in preds.items():
                if include_period:
                    rows.append((sid, sp, h, name, float(y[j]), float(yhat),
                                 scale, str(g["datetime"].iloc[j].date())))
                else:
                    rows.append((sid, sp, h, name, float(y[j]), float(yhat), scale))
    return rows
