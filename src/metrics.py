import numpy as np
import pandas as pd

def summarize(df):
    """df cols: station_id, split, h, model, y, yhat, scale (per-well seasonal-naive MAE)."""
    out = []
    for (split, h, model), g in df.groupby(["split", "h", "model"]):
        err = g["yhat"] - g["y"]
        rmse = float(np.sqrt((err ** 2).mean()))
        mae = float(err.abs().mean())
        bias = float(err.mean())
        nse_list, mase_list = [], []
        for sid, gg in g.groupby("station_id"):
            denom = float(((gg["y"] - gg["y"].mean()) ** 2).sum())
            if denom > 0:
                nse_list.append(1 - float(((gg["yhat"] - gg["y"]) ** 2).sum()) / denom)
            sc = gg["scale"].iloc[0]
            if sc and not np.isnan(sc) and sc > 0:
                mase_list.append(float((gg["yhat"] - gg["y"]).abs().mean()) / sc)
        out.append({
            "split": split, "h": int(h), "model": model, "n": int(len(g)),
            "RMSE": round(rmse, 4), "MAE": round(mae, 4), "bias": round(bias, 4),
            "MASE_mean": round(float(np.nanmean(mase_list)), 4) if mase_list else None,
            "NSE_median": round(float(np.nanmedian(nse_list)), 4) if nse_list else None,
        })
    return pd.DataFrame(out).sort_values(["split", "model", "h"]).reset_index(drop=True)
