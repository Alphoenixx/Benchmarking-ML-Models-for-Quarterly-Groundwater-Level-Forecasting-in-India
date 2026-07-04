import numpy as np
import pandas as pd
from scipy.stats import norm

def get_prob_climatology(train_df: pd.DataFrame, test_df: pd.DataFrame, tau_grid: np.ndarray) -> np.ndarray:
    """
    Computes a probabilistic climatology baseline.
    train_df must have: station_id, q (quarter), y (target value), region (bool or str).
    test_df must have: station_id, q.
    """
    train_df = train_df.dropna(subset=["y"]).copy()
    
    # Precompute pools
    # 1. well-quarter
    wq_pool = {}
    for (sid, q_val), grp in train_df.groupby(["station_id", "q"]):
        wq_pool[(sid, q_val)] = grp["y"].values
        
    # 2. well (all quarters)
    w_pool = {}
    for sid, grp in train_df.groupby("station_id"):
        w_pool[sid] = grp["y"].values
        
    # 3. region-quarter
    rq_pool = {}
    for (reg, q_val), grp in train_df.groupby(["region", "q"]):
        rq_pool[(reg, q_val)] = grp["y"].values
        
    preds = np.zeros((len(test_df), len(tau_grid)))
    
    # Generate predictions
    for i, (_, row) in enumerate(test_df.iterrows()):
        sid = row["station_id"]
        q_val = row["q"]
        reg = row["region"]
        
        y_vals = wq_pool.get((sid, q_val), np.array([]))
        
        if len(y_vals) >= 8:
            q_arr = np.quantile(y_vals, tau_grid)
        else:
            # Fallback to Gaussian
            fallback_vals = w_pool.get(sid, np.array([]))
            if len(fallback_vals) < 8:
                fallback_vals = rq_pool.get((reg, q_val), np.array([]))
            
            if len(fallback_vals) == 0:
                # Absolute fallback
                fallback_vals = train_df["y"].values
                
            mean_val = np.mean(fallback_vals)
            sd_val = np.std(fallback_vals)
            if sd_val < 1e-3:
                # Try region-quarter sd
                rq_vals = rq_pool.get((reg, q_val), np.array([]))
                if len(rq_vals) > 1:
                    sd_val = np.std(rq_vals)
                if sd_val < 1e-3:
                    sd_val = 1e-3
            
            q_arr = norm.ppf(tau_grid, loc=mean_val, scale=sd_val)
            
        # Ensure strictly increasing (to avoid any degenerate cross-over)
        # norm.ppf is strictly increasing if sd > 0.
        # np.quantile is non-decreasing. To make strictly increasing, we can add a tiny epsilon.
        q_arr = np.maximum.accumulate(q_arr)
        # add tiny strictly increasing noise
        q_arr += np.linspace(0, 1e-5, len(q_arr))
        
        preds[i] = q_arr
        
    return preds

def decompose_crps(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Decomposes CRPS by specified group_cols.
    df must have 'crps', 'crps_baseline' (prob climatology), and group_cols.
    """
    res = df.groupby(group_cols).agg(
        mean_crps=("crps", "mean"),
        mean_baseline=("crps_baseline", "mean"),
        n=("crps", "count")
    ).reset_index()
    
    res["crps_skill_vs_prob_climatology"] = 1.0 - res["mean_crps"] / res["mean_baseline"]
    return res
