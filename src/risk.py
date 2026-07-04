import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def expected_calibration_error(y_true, p_hat, n_bins=10):
    """
    10 equal-width probability bins [0, 1].
    Returns: ECE, reliability_df (cols: bin, mean_phat, emp_freq, count)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(p_hat, bins, right=True)
    bin_idx[bin_idx == 0] = 1
    
    ece = 0.0
    n = len(y_true)
    rel_data = []
    
    for i in range(1, n_bins + 1):
        mask = (bin_idx == i)
        count = np.sum(mask)
        if count > 0:
            mean_p = np.mean(p_hat[mask])
            emp_p = np.mean(y_true[mask])
            ece += (count / n) * np.abs(mean_p - emp_p)
            rel_data.append({
                "bin": i,
                "mean_phat": mean_p,
                "emp_freq": emp_p,
                "count": count
            })
        else:
            rel_data.append({
                "bin": i,
                "mean_phat": np.nan,
                "emp_freq": np.nan,
                "count": 0
            })
            
    rel_df = pd.DataFrame(rel_data)
    return ece, rel_df

def compute_roc_auc(y_true, p_hat):
    """
    Compute ROC AUC, returning NaN if y_true doesn't have 2 classes.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, p_hat)
