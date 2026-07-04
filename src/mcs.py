import numpy as np
import scipy.stats as stats

def mcs_procedure(L, station_ids, alpha=0.10, B=2000, seed=42):
    """
    Model Confidence Set (Hansen-Lunde-Nason 2011) with cluster bootstrap.
    L: (n_obs, n_models) aligned loss matrix
    station_ids: (n_obs,) array of well identifiers
    alpha: significance level for elimination
    Returns: mcs_pvalues (n_models,), elim_order (list of model indices)
    """
    n_obs, n_models = L.shape
    rng = np.random.default_rng(seed)
    
    # Precompute per-well loss sums and counts
    # This ensures cluster-robust bootstrapping
    unique_wells, well_indices = np.unique(station_ids, return_inverse=True)
    n_wells = len(unique_wells)
    
    well_sum = np.zeros((n_wells, n_models))
    well_n = np.zeros(n_wells)
    
    for i in range(n_models):
        well_sum[:, i] = np.bincount(well_indices, weights=L[:, i], minlength=n_wells)
    well_n = np.bincount(well_indices, minlength=n_wells)
    
    # Obs mean loss per model
    obs_mean = well_sum.sum(axis=0) / well_n.sum()
    
    # Draw bootstrap indices once
    boot_idx = rng.integers(0, n_wells, size=(B, n_wells))
    
    # Compute bootstrap means
    # well_sum[boot_idx] is (B, n_wells, n_models)
    # sum over wells -> (B, n_models)
    boot_sum = well_sum[boot_idx].sum(axis=1)
    boot_n_b = well_n[boot_idx].sum(axis=1)
    boot_mean = boot_sum / boot_n_b[:, None]
    
    active_models = list(range(n_models))
    mcs_pvalues = np.ones(n_models)
    elim_order = []
    
    running_max_p = 0.0
    
    while len(active_models) > 1:
        cols = np.array(active_models)
        
        # Relative loss vs mean of active models
        d_rel = obs_mean[cols] - obs_mean[cols].mean()
        
        # Bootstrap relative loss
        b_rel = boot_mean[:, cols] - boot_mean[:, cols].mean(axis=1, keepdims=True)
        
        # Centered bootstrap deviation
        xi = b_rel - d_rel
        
        # Bootstrap variance
        var = (xi**2).mean(axis=0)
        
        # Guard against zero variance
        with np.errstate(divide='ignore', invalid='ignore'):
            t_i = d_rel / np.sqrt(var)
            t_i[var <= 0] = 0.0
            
            T_obs = np.max(t_i)
            
            # T_boot for each bootstrap resample
            t_boot = xi / np.sqrt(var)
            t_boot[:, var <= 0] = 0.0
            T_boot = np.max(t_boot, axis=1)
            
        p = np.mean(T_boot >= T_obs)
        
        # The worst model has the largest t_i
        worst_idx_in_active = np.argmax(t_i)
        worst_model = active_models[worst_idx_in_active]
        
        if p < alpha:
            running_max_p = max(running_max_p, p)
            mcs_pvalues[worst_model] = running_max_p
            elim_order.append(worst_model)
            active_models.remove(worst_model)
        else:
            break
            
    return mcs_pvalues, elim_order, obs_mean

def pairwise_dm_crps(crps_best, crps_other, station_ids):
    """
    Cluster-robust Diebold-Mariano test on CRPS.
    """
    unique_wells, well_indices = np.unique(station_ids, return_inverse=True)
    n_wells = len(unique_wells)
    
    d = crps_other - crps_best
    
    # Per-well mean diff
    well_sum = np.bincount(well_indices, weights=d, minlength=n_wells)
    well_n = np.bincount(well_indices, minlength=n_wells)
    
    valid = well_n > 0
    d_w = well_sum[valid] / well_n[valid]
    
    n = len(d_w)
    if n <= 1:
        return np.nan, np.nan, np.nan
        
    mean_dw = np.mean(d_w)
    std_dw = np.std(d_w, ddof=1)
    
    if std_dw == 0:
        return mean_dw, 0.0, 1.0
        
    t_stat = mean_dw / (std_dw / np.sqrt(n))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    return mean_dw, t_stat, p_val
