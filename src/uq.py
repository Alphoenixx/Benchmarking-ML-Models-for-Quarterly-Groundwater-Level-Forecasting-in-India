import numpy as np

def get_conformal_radius(residuals, c):
    """
    Finite-sample corrected conformal radius for nominal coverage c.
    r = |y - yhat| or |y - yhat| / scale
    """
    if len(residuals) == 0:
        return np.nan
    n = len(residuals)
    idx = int(np.ceil((n + 1) * c)) - 1
    idx = max(0, min(idx, n - 1))
    r_sorted = np.sort(residuals)
    return r_sorted[idx]

def get_conformal_radius_grid(residuals, c_array):
    """
    Vectorized conformal radius for an array of coverages.
    """
    if len(residuals) == 0:
        return np.full_like(c_array, np.nan)
    n = len(residuals)
    idxs = np.ceil((n + 1) * c_array).astype(int) - 1
    idxs = np.clip(idxs, 0, n - 1)
    r_sorted = np.sort(residuals)
    return r_sorted[idxs]

def compute_picp(y, lower, upper):
    if len(y) == 0: return np.nan
    return np.mean((y >= lower) & (y <= upper))

def compute_mpiw(lower, upper):
    if len(lower) == 0: return np.nan
    return np.mean(upper - lower)

def compute_winkler(y, lower, upper, alpha):
    if len(y) == 0: return np.nan
    width = upper - lower
    penalty_low = (2.0 / alpha) * (lower - y) * (y < lower)
    penalty_high = (2.0 / alpha) * (y - upper) * (y > upper)
    return np.mean(width + penalty_low + penalty_high)

def compute_pinball(y, q_val, tau):
    return np.where(y >= q_val, (y - q_val) * tau, (q_val - y) * (1 - tau))

def compute_crps_per_obs(y, yhat, residuals, tau_grid):
    if len(residuals) == 0 or len(y) == 0:
        return np.array([])
    
    c_vals = np.where(tau_grid >= 0.5, 2*tau_grid - 1, 1 - 2*tau_grid)
    q_vals = np.zeros_like(c_vals)
    mask = c_vals > 0
    if np.any(mask):
        q_vals[mask] = get_conformal_radius_grid(residuals, c_vals[mask])
    
    yhat_exp = yhat[:, None]
    q_vals_exp = q_vals[None, :]
    tau_grid_exp = tau_grid[None, :]
    
    Q_tau = np.where(tau_grid_exp >= 0.5, yhat_exp + q_vals_exp, yhat_exp - q_vals_exp)
    y_exp = y[:, None]
    
    loss = compute_pinball(y_exp, Q_tau, tau_grid_exp)
    crps_i = 2 * np.mean(loss, axis=1)
    return crps_i

def compute_crps_from_quantiles(y, Q, tau_grid):
    y_exp = y[:, None]
    tau_grid_exp = tau_grid[None, :]
    loss = compute_pinball(y_exp, Q, tau_grid_exp)
    return 2 * np.mean(loss, axis=1)


def compute_crps_and_pinball(y, yhat, residuals, tau_grid):
    """
    Build predictive quantile function from MARGINAL conformal radii.
    tau_grid = 0.01..0.99
    For tau>=0.5: c = 2*tau-1 => Q(tau) = yhat + q(c)
    For tau<0.5: c = 1-2*tau => Q(tau) = yhat - q(c)
    q(0) = 0
    """
    if len(residuals) == 0 or len(y) == 0:
        return np.nan, np.nan
    
    c_vals = np.where(tau_grid >= 0.5, 2*tau_grid - 1, 1 - 2*tau_grid)
    q_vals = np.zeros_like(c_vals)
    mask = c_vals > 0
    if np.any(mask):
        q_vals[mask] = get_conformal_radius_grid(residuals, c_vals[mask])
    
    yhat_exp = yhat[:, None]
    q_vals_exp = q_vals[None, :]
    tau_grid_exp = tau_grid[None, :]
    
    Q_tau = np.where(tau_grid_exp >= 0.5, yhat_exp + q_vals_exp, yhat_exp - q_vals_exp)
    y_exp = y[:, None]
    
    loss = compute_pinball(y_exp, Q_tau, tau_grid_exp)
    mean_pinball_per_tau = np.mean(loss, axis=0)
    overall_mean_pinball = np.mean(mean_pinball_per_tau)
    
    crps_i = 2 * np.mean(loss, axis=1)
    crps = np.mean(crps_i)
    
    return crps, overall_mean_pinball

def exceedance_prob(yhat, residuals, tau_well, tau_grid):
    """
    Calculates P(Y > tau_well) = 1 - F(tau_well).
    """
    c_vals = np.where(tau_grid >= 0.5, 2*tau_grid - 1, 1 - 2*tau_grid)
    q_vals = np.zeros_like(c_vals)
    mask = c_vals > 0
    if np.any(mask):
        q_vals[mask] = get_conformal_radius_grid(residuals, c_vals[mask])
        
    q_eps = np.where(tau_grid >= 0.5, q_vals, -q_vals)
    sort_idx = np.argsort(q_eps)
    q_eps_sorted = q_eps[sort_idx]
    p_sorted = tau_grid[sort_idx]
    
    # Ensure strict monotonicity for np.interp
    q_eps_sorted = q_eps_sorted + np.linspace(0, 1e-7, len(q_eps_sorted))
    
    q_min = q_eps_sorted[0] - 1.0
    q_max = q_eps_sorted[-1] + 1.0
    
    q_eps_ext = np.concatenate([[-np.inf, q_min], q_eps_sorted, [q_max, np.inf]])
    p_ext = np.concatenate([[0.0, 0.0], p_sorted, [1.0, 1.0]])
    
    E = tau_well - yhat
    F_E = np.interp(E, q_eps_ext, p_ext)
    
    p_hat = 1.0 - F_E
    return np.clip(p_hat, 0.0, 1.0)

def get_pit(y, yhat, residuals, tau_grid):
    """
    PIT_i = the tau in the grid s.t. Q_i(tau) is closest to y_i.
    """
    if len(residuals) == 0 or len(y) == 0:
        return np.array([])
    c_vals = np.where(tau_grid >= 0.5, 2*tau_grid - 1, 1 - 2*tau_grid)
    q_vals = np.zeros_like(c_vals)
    mask = c_vals > 0
    if np.any(mask):
        q_vals[mask] = get_conformal_radius_grid(residuals, c_vals[mask])
    
    yhat_exp = yhat[:, None]
    q_vals_exp = q_vals[None, :]
    tau_grid_exp = tau_grid[None, :]
    
    Q_tau = np.where(tau_grid_exp >= 0.5, yhat_exp + q_vals_exp, yhat_exp - q_vals_exp)
    y_exp = y[:, None]
    
    # distance to y
    dist = np.abs(Q_tau - y_exp)
    min_idx = np.argmin(dist, axis=1)
    return tau_grid[min_idx]
