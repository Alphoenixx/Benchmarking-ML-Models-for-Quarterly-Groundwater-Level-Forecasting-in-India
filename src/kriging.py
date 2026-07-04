import numpy as np
from scipy.optimize import curve_fit

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def compute_empirical_variogram(lat, lon, values, bins=20, max_dist=1500):
    n = len(lat)
    dists = []
    sq_diffs = []
    
    lats = np.array(lat)
    lons = np.array(lon)
    vals = np.array(values)
    
    for i in range(n):
        d = haversine(lats[i], lons[i], lats[i+1:], lons[i+1:])
        valid = d <= max_dist
        dists.append(d[valid])
        sq_diffs.append((vals[i] - vals[i+1:])[valid]**2)
        
    dists = np.concatenate(dists)
    sq_diffs = np.concatenate(sq_diffs)
    
    bin_edges = np.linspace(0, max_dist, bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    semi_vars = []
    n_pairs = []
    
    for i in range(bins):
        mask = (dists > bin_edges[i]) & (dists <= bin_edges[i+1])
        n_pairs.append(np.sum(mask))
        if n_pairs[-1] > 0:
            semi_vars.append(0.5 * np.mean(sq_diffs[mask]))
        else:
            semi_vars.append(np.nan)
            
    return bin_centers, np.array(semi_vars), np.array(n_pairs)

def exp_variogram(h, nugget, sill, range_):
    return nugget + sill * (1 - np.exp(-h / range_))

def fit_variogram(h, gamma, n_pairs):
    mask = ~np.isnan(gamma) & (n_pairs > 0)
    h_fit = h[mask]
    g_fit = gamma[mask]
    
    if len(h_fit) < 3:
        return [0.0, np.var(g_fit) if len(g_fit)>0 else 1.0, 300.0]
        
    try:
        p0 = [np.min(g_fit), np.max(g_fit) - np.min(g_fit), 300.0]
        popt, _ = curve_fit(exp_variogram, h_fit, g_fit, p0=p0,
                            bounds=([0, 0, 10], [np.inf, np.inf, 1500]),
                            maxfev=2000)
        return popt
    except:
        return [0.0, np.var(g_fit), 300.0]

def ordinary_kriging(target_lat, target_lon, source_lats, source_lons, source_vals, nugget, sill, range_, K=16):
    n_targets = len(target_lat)
    corrections = np.zeros(n_targets)
    n_neighbors = np.zeros(n_targets, dtype=int)
    
    for i in range(n_targets):
        dists = haversine(target_lat[i], target_lon[i], source_lats, source_lons)
        
        valid = dists > 1e-5
        d_valid = dists[valid]
        v_valid = source_vals[valid]
        lat_valid = source_lats[valid]
        lon_valid = source_lons[valid]
        
        if len(d_valid) == 0:
            corrections[i] = 0.0
            continue
            
        idx = np.argsort(d_valid)[:K]
        d_k = d_valid[idx]
        v_k = v_valid[idx]
        lat_k = lat_valid[idx]
        lon_k = lon_valid[idx]
        
        n_neighbors[i] = len(d_k)
        n = len(d_k)
        
        C = np.zeros((n+1, n+1))
        for j in range(n):
            for k in range(n):
                if j == k:
                    C[j, k] = nugget + sill
                else:
                    dist_jk = haversine(lat_k[j], lon_k[j], lat_k[k], lon_k[k])
                    C[j, k] = sill * np.exp(-dist_jk / range_)
                    
        C[:n, n] = 1.0
        C[n, :n] = 1.0
        C[n, n] = 0.0
        
        b = np.zeros(n+1)
        b[:n] = sill * np.exp(-d_k / range_)
        b[n] = 1.0
        
        try:
            w = np.linalg.solve(C, b)
            corrections[i] = np.sum(w[:n] * v_k)
        except np.linalg.LinAlgError:
            corrections[i] = 0.0
            
    return corrections, n_neighbors
