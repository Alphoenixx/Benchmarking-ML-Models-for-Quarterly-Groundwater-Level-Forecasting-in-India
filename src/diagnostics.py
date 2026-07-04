import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.config import PROC, FIG, TAB, REP
from src.logging_utils import get_logger, Timer

# IGP bounding box (Punjab, Haryana, Delhi, Chandigarh, western UP)
IGP = {"lat_min": 27.5, "lat_max": 32.5, "lon_min": 73.5, "lon_max": 81.0}

# Approx reference centroids for best-effort decoding of State_encoded
STATE_REF = {
    "Punjab": (31.0, 75.5), "Haryana": (29.2, 76.3), "Delhi": (28.6, 77.2),
    "Chandigarh": (30.7, 76.8), "Uttar Pradesh": (27.0, 80.5),
    "Rajasthan": (26.6, 73.8), "Gujarat": (22.5, 71.5), "Telangana": (17.9, 79.6),
    "Andhra Pradesh": (15.5, 79.5), "Karnataka": (15.0, 76.0), "Tamil Nadu": (11.0, 78.5),
    "Maharashtra": (19.5, 76.0), "Madhya Pradesh": (23.5, 78.5), "Bihar": (25.8, 85.5),
    "West Bengal": (23.5, 87.5), "Odisha": (20.5, 84.5),
}

def _nearest_state(lat, lon):
    best, bd = None, 1e9
    for name, (rlat, rlon) in STATE_REF.items():
        d = (lat - rlat) ** 2 + (lon - rlon) ** 2
        if d < bd:
            bd, best = d, name
    return best

def run(df, log):
    out = {}
    # --- exact column names (repr catches hidden chars) ---
    log.info(f"EXACT COLUMNS: {[repr(c) for c in df.columns]}")
    out["columns"] = [str(c) for c in df.columns]

    # --- parse dates ---
    with Timer(log, "parse datetime"):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        n_bad = int(df["datetime"].isna().sum())
        log.info(f"unparseable dates: {n_bad}")
        out["unparseable_dates"] = n_bad
        df = df.dropna(subset=["datetime"])

    # --- target sanity (sign/units) ---
    t = df["target"]
    out["target_stats"] = {k: float(v) for k, v in
        {"min": t.min(), "p05": t.quantile(.05), "median": t.median(),
         "mean": t.mean(), "p95": t.quantile(.95), "max": t.max()}.items()}
    log.info(f"target_stats: {out['target_stats']}")

    # --- best-effort state decoding via centroids ---
    with Timer(log, "decode states"):
        cen = df.groupby("State_encoded")[["latitude", "longitude"]].mean()
        cnt = df.groupby("State_encoded").size()
        decode = {}
        for se, row in cen.iterrows():
            decode[int(se)] = {"guess": _nearest_state(row["latitude"], row["longitude"]),
                               "centroid": [round(row["latitude"],3), round(row["longitude"],3)],
                               "rows": int(cnt.loc[se])}
        out["state_decode"] = decode
        log.info(f"state_decode: {json.dumps(decode, indent=2)}")

    # --- region subset by bbox ---
    reg = df[(df.latitude.between(IGP["lat_min"], IGP["lat_max"])) &
             (df.longitude.between(IGP["lon_min"], IGP["lon_max"]))].copy()
    out["region_bbox"] = IGP
    out["region_rows"] = int(len(reg))
    out["region_wells"] = int(reg["station_id"].nunique())
    log.info(f"REGION: {out['region_rows']} rows, {out['region_wells']} wells in IGP bbox")

    # --- per-well profiling (whole dataset AND region) ---
    def profile(frame, label):
        g = frame.sort_values("datetime").groupby("station_id")
        prof = g["datetime"].agg(["count", "min", "max"])
        prof["span_days"] = (prof["max"] - prof["min"]).dt.days
        # median sampling gap per well
        med_gap = g["datetime"].apply(lambda s: s.diff().dt.days.median())
        prof["median_gap_days"] = med_gap
        prof.to_csv(TAB / f"well_profile_{label}.csv")
        log.info(f"[{label}] wells={len(prof)} | "
                 f"count: median={prof['count'].median():.0f} "
                 f"p90={prof['count'].quantile(.9):.0f} max={prof['count'].max():.0f} | "
                 f"span_days: median={prof['span_days'].median():.0f} "
                 f"max={prof['span_days'].max():.0f} | "
                 f"median_gap: median={prof['median_gap_days'].median():.1f}")
        return prof

    with Timer(log, "profile all"):
        prof_all = profile(df, "all")
    with Timer(log, "profile region"):
        prof_reg = profile(reg, "region")

    # --- cohort grid: how many region wells survive thresholds ---
    grid = {}
    for min_rows in (180, 365, 730, 1095):
        for min_span in (365, 730, 1095):
            m = ((prof_reg["count"] >= min_rows) & (prof_reg["span_days"] >= min_span)).sum()
            grid[f"rows>={min_rows}&span>={min_span}d"] = int(m)
    out["region_cohort_grid"] = grid
    log.info(f"region_cohort_grid: {json.dumps(grid, indent=2)}")

    # --- monthly resample feasibility (region) ---
    with Timer(log, "monthly feasibility"):
        monthly_counts = (reg.set_index("datetime")
                             .groupby("station_id")["target"]
                             .resample("MS").mean().groupby("station_id").count())
        out["region_monthly_points"] = {
            "median": float(monthly_counts.median()),
            "p90": float(monthly_counts.quantile(.9)),
            "max": int(monthly_counts.max()),
            "wells_ge_24mo": int((monthly_counts >= 24).sum()),
            "wells_ge_36mo": int((monthly_counts >= 36).sum()),
        }
        log.info(f"region_monthly_points: {out['region_monthly_points']}")

    # --- figures ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for col, name in [("count", "records_per_well"), ("span_days", "span_days_per_well")]:
        plt.figure(figsize=(7,4))
        prof_reg[col].plot(kind="hist", bins=50)
        plt.title(f"IGP region: {name}"); plt.xlabel(col); plt.tight_layout()
        plt.savefig(FIG / f"region_{name}.png", dpi=150); plt.close()

    # region map colored by mean target per well
    wmean = reg.groupby("station_id").agg(lat=("latitude","mean"),
                                          lon=("longitude","mean"),
                                          mt=("target","mean"))
    plt.figure(figsize=(6,6))
    sc = plt.scatter(wmean["lon"], wmean["lat"], c=wmean["mt"], s=8, cmap="viridis")
    plt.colorbar(sc, label="mean target (m)")
    plt.title("IGP wells (color = mean groundwater level)")
    plt.xlabel("longitude"); plt.ylabel("latitude"); plt.tight_layout()
    plt.savefig(FIG / "region_well_map.png", dpi=150); plt.close()

    # --- regional depletion slope (yearly mean target vs year) ---
    reg["year"] = reg["datetime"].dt.year
    ym = reg.groupby("year")["target"].mean()
    slope = float(np.polyfit(ym.index.values, ym.values, 1)[0])
    out["region_depletion_slope_m_per_yr"] = slope
    log.info(f"region depletion slope = {slope:.4f} m/yr")
    ym.to_csv(TAB / "region_yearly_mean_level.csv")

    return out
