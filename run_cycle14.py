import json, time, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import ROOT
from src.logging_utils import get_logger
import src.uq as uq

warnings.filterwarnings("ignore")

TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"
LOG_DIR = ROOT / "outputs" / "logs"

def main():
    t0 = time.time()
    log = get_logger("cycle14")
    
    log.info("="*50)
    log.info("CYCLE 14: PROBABILISTIC UQ CORE")
    log.info("="*50)
    log.info(f"Libraries: numpy={np.__version__}, pandas={pd.__version__}, matplotlib={matplotlib.__version__}")
    
    sum_dict = {"metadata": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__}}
    
    # Load all_preds_long
    preds_file = TAB / "all_preds_long.parquet"
    log.info(f"Loading predictions from {preds_file}")
    df = pd.read_parquet(preds_file)
    log.info(f"Predictions loaded: {df.shape}")
    
    # Load quarterly_panel to get lat, lon
    panel_file = ROOT / "data" / "processed" / "quarterly_panel.parquet"
    log.info(f"Loading panel from {panel_file}")
    panel = pd.read_parquet(panel_file)
    
    # Map lat, lon, is_igp
    st_meta = panel.groupby("station_id")[["lat", "lon"]].first().reset_index()
    st_meta["is_igp"] = st_meta["lat"].between(27.5, 32.5) & st_meta["lon"].between(73.5, 81.0)
    
    df = df.merge(st_meta, on="station_id", how="left")
    
    # History tercile (train pre-2019)
    # 'period' is something like '2019-01-01'. 
    # Or just use the 'split' column? The prompt says "pre-2019 (train-split) observations".
    # Since panel has year, let's use it.
    train_counts = panel[panel["year"] <= 2018].groupby("station_id").size()
    q33, q67 = train_counts.quantile([0.3333, 0.6667])
    def get_tercile(c):
        if c <= q33: return "short"
        elif c <= q67: return "medium"
        else: return "long"
    
    train_counts_df = train_counts.reset_index().rename(columns={0: "train_obs"})
    train_counts_df["tercile"] = train_counts_df["train_obs"].apply(get_tercile)
    df = df.merge(train_counts_df[["station_id", "tercile"]], on="station_id", how="left")
    # For wells with no train obs? Should be rare or they wouldn't be in the cohort.
    df["tercile"] = df["tercile"].fillna("short")
    
    sum_dict["tercile_cutoffs"] = {"q33": float(q33), "q67": float(q67)}
    log.info(f"History tercile cutoffs: q33={q33}, q67={q67}")
    
    # Handle scale
    # Floor to global-median positive scale
    pos_scale_median = df.loc[df["scale"] > 0, "scale"].median()
    bad_scale_mask = df["scale"].isna() | (df["scale"] <= 0)
    n_bad = int(bad_scale_mask.sum())
    df.loc[bad_scale_mask, "scale"] = pos_scale_median
    log.info(f"Floored {n_bad} rows with non-positive/NaN scale to {pos_scale_median:.4f}")
    sum_dict["scale_floored_rows"] = n_bad
    sum_dict["global_median_positive_scale"] = float(pos_scale_median)
    
    models = sorted(df["model"].unique())
    horizons = sorted(df["h"].unique())
    C = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    tau_grid = np.arange(0.01, 1.0, 0.01)
    
    log.info(f"Models: {models}")
    log.info(f"Horizons: {horizons}")
    
    results_nat = []
    results_igp = []
    rel_nat_h1 = []
    pit_nat_h1 = []
    mondrian_reg = []
    mondrian_hist = []
    
    cell_stats = {}
    
    # We will need climatology CRPS for CRPSS.
    # What is the climatology prediction? 
    # Usually, climatology prediction is just a naive model or training mean. 
    # The prompt says "climatology (same cell)". The simplest climatology for forecasting is the training mean or seasonal naive.
    # Wait, climatology usually means yhat = training mean. 
    # Wait, is 'seasonalnaive' or 'climatology' one of the models in the 8 benchmarked models?
    # Yes, 'climatology' is usually in the models. Let's see if 'climatology' is in models.
    # If not, wait, the prompt says "CRPS_skill CRPSS = 1 - CRPS_model / CRPS_climatology".
    # I should find CRPS of the 'climatology' model for the same horizon.
    
    # First, let's pre-calculate all CRPS so we can lookup CRPS_climatology.
    # Since we iterate over models, we might not have climatology yet.
    # We will do a two-pass for CRPSS or just compute all CRPS first.
    
    # Let's iterate
    pbar = tqdm(total=len(models) * len(horizons), file=sys.stdout)
    
    for m in models:
        for h in horizons:
            cell_df = df[(df["model"] == m) & (df["h"] == h)]
            
            cal_df = cell_df[cell_df["split"] == "val"]
            eval_df = cell_df[cell_df["split"] == "test"]
            
            n_cal = len(cal_df)
            n_eval = len(eval_df)
            cell_key = f"{m}_h{h}"
            cell_stats[cell_key] = {"n_cal": n_cal, "n_eval": n_eval}
            
            if n_cal == 0 or n_eval == 0:
                raise ValueError(f"Zero rows in {m} h{h}: cal={n_cal}, eval={n_eval}")
            
            cal_y = cal_df["y"].values
            cal_yhat = cal_df["yhat"].values
            cal_scale = cal_df["scale"].values
            
            eval_y = eval_df["y"].values
            eval_yhat = eval_df["yhat"].values
            eval_scale = eval_df["scale"].values
            eval_igp = eval_df["is_igp"].values
            eval_tercile = eval_df["tercile"].values
            
            r_cal = np.abs(cal_y - cal_yhat)
            s_cal = r_cal / cal_scale
            
            # 1) SPLIT-CONFORMAL
            # Marginal radii
            q_c = {c: uq.get_conformal_radius(r_cal, c) for c in C}
            qn_c = {c: uq.get_conformal_radius(s_cal, c) for c in C}
            
            # Mondrian grouping A (region)
            r_cal_igp = np.abs(cal_df[cal_df["is_igp"]]["y"] - cal_df[cal_df["is_igp"]]["yhat"])
            r_cal_non = np.abs(cal_df[~cal_df["is_igp"]]["y"] - cal_df[~cal_df["is_igp"]]["yhat"])
            q_igp_90 = uq.get_conformal_radius(r_cal_igp.values, 0.90) if len(r_cal_igp) > 0 else q_c[0.90]
            q_non_90 = uq.get_conformal_radius(r_cal_non.values, 0.90) if len(r_cal_non) > 0 else q_c[0.90]
            
            # Mondrian grouping B (history)
            r_cal_short = np.abs(cal_df[cal_df["tercile"]=="short"]["y"] - cal_df[cal_df["tercile"]=="short"]["yhat"])
            r_cal_med = np.abs(cal_df[cal_df["tercile"]=="medium"]["y"] - cal_df[cal_df["tercile"]=="medium"]["yhat"])
            r_cal_long = np.abs(cal_df[cal_df["tercile"]=="long"]["y"] - cal_df[cal_df["tercile"]=="long"]["yhat"])
            q_short_90 = uq.get_conformal_radius(r_cal_short.values, 0.90) if len(r_cal_short) > 0 else q_c[0.90]
            q_med_90 = uq.get_conformal_radius(r_cal_med.values, 0.90) if len(r_cal_med) > 0 else q_c[0.90]
            q_long_90 = uq.get_conformal_radius(r_cal_long.values, 0.90) if len(r_cal_long) > 0 else q_c[0.90]
            
            # Eval for both cohorts
            for is_igp_cohort in [False, True]:
                if is_igp_cohort:
                    mask = eval_igp
                else:
                    mask = np.ones_like(eval_igp, dtype=bool)
                
                if np.sum(mask) == 0:
                    continue
                
                y = eval_y[mask]
                yhat = eval_yhat[mask]
                scale = eval_scale[mask]
                tercile = eval_tercile[mask]
                igp_mask_subset = eval_igp[mask]
                
                # Marginal
                lower_90 = yhat - q_c[0.90]
                upper_90 = yhat + q_c[0.90]
                picp90 = uq.compute_picp(y, lower_90, upper_90)
                mpiw90 = uq.compute_mpiw(lower_90, upper_90)
                ace90 = picp90 - 0.90
                winkler90 = uq.compute_winkler(y, lower_90, upper_90, 0.10)
                
                picp50 = uq.compute_picp(y, yhat - q_c[0.50], yhat + q_c[0.50])
                picp95 = uq.compute_picp(y, yhat - q_c[0.95], yhat + q_c[0.95])
                
                # Normalized
                lower_norm_90 = yhat - qn_c[0.90] * scale
                upper_norm_90 = yhat + qn_c[0.90] * scale
                picp90_norm = uq.compute_picp(y, lower_norm_90, upper_norm_90)
                mpiw90_norm = uq.compute_mpiw(lower_norm_90, upper_norm_90)
                
                # CRPS
                crps, pinball = uq.compute_crps_and_pinball(y, yhat, r_cal, tau_grid)
                
                row = {
                    "model": m, "h": h,
                    "picp50": picp50, "picp90": picp90, "picp95": picp95,
                    "mpiw90": mpiw90, "ace90": ace90,
                    "picp90_norm": picp90_norm, "mpiw90_norm": mpiw90_norm,
                    "winkler90": winkler90, "crps": crps, "pinball": pinball
                }
                
                if is_igp_cohort:
                    results_igp.append(row)
                else:
                    results_nat.append(row)
            
            # Mondrian reporting (national only)
            if True:
                y = eval_y
                yhat = eval_yhat
                igp_flag = eval_igp
                terc_flag = eval_tercile
                
                for r_name in ["igp", "non_igp"]:
                    r_m = igp_flag if r_name == "igp" else ~igp_flag
                    if np.sum(r_m) > 0:
                        y_r = y[r_m]; yhat_r = yhat[r_m]
                        q_r = q_igp_90 if r_name == "igp" else q_non_90
                        p = uq.compute_picp(y_r, yhat_r - q_r, yhat_r + q_r)
                        w = uq.compute_mpiw(yhat_r - q_r, yhat_r + q_r)
                        mondrian_reg.append({"model": m, "h": h, "region": r_name, "picp90": p, "mpiw90": w, "ace90": p - 0.90})
                        
                for t_name in ["short", "medium", "long"]:
                    t_m = terc_flag == t_name
                    if np.sum(t_m) > 0:
                        y_t = y[t_m]; yhat_t = yhat[t_m]
                        if t_name == "short": q_t = q_short_90
                        elif t_name == "medium": q_t = q_med_90
                        else: q_t = q_long_90
                        p = uq.compute_picp(y_t, yhat_t - q_t, yhat_t + q_t)
                        w = uq.compute_mpiw(yhat_t - q_t, yhat_t + q_t)
                        mondrian_hist.append({"model": m, "h": h, "tercile": t_name, "picp90": p, "mpiw90": w, "ace90": p - 0.90})

            # Reliability & PIT (national h=1)
            if h == 1:
                # Reliability
                for c in C:
                    p_mar = uq.compute_picp(eval_y, eval_yhat - q_c[c], eval_yhat + q_c[c])
                    p_nor = uq.compute_picp(eval_y, eval_yhat - qn_c[c]*eval_scale, eval_yhat + qn_c[c]*eval_scale)
                    rel_nat_h1.append({"model": m, "nominal_c": c, "empirical_picp": p_mar, "type": "marginal"})
                    rel_nat_h1.append({"model": m, "nominal_c": c, "empirical_picp": p_nor, "type": "normalized"})
                    
                # PIT
                pit_vals = uq.get_pit(eval_y, eval_yhat, r_cal, tau_grid)
                counts, edges = np.histogram(pit_vals, bins=10, range=(0,1))
                density = counts / len(pit_vals)
                expected = 1.0 / 10
                unif_dev = np.sum((density - expected)**2 / expected)
                for i in range(10):
                    pit_nat_h1.append({"model": m, "bin_left": edges[i], "bin_right": edges[i+1], "density": density[i], "uniformity_dev": unif_dev})

            pbar.update(1)
            
    pbar.close()

    # CRPSS
    df_nat = pd.DataFrame(results_nat)
    df_igp = pd.DataFrame(results_igp)
    
    # find climatology model
    clim_name = "climatology"
    if clim_name not in df_nat["model"].values:
        clim_name = "seasonalnaive" # fallback to seasonalnaive if it exists
        if clim_name not in df_nat["model"].values:
            log.warning("No climatology or seasonalnaive model found. Setting crps_clim=NaN, crpss=NaN")
            clim_name = None
    
    for _df in [df_nat, df_igp]:
        _df["crps_clim"] = np.nan
        _df["crpss"] = np.nan
        if clim_name is not None:
            for h in horizons:
                mask_h = _df["h"] == h
                clim_crps = _df.loc[mask_h & (_df["model"] == clim_name), "crps"]
                if len(clim_crps) > 0:
                    c_val = clim_crps.values[0]
                    _df.loc[mask_h, "crps_clim"] = c_val
                    _df.loc[mask_h, "crpss"] = 1 - _df.loc[mask_h, "crps"] / c_val
                
    # Checks
    for _df, name in [(df_nat, "national"), (df_igp, "igp")]:
        has_nan = _df.isna().sum().sum()
        if has_nan > 0:
            nan_cols = _df.columns[_df.isna().any()].tolist()
            # Ignore crps_clim/crpss if clim_name is missing
            if clim_name is None and all(c in ["crps_clim", "crpss"] for c in nan_cols):
                pass
            else:
                offending = _df[_df.isna().any(axis=1)][["model", "h"]]
                raise ValueError(f"NaNs found in table {name}. Columns: {nan_cols}. Offending:\n{offending}")
            
        assert (_df["mpiw90"] > 0).all(), f"MPIW90 strictly positive failed in {name}"
        assert (_df["picp90"].between(0, 1)).all(), f"PICP90 outside [0,1] in {name}"
        if clim_name is not None:
            clim_rows = _df[_df["model"] == clim_name]
            assert np.allclose(clim_rows["crpss"], 0, atol=1e-5), f"CRPSS != 0 for climatology in {name}"

    log.info("Acceptance checks passed.")
    sum_dict["acceptance_checks_passed"] = True

    # Output CSVs
    df_nat.to_csv(TAB / "uq_coverage_scores_national.csv", index=False)
    df_igp.to_csv(TAB / "uq_coverage_scores_igp.csv", index=False)
    
    df_rel = pd.DataFrame(rel_nat_h1)
    df_rel.to_csv(TAB / "uq_reliability_national_h1.csv", index=False)
    
    df_pit = pd.DataFrame(pit_nat_h1)
    df_pit.to_csv(TAB / "uq_pit_national_h1.csv", index=False)
    
    df_m_reg = pd.DataFrame(mondrian_reg)
    df_m_reg.to_csv(TAB / "uq_mondrian_region.csv", index=False)
    
    df_m_his = pd.DataFrame(mondrian_hist)
    df_m_his.to_csv(TAB / "uq_mondrian_history_tercile.csv", index=False)
    
    # Plotting
    log.info("Generating figures...")
    
    # 1. Reliability curve h1
    plt.figure(figsize=(6,6))
    plt.plot([0,1], [0,1], "k--", label="Ideal")
    df_rel_mar = df_rel[df_rel["type"] == "marginal"]
    for m in models:
        dm = df_rel_mar[df_rel_mar["model"] == m]
        plt.plot(dm["nominal_c"], dm["empirical_picp"], marker="o", label=m)
    plt.xlabel("Nominal Coverage")
    plt.ylabel("Empirical Coverage")
    plt.title("Reliability Curve (h=1, Marginal)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(FIG / "reliability_curve_h1.png", dpi=200)
    plt.close()
    
    # 2. PIT hist h1
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, m in enumerate(models):
        if i >= len(axes): break
        dm = df_pit[df_pit["model"] == m]
        x = (dm["bin_left"] + dm["bin_right"]) / 2
        w = dm["bin_right"] - dm["bin_left"]
        axes[i].bar(x, dm["density"], width=w, edgecolor="k", align="center")
        axes[i].axhline(0.1, color="r", linestyle="--")
        axes[i].set_title(f"{m}\nUnif Dev: {dm['uniformity_dev'].iloc[0]:.4f}")
        axes[i].set_ylim(0, 0.3)
    plt.tight_layout()
    plt.savefig(FIG / "pit_hist_h1.png", dpi=200)
    plt.close()
    
    # 3. Interval width vs h
    plt.figure(figsize=(8,5))
    for m in models:
        dm = df_nat[df_nat["model"] == m]
        plt.plot(dm["h"], dm["mpiw90"], marker="o", label=m)
    plt.xlabel("Horizon (h)")
    plt.ylabel("MPIW @ 90%")
    plt.title("Interval Width vs Horizon")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(FIG / "interval_width_vs_horizon.png", dpi=200)
    plt.close()
    
    # 4. CRPSS vs h
    plt.figure(figsize=(8,5))
    for m in models:
        dm = df_nat[df_nat["model"] == m]
        plt.plot(dm["h"], dm["crpss"], marker="o", label=m)
    plt.xlabel("Horizon (h)")
    plt.ylabel("CRPS Skill Score")
    plt.title("CRPSS vs Horizon")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(FIG / "crps_skill_vs_horizon.png", dpi=200)
    plt.close()
    
    # Console summary
    log.info("\n" + "="*50 + "\nCONSOLE SUMMARY")
    log.info(f"Elapsed: {time.time()-t0:.1f}s")
    for k, v in cell_stats.items():
        log.info(f"Cell {k}: cal={v['n_cal']}, eval={v['n_eval']}")
        
    log.info("\nNational Scores:")
    log.info(df_nat[["model", "h", "picp90", "mpiw90", "ace90", "crps", "crpss", "winkler90"]].to_string())
    
    log.info("\nIGP Scores:")
    log.info(df_igp[["model", "h", "picp90", "mpiw90", "ace90", "crps", "crpss", "winkler90"]].to_string())
    
    log.info("\nMarginal vs Normalized vs Mondrian PICP90 at h=1:")
    for m in models:
        p_mar = df_nat[(df_nat["model"]==m)&(df_nat["h"]==1)]["picp90"].values[0]
        p_nor = df_nat[(df_nat["model"]==m)&(df_nat["h"]==1)]["picp90_norm"].values[0]
        m_igp = df_m_reg[(df_m_reg["model"]==m)&(df_m_reg["h"]==1)&(df_m_reg["region"]=="igp")]["picp90"].values[0]
        m_non = df_m_reg[(df_m_reg["model"]==m)&(df_m_reg["h"]==1)&(df_m_reg["region"]=="non_igp")]["picp90"].values[0]
        log.info(f"  {m}: Mar={p_mar:.4f}, Nor={p_nor:.4f}, MonIGP={m_igp:.4f}, MonNon={m_non:.4f}")
        
    log.info("\nReliability max |emp - nom| per model (h=1, marginal):")
    for m in models:
        dm = df_rel_mar[df_rel_mar["model"] == m]
        max_err = np.max(np.abs(dm["empirical_picp"] - dm["nominal_c"]))
        log.info(f"  {m}: {max_err:.4f}")
        
    log.info("\nPIT Uniformity Dev per model (h=1):")
    for m in models:
        dm = df_pit[df_pit["model"] == m]
        dev = dm["uniformity_dev"].iloc[0]
        log.info(f"  {m}: {dev:.4f}")
        
    # Write cycle summary JSON
    with open(REP / "cycle14_summary.json", "w") as f:
        json.dump(sum_dict, f, indent=2)

    log.info(f"DONE in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
