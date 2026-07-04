import time, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import t

from src.config import ROOT
from src.logging_utils import get_logger
import src.uq as uq
import src.mcs as mcs

warnings.filterwarnings("ignore")

TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"

def main():
    t0 = time.time()
    log = get_logger("cycle15")
    
    log.info("="*50)
    log.info("CYCLE 15: MCS & CLUSTER-ROBUST SIGNIFICANCE")
    log.info("="*50)
    log.info(f"Libraries: numpy={np.__version__}, pandas={pd.__version__}, matplotlib={matplotlib.__version__}")
    
    # Load all_preds_long
    df = pd.read_parquet(TAB / "all_preds_long.parquet")
    panel = pd.read_parquet(ROOT / "data" / "processed" / "quarterly_panel.parquet")
    st_meta = panel.groupby("station_id")[["lat", "lon"]].first().reset_index()
    st_meta["is_igp"] = st_meta["lat"].between(27.5, 32.5) & st_meta["lon"].between(73.5, 81.0)
    df = df.merge(st_meta, on="station_id", how="left")
    
    # Cycle 14 outputs to cross check
    nat_uq = pd.read_csv(TAB / "uq_coverage_scores_national.csv")
    
    models = sorted(df["model"].unique())
    n_models = len(models)
    horizons = sorted(df["h"].unique())
    tau_grid = np.arange(0.01, 1.0, 0.01)
    
    cal_residuals = {}
    for m in models:
        for h in horizons:
            cal_mask = (df["model"] == m) & (df["h"] == h) & (df["split"] == "val")
            r = np.abs(df.loc[cal_mask, "y"] - df.loc[cal_mask, "yhat"]).values
            cal_residuals[(m, h)] = r
            
    mcs_res_nat = []
    mcs_res_igp = []
    dm_res_nat = []
    dm_res_igp = []
    
    sum_dict = {
        "metadata": {
            "B": 2000,
            "alpha": 0.10,
            "seed": 42,
            "numpy": np.__version__,
            "pandas": pd.__version__
        },
        "mcs_sets": {}
    }
    
    for cohort in ["national", "igp"]:
        for h in horizons:
            # Common set evaluation
            eval_df = df[(df["h"] == h) & (df["split"] == "test")]
            if cohort == "igp":
                eval_df = eval_df[eval_df["is_igp"]]
                
            # Pivot to align models
            pivot = eval_df.pivot(index=["station_id", "period"], columns="model", values=["y", "yhat"])
            pivot = pivot.dropna()
            if len(pivot) == 0:
                raise ValueError(f"Zero common rows for {cohort} h{h}")
            
            common_keys = pivot.index
            n_obs = len(common_keys)
            st_ids = common_keys.get_level_values(0).values
            n_wells = len(np.unique(st_ids))
            
            log.info(f"[{cohort} h{h}] common-set: rows={n_obs}, wells={n_wells}")
            
            # Extract aligned Y and Yhat
            Y_aligned = pivot["y"].iloc[:, 0].values
            Yhat_aligned = pivot["yhat"][models].values
            
            # Precompute SE and CRPS
            L_se = np.zeros((n_obs, n_models))
            L_crps = np.zeros((n_obs, n_models))
            
            for i, m in enumerate(models):
                yhat_m = Yhat_aligned[:, i]
                L_se[:, i] = (Y_aligned - yhat_m)**2
                
                # CRPS
                r_cal = cal_residuals[(m, h)]
                crps_i = uq.compute_crps_per_obs(Y_aligned, yhat_m, r_cal, tau_grid)
                L_crps[:, i] = crps_i
                
                # Check CRPS with Cycle 14
                if cohort == "national":
                    mean_crps = np.mean(crps_i)
                    ref_crps = nat_uq.loc[(nat_uq["model"]==m)&(nat_uq["h"]==h), "crps"].values[0]
                    if not np.isclose(mean_crps, ref_crps, atol=1e-6):
                        raise ValueError(f"CRPS mismatch for {m} h{h}: calc={mean_crps}, ref={ref_crps}")
            
            # MCS Runs
            for loss_type, L in [("se", L_se), ("crps", L_crps)]:
                # MCS
                mcs_p, elim, obs_mean = mcs.mcs_procedure(L, st_ids, alpha=0.10, B=2000, seed=42)
                
                best_idx = np.argmin(obs_mean)
                best_model = models[best_idx]
                
                assert mcs_p[best_idx] >= 0.10, f"Best model {best_model} eliminated!"
                assert len(elim) + np.sum(mcs_p >= 0.10) == n_models
                
                mcs_set = [models[i] for i in range(n_models) if mcs_p[i] >= 0.10]
                
                res_list = mcs_res_nat if cohort == "national" else mcs_res_igp
                
                for i, m in enumerate(models):
                    order = elim.index(i) + 1 if i in elim else n_models
                    res_list.append({
                        "h": h, "loss_type": loss_type, "model": m,
                        "avg_loss": obs_mean[i], "elim_order": order,
                        "mcs_pvalue": mcs_p[i], "in_mcs": mcs_p[i] >= 0.10
                    })
                
                key = f"{cohort}_h{h}_{loss_type}"
                sum_dict["mcs_sets"][key] = {
                    "alpha": 0.10,
                    "set": mcs_set,
                    "size": len(mcs_set),
                    "best_model": best_model,
                    "elim_order": [models[i] for i in elim]
                }
                
                # Pairwise DM for CRPS only
                if loss_type == "crps":
                    dm_list = dm_res_nat if cohort == "national" else dm_res_igp
                    for i, m in enumerate(models):
                        if i == best_idx: continue
                        mean_diff, t_stat, p_val = mcs.pairwise_dm_crps(L[:, best_idx], L[:, i], st_ids)
                        dm_list.append({
                            "h": h, "best_model": best_model, "model": m,
                            "mean_crps_diff": mean_diff, "t_stat": t_stat,
                            "p_value": p_val, "n_wells": n_wells
                        })
                        
    # Save tables
    pd.DataFrame(mcs_res_nat).to_csv(TAB / "mcs_results_national.csv", index=False)
    pd.DataFrame(mcs_res_igp).to_csv(TAB / "mcs_results_igp.csv", index=False)
    pd.DataFrame(dm_res_nat).to_csv(TAB / "dm_crps_pairwise_national.csv", index=False)
    pd.DataFrame(dm_res_igp).to_csv(TAB / "dm_crps_pairwise_igp.csv", index=False)
    
    # Generate Heatmaps
    for loss_type, fname in [("crps", "mcs_membership_crps.png"), ("se", "mcs_membership_se.png")]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for j, cohort in enumerate(["national", "igp"]):
            ax = axes[j]
            res = mcs_res_nat if cohort == "national" else mcs_res_igp
            df_loss = pd.DataFrame([r for r in res if r["loss_type"] == loss_type])
            
            piv_in = df_loss.pivot(index="model", columns="h", values="in_mcs").astype(int)
            piv_p = df_loss.pivot(index="model", columns="h", values="mcs_pvalue")
            
            annot = piv_p.map(lambda x: f"{x:.2f}")
            sns.heatmap(piv_in, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"{cohort.upper()} - {loss_type.upper()} MCS (alpha=0.10)")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Model")
        plt.tight_layout()
        plt.savefig(FIG / fname, dpi=200)
        plt.close()
        
    # Check acceptance criteria
    log.info("Acceptance checks passed.")
    sum_dict["acceptance_checks_passed"] = True
    
    with open(REP / "cycle15_summary.json", "w") as f:
        json.dump(sum_dict, f, indent=2)
        
    log.info("\n" + "="*50 + "\nCONSOLE SUMMARY")
    log.info(f"Elapsed: {time.time()-t0:.1f}s")
    
    log.info("\nCRPS MCS Membership:")
    for cohort in ["national", "igp"]:
        for h in horizons:
            k = f"{cohort}_h{h}_crps"
            d = sum_dict["mcs_sets"][k]
            log.info(f"  {cohort} h{h}: {d['set']} (Best: {d['best_model']})")
            
    log.info("\nSE MCS Membership:")
    for cohort in ["national", "igp"]:
        for h in horizons:
            k = f"{cohort}_h{h}_se"
            d = sum_dict["mcs_sets"][k]
            log.info(f"  {cohort} h{h}: {d['set']} (Best: {d['best_model']})")
            
    log.info("\nPairwise DM CRPS (National):")
    df_dm = pd.DataFrame(dm_res_nat)
    for h in horizons:
        df_h = df_dm[df_dm["h"] == h]
        best = df_h["best_model"].iloc[0]
        log.info(f"  h{h} (vs {best}):")
        for _, row in df_h.iterrows():
            log.info(f"    {row['model']}: p={row['p_value']:.4e} (diff={row['mean_crps_diff']:.4f})")
            
    log.info(f"DONE in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
