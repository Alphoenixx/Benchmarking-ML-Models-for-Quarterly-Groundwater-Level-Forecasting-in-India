import time, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.config import ROOT
from src.logging_utils import get_logger
import src.uq as uq
import src.mcs as mcs
import src.risk as risk

warnings.filterwarnings("ignore")

TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"

def main():
    t0 = time.time()
    log = get_logger("cycle16")
    
    log.info("="*50)
    log.info("CYCLE 16: CRITICAL DEPLETION RISK EARLY-WARNING")
    log.info("="*50)
    log.info(f"Libraries: numpy={np.__version__}, pandas={pd.__version__}, matplotlib={matplotlib.__version__}")
    
    df = pd.read_parquet(TAB / "all_preds_long.parquet")
    panel = pd.read_parquet(ROOT / "data" / "processed" / "quarterly_panel.parquet")
    st_meta = panel.groupby("station_id")[["lat", "lon"]].first().reset_index()
    st_meta["is_igp"] = st_meta["lat"].between(27.5, 32.5) & st_meta["lon"].between(73.5, 81.0)
    df = df.merge(st_meta, on="station_id", how="left")
    
    train_df = panel[panel["year"] <= 2018]
    train_counts = train_df.groupby("station_id")["target"].count()
    train_90th = train_df.groupby("station_id")["target"].quantile(0.90)
    
    thresholds = pd.DataFrame({
        "n_train": train_counts,
        "tau_well": train_90th
    }).reset_index()
    
    thresholds = thresholds.merge(st_meta[["station_id", "is_igp"]], on="station_id", how="left")
    
    valid_thresholds = thresholds[thresholds["n_train"] >= 4].copy()
    excluded = thresholds[thresholds["n_train"] < 4]
    
    n_excluded_nat = len(excluded)
    n_excluded_igp = excluded["is_igp"].sum()
    log.info(f"Thresholds defined: {len(valid_thresholds)} wells. Excluded (<4 train obs): {n_excluded_nat} national, {n_excluded_igp} IGP.")
    
    valid_thresholds.to_csv(TAB / "risk_thresholds.csv", index=False)
    
    df = df[df["station_id"].isin(valid_thresholds["station_id"])]
    df = df.merge(valid_thresholds[["station_id", "tau_well"]], on="station_id", how="left")
    
    df["event"] = (df["y"] > df["tau_well"]).astype(int)
    
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
            
    test_df = df[df["split"] == "test"].copy()
    
    sample_r = cal_residuals[(models[0], horizons[0])]
    sample_yhat = np.array([10.0])
    sample_tau_well = np.array([10.0])
    p_hat_check = uq.exceedance_prob(sample_yhat, sample_r, sample_tau_well, tau_grid)[0]
    log.info(f"Median-unbiased check: p_hat(tau=yhat) = {p_hat_check:.4f} (should be ~0.5)")
    assert np.isclose(p_hat_check, 0.5, atol=0.05), "exceedance_prob is not median unbiased!"
    
    test_df["p_hat"] = np.nan
    for m in tqdm(models, desc="Computing p_hat"):
        for h in horizons:
            mask = (test_df["model"] == m) & (test_df["h"] == h)
            if not np.any(mask):
                continue
            r_cal = cal_residuals[(m, h)]
            yhat_m = test_df.loc[mask, "yhat"].values
            tau_m = test_df.loc[mask, "tau_well"].values
            p_hat_m = uq.exceedance_prob(yhat_m, r_cal, tau_m, tau_grid)
            assert np.all((p_hat_m >= 0) & (p_hat_m <= 1))
            test_df.loc[mask, "p_hat"] = p_hat_m
            
    risk_nat = []
    risk_igp = []
    rel_nat_h1 = []
    rel_igp_h1 = []
    
    sum_dict = {
        "metadata": {
            "seed": 42,
            "B": 2000,
            "alpha": 0.10,
            "excluded_wells_nat": int(n_excluded_nat),
            "excluded_wells_igp": int(n_excluded_igp),
            "numpy": np.__version__,
            "pandas": pd.__version__
        },
        "cells": {}
    }
    
    for cohort in ["national", "igp"]:
        for h in horizons:
            eval_df = test_df[test_df["h"] == h]
            if cohort == "igp":
                eval_df = eval_df[eval_df["is_igp"]]
                
            pivot = eval_df.pivot(index=["station_id", "period"], columns="model", values=["y", "yhat", "tau_well", "event", "p_hat"])
            pivot = pivot.dropna()
            
            if len(pivot) == 0:
                raise ValueError(f"Zero common rows for {cohort} h{h}")
                
            st_ids = pivot.index.get_level_values(0).values
            events = pivot["event"].iloc[:, 0].values
            
            n_obs = len(events)
            n_pos = np.sum(events)
            n_neg = n_obs - n_pos
            base_rate = n_pos / n_obs
            
            log.info(f"[{cohort} h{h}] base_rate={base_rate:.4f} (pos={n_pos}, neg={n_neg})")
            assert base_rate > 0 and base_rate < 1, f"Degenerate base rate {base_rate} for {cohort} h{h}"
            
            brier_loss = np.zeros((n_obs, n_models))
            p_hats = np.zeros((n_obs, n_models))
            
            for i, m in enumerate(models):
                p = pivot["p_hat"][m].values
                brier_loss[:, i] = (p - events)**2
                p_hats[:, i] = p
                
            mcs_p, elim, obs_mean = mcs.mcs_procedure(brier_loss, st_ids, alpha=0.10, B=2000, seed=42)
            
            clim_idx = models.index("climatology")
            brier_clim = obs_mean[clim_idx]
            
            res_list = risk_nat if cohort == "national" else risk_igp
            
            best_bss = -np.inf
            best_auc = -np.inf
            best_model_bss = None
            best_model_auc = None
            
            for i, m in enumerate(models):
                p = p_hats[:, i]
                brier = obs_mean[i]
                assert brier >= 0 and brier <= 1
                
                bss = 1 - brier / brier_clim if brier_clim > 0 else 0.0
                
                if m == "climatology":
                    assert abs(bss) < 1e-9, f"Climatology BSS is not 0: {bss}"
                
                auc = risk.compute_roc_auc(events, p)
                if n_pos == 0 or n_neg == 0:
                    log.error(f"Degenerate AUC in {cohort} h{h} model {m}")
                    auc = np.nan
                else:
                    assert np.isnan(auc) or (auc >= 0 and auc <= 1)
                    
                ece, rel_df = risk.expected_calibration_error(events, p, n_bins=10)
                
                res_list.append({
                    "cohort": cohort,
                    "h": h,
                    "model": m,
                    "brier": brier,
                    "bss": bss,
                    "auc": auc,
                    "ece": ece,
                    "base_rate": base_rate,
                    "n_pos": int(n_pos),
                    "n_neg": int(n_neg),
                    "in_brier_mcs": mcs_p[i] >= 0.10,
                    "brier_mcs_p": mcs_p[i]
                })
                
                if h == 1:
                    rel_df["model"] = m
                    if cohort == "national":
                        rel_nat_h1.append(rel_df)
                    else:
                        rel_igp_h1.append(rel_df)
                        
                if bss > best_bss:
                    best_bss = bss
                    best_model_bss = m
                if not np.isnan(auc) and auc > best_auc:
                    best_auc = auc
                    best_model_auc = m
                    
            mcs_set = [models[i] for i in range(n_models) if mcs_p[i] >= 0.10]
            
            sum_dict["cells"][f"{cohort}_h{h}"] = {
                "base_rate": base_rate,
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "best_model_bss": best_model_bss,
                "best_model_auc": best_model_auc,
                "brier_mcs_set": mcs_set
            }

    df_nat = pd.DataFrame(risk_nat)
    df_igp = pd.DataFrame(risk_igp)
    
    assert not df_nat.drop(columns=["auc"]).isnull().any().any()
    assert not df_igp.drop(columns=["auc"]).isnull().any().any()
    
    df_nat.to_csv(TAB / "risk_scores_national.csv", index=False)
    df_igp.to_csv(TAB / "risk_scores_igp.csv", index=False)
    
    df_rel_nat = pd.concat(rel_nat_h1, ignore_index=True)
    df_rel_igp = pd.concat(rel_igp_h1, ignore_index=True)
    df_rel_nat.to_csv(TAB / "risk_reliability_national_h1.csv", index=False)
    df_rel_igp.to_csv(TAB / "risk_reliability_igp_h1.csv", index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, (cohort, df_rel) in enumerate([("National", df_rel_nat), ("IGP", df_rel_igp)]):
        ax = axes[j]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Perfect")
        for m in models:
            m_df = df_rel[df_rel["model"] == m].dropna(subset=["mean_phat", "emp_freq"])
            ax.plot(m_df["mean_phat"], m_df["emp_freq"], marker='o', label=m, alpha=0.7)
        ax.set_title(f"{cohort} - Reliability (h=1)")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Empirical Frequency")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIG / "risk_reliability_h1.png", dpi=200)
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, (cohort, df_res) in enumerate([("National", df_nat), ("IGP", df_igp)]):
        ax = axes[j]
        sns.lineplot(data=df_res, x="h", y="bss", hue="model", marker="o", ax=ax)
        ax.set_title(f"{cohort} - BSS vs Horizon")
        ax.set_ylabel("Brier Skill Score (vs Climatology)")
        ax.set_xticks(horizons)
        if j == 0:
            ax.get_legend().remove()
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIG / "risk_bss_vs_horizon.png", dpi=200)
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for j, (cohort, df_res) in enumerate([("National", df_nat), ("IGP", df_igp)]):
        ax = axes[j]
        piv = df_res.pivot(index="model", columns="h", values="auc")
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, vmin=0.5, vmax=1.0)
        ax.set_title(f"{cohort} - ROC AUC")
    plt.tight_layout()
    plt.savefig(FIG / "risk_auc_heatmap.png", dpi=200)
    plt.close()
    
    log.info("Acceptance checks passed.")
    sum_dict["acceptance_checks_passed"] = True
    
    with open(REP / "cycle16_summary.json", "w") as f:
        json.dump(sum_dict, f, indent=2)
        
    log.info("\n" + "="*50 + "\nCONSOLE SUMMARY")
    log.info(f"Elapsed: {time.time()-t0:.1f}s")
    
    log.info("\nBSS Table (National):")
    piv_bss_nat = df_nat.pivot(index="model", columns="h", values="bss")
    log.info("\n" + piv_bss_nat.to_string())
    log.info("\nBSS Table (IGP):")
    piv_bss_igp = df_igp.pivot(index="model", columns="h", values="bss")
    log.info("\n" + piv_bss_igp.to_string())
    
    log.info("\nAUC Table (National):")
    piv_auc_nat = df_nat.pivot(index="model", columns="h", values="auc")
    log.info("\n" + piv_auc_nat.to_string())
    log.info("\nAUC Table (IGP):")
    piv_auc_igp = df_igp.pivot(index="model", columns="h", values="auc")
    log.info("\n" + piv_auc_igp.to_string())
    
    log.info("\nECE per model at h1:")
    for m in models:
        ece_nat = df_nat[(df_nat["h"]==1) & (df_nat["model"]==m)]["ece"].values[0]
        ece_igp = df_igp[(df_igp["h"]==1) & (df_igp["model"]==m)]["ece"].values[0]
        log.info(f"  {m}: Nat={ece_nat:.4f}, IGP={ece_igp:.4f}")
        
    log.info("\nBrier-MCS Membership (alpha=0.10):")
    for cohort in ["national", "igp"]:
        for h in horizons:
            k = f"{cohort}_h{h}"
            d = sum_dict["cells"][k]
            log.info(f"  {cohort} h{h}: {d['brier_mcs_set']}")
            
    log.info(f"DONE in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
