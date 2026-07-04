import json
import time
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ROOT, SEED
from src.logging_utils import get_logger
import src.uq as uq
from src import dataset as D
from src.prob_extra import get_prob_climatology, decompose_crps

np.random.seed(SEED)
torch.manual_seed(SEED)

TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"
LOG = ROOT / "outputs" / "logs"

log_file = LOG / f"cycle18_{time.strftime('%Y%m%d_%H%M%S')}.log"
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

HORIZONS = [1, 2, 3, 4]
TAU_GRID = np.arange(0.01, 1.0, 0.01)

def main():
    t0 = time.time()
    log.info(f"Logger started -> {log_file}")
    log.info("="*50)
    log.info("CYCLE 18: NATIVE CHRONOS VS CONFORMAL + CRPS DECOMPOSITION")
    log.info("="*50)
    
    import sklearn
    log.info(f"np={np.__version__} pd={pd.__version__} mpl={matplotlib.__version__} skl={sklearn.__version__} torch={torch.__version__}")
    
    # 1) Load Common Set
    canonical = pd.read_parquet(TAB / "all_preds_long.parquet")
    canonical["period"] = pd.to_datetime(canonical["period"])
    
    # Common test set per horizon
    test_idx = canonical[(canonical["model"] == "chronos") & (canonical["split"] == "test")]
    for h in HORIZONS:
        n_h = len(test_idx[test_idx["h"] == h])
        log.info(f"Common test set h={h}: n={n_h}")
        
    # Get test context keys
    com_keys = test_idx[["station_id", "period", "h", "split", "y", "scale"]].copy()
    com_keys["period"] = pd.to_datetime(com_keys["period"])
    
    # Load Panel
    panel = D.load_panel()
    nat = D.cohort(panel, min_q=16)
    
    # Build histories for Chronos (Part A)
    native_parquet = TAB / "chronos_native_quantiles.parquet"
    if native_parquet.exists():
        log.info("Reusing chronos native quantiles from cache")
        native_q_df = pd.read_parquet(native_parquet)
        native_q_df["period"] = pd.to_datetime(native_q_df["period"])
    else:
        log.info("Running Chronos zero-shot inference for native quantiles...")
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)
        
        # Build contexts
        uni_ctx, uni_key = [], []
        seen = set()
        meta = []
        for sid, g in tqdm(nat.groupby("station_id"), desc="chronos-ctx"):
            gg = D.reindex_well(g).reset_index(drop=True)
            dates = [str(pd.Timestamp(t).date()) for t in gg["datetime"]]
            pos = {pd.to_datetime(d): i for i, d in enumerate(dates)}
            y_arr = gg["target"].to_numpy(dtype=float)
            
            sub = com_keys[com_keys["station_id"] == sid]
            if sub.empty: continue
            
            for _, r in sub.iterrows():
                j = pos.get(pd.to_datetime(r["period"]))
                if j is None: continue
                o = j - int(r["h"])
                if o < 0: continue
                
                ctx_raw = y_arr[:o + 1]
                if np.isfinite(ctx_raw).sum() < 4: continue
                
                arr = pd.Series(ctx_raw).interpolate(limit_direction="both").to_numpy()
                if not np.isfinite(arr).all(): continue
                
                key = (sid, o)
                if key not in seen:
                    seen.add(key)
                    uni_ctx.append(torch.tensor(arr, dtype=torch.float32))
                    uni_key.append(key)
                meta.append((sid, int(r["h"]), pd.to_datetime(r["period"]), o))
                
        log.info(f"Unique contexts: {len(uni_ctx)}")
        
        preds_q = {}
        CH_BATCH = 64
        for i in tqdm(range(0, len(uni_ctx), CH_BATCH), desc="chronos-predict"):
            batch = uni_ctx[i:i+CH_BATCH]
            q, _ = pipe.predict_quantiles(batch, prediction_length=max(HORIZONS), quantile_levels=TAU_GRID.tolist())
            if isinstance(q, (list, tuple)):
                q = torch.stack(q)
            q_batch = q.cpu().numpy()[:, 0, :, :] # [B, H, len(TAU)]
            
            for k, key in enumerate(uni_key[i:i+CH_BATCH]):
                preds_q[key] = q_batch[k]
                
        rows = []
        for sid, h, period, o in meta:
            p = preds_q.get((sid, o))
            if p is None: continue
            if len(p) < h: continue
            q_row = p[h-1] # [99]
            rows.append([sid, h, period] + q_row.tolist())
            
        col_names = ["station_id", "h", "period"] + [f"q_{tau:.2f}" for tau in TAU_GRID]
        native_q_df = pd.DataFrame(rows, columns=col_names)
        
        # Check crossings
        q_cols = [f"q_{tau:.2f}" for tau in TAU_GRID]
        arr = native_q_df[q_cols].values
        crossings = (np.diff(arr, axis=1) < -1e-6).sum()
        if crossings > 0:
            log.warning(f"Found {crossings} quantile crossings in native Chronos! Fixing with maximum.accumulate.")
            arr = np.maximum.accumulate(arr, axis=1)
            native_q_df[q_cols] = arr
            
        native_q_df.to_parquet(native_parquet, index=False)
        log.info(f"Saved native quantiles to {native_parquet.name}")

    # Build conformal chronos
    log.info("Building conformalized Chronos...")
    conformal_q_list = []
    
    val_idx = canonical[(canonical["model"] == "chronos") & (canonical["split"] == "val")]
    for h in HORIZONS:
        # Calibrate
        val_h = val_idx[val_idx["h"] == h]
        r_cal = np.abs(val_h["y"] - val_h["yhat"]).values
        
        # Predict on test
        te_h = test_idx[test_idx["h"] == h].copy()
        yhat_te = te_h["yhat"].values
        
        c_vals = np.where(TAU_GRID >= 0.5, 2*TAU_GRID - 1, 1 - 2*TAU_GRID)
        q_vals = np.zeros_like(c_vals)
        mask = c_vals > 0
        if np.any(mask):
            q_vals[mask] = uq.get_conformal_radius_grid(r_cal, c_vals[mask])
            
        yhat_exp = yhat_te[:, None]
        q_vals_exp = q_vals[None, :]
        tau_exp = TAU_GRID[None, :]
        
        Q_tau = np.where(tau_exp >= 0.5, yhat_exp + q_vals_exp, yhat_exp - q_vals_exp)
        
        q_df = pd.DataFrame(Q_tau, columns=[f"q_{tau:.2f}" for tau in TAU_GRID])
        q_df["station_id"] = te_h["station_id"].values
        q_df["period"] = pd.to_datetime(te_h["period"].values)
        q_df["h"] = h
        q_df["y"] = te_h["y"].values
        q_df["yhat"] = te_h["yhat"].values
        q_df["scale"] = te_h["scale"].values
        
        conformal_q_list.append(q_df)
        
    conf_q_df = pd.concat(conformal_q_list, ignore_index=True)
    
    # Helper to compute metrics
    st_meta = panel.groupby("station_id")[["lat", "lon"]].first().reset_index()
    st_meta["is_igp"] = st_meta["lat"].between(27.5, 32.5) & st_meta["lon"].between(73.5, 81.0)
    
    def evaluate_quantiles(q_df, variant_name):
        q_df = q_df.merge(st_meta[["station_id", "is_igp"]], on="station_id", how="left")
        res = []
        q_cols = [f"q_{tau:.2f}" for tau in TAU_GRID]
        
        for igp_mode in [False, True]:
            cohort_name = "igp" if igp_mode else "national"
            for h in HORIZONS:
                sub = q_df[(q_df["h"] == h)]
                if igp_mode: sub = sub[sub["is_igp"]]
                
                if len(sub) == 0: continue
                
                y = sub["y"].values
                Q = sub[q_cols].values
                
                # CRPS
                crps_arr = uq.compute_crps_from_quantiles(y, Q, TAU_GRID)
                crps_val = np.mean(crps_arr)
                
                # Pinball
                pinball_val = np.mean([uq.compute_pinball(y, Q[:, i], t) for i, t in enumerate(TAU_GRID)])
                
                row = {"variant": variant_name, "cohort": cohort_name, "h": h, "n": len(y), "crps": crps_val, "pinball": pinball_val}
                
                # Coverages
                for level in [0.5, 0.8, 0.9, 0.95]:
                    alpha = 1 - level
                    low_idx = np.argmin(np.abs(TAU_GRID - alpha/2))
                    high_idx = np.argmin(np.abs(TAU_GRID - (1 - alpha/2)))
                    lower = Q[:, low_idx]
                    upper = Q[:, high_idx]
                    
                    picp = np.mean((y >= lower) & (y <= upper))
                    mpiw = np.mean(upper - lower)
                    
                    row[f"picp_{level}"] = picp
                    if level == 0.9:
                        row["mpiw_0.9"] = mpiw
                        row["ace_0.9"] = picp - level
                        row["winkler_0.9"] = uq.compute_winkler(y, lower, upper, alpha)
                        
                # Reliability max_dev
                emp_covs = []
                nom_covs = []
                for lev in np.arange(0.1, 1.0, 0.1):
                    a = 1 - lev
                    l_idx = np.argmin(np.abs(TAU_GRID - a/2))
                    h_idx = np.argmin(np.abs(TAU_GRID - (1 - a/2)))
                    emp_covs.append(np.mean((y >= Q[:, l_idx]) & (y <= Q[:, h_idx])))
                    nom_covs.append(lev)
                row["reliability_maxdev"] = np.max(np.abs(np.array(emp_covs) - np.array(nom_covs)))
                
                res.append(row)
        return pd.DataFrame(res)

    # Merge target y into native_q_df
    native_q_df = native_q_df.merge(com_keys[["station_id", "period", "h", "y"]], on=["station_id", "period", "h"])
    
    native_scores = evaluate_quantiles(native_q_df, "native")
    conf_scores = evaluate_quantiles(conf_q_df, "conformal")
    
    cv_chronos = pd.concat([native_scores, conf_scores], ignore_index=True)
    cv_chronos.to_csv(TAB / "chronos_native_vs_conformal_scores.csv", index=False)
    
    # Gate: conformal chronos CRPS reproduces cycle 14? 
    crps_c14_approx = conf_scores[(conf_scores["cohort"]=="national") & (conf_scores["h"]==1)]["crps"].values[0]
    log.info(f"Conformal Chronos h=1 CRPS: {crps_c14_approx:.6f} (Expected: 1.082396)")
    assert abs(crps_c14_approx - 1.082396) < 1e-5, f"CRPS mismatch! {crps_c14_approx}"
    
    # (B) Probabilistic Classical Baseline
    log.info("\n--- PART B: Probabilistic Climatology Baseline ---")
    train_y = canonical[(canonical["split"] == "train") & (canonical["model"] == "rf")].copy()
    test_y = com_keys.copy()
    
    prob_clim_preds = get_prob_climatology(train_y, test_y, TAU_GRID)
    
    pc_df = test_y.copy()
    pc_q_cols = [f"q_{tau:.2f}" for tau in TAU_GRID]
    pc_df[pc_q_cols] = prob_clim_preds
    
    pc_scores = evaluate_quantiles(pc_df, "prob_climatology")
    pc_scores.to_csv(TAB / "prob_classical_scores.csv", index=False)
    
    # (C) CRPS Decomposition
    log.info("\n--- PART C: CRPS Decomposition ---")
    
    # History length cuts
    hist_counts = panel.dropna(subset=["target"]).groupby("station_id").size()
    
    def get_tercile(n):
        if n <= 18: return "short"
        if n <= 58: return "medium"
        return "long"
        
    tercile_map = hist_counts.apply(get_tercile).to_dict()
    
    decomp_rows = []
    
    # We will compute per-row CRPS for: prob_climatology, rf, chronos_conformal
    # 1. Prob climatology
    pc_df["crps_pc"] = uq.compute_crps_from_quantiles(pc_df["y"].values, pc_df[pc_q_cols].values, TAU_GRID)
    pc_df["history_tercile"] = pc_df["station_id"].map(tercile_map)
    pc_df = pc_df.merge(st_meta[["station_id", "is_igp"]], on="station_id", how="left")
    
    # 2. RF Conformal (we have this in run_cycle14, but we can reconstruct it fast)
    # Actually, we can just rebuild it or get it. Let's rebuild RF conformal test predictions.
    rf_val = canonical[(canonical["model"] == "random_forest") & (canonical["split"] == "val")]
    rf_te = canonical[(canonical["model"] == "random_forest") & (canonical["split"] == "test")]
    rf_crps = []
    for h in HORIZONS:
        v_h = rf_val[rf_val["h"] == h]
        t_h = rf_te[rf_te["h"] == h].copy()
        
        r_cal = np.abs(v_h["y"] - v_h["yhat"]).values
        c_vals = np.where(TAU_GRID >= 0.5, 2*TAU_GRID - 1, 1 - 2*TAU_GRID)
        q_vals = np.zeros_like(c_vals)
        mask = c_vals > 0
        if np.any(mask): q_vals[mask] = uq.get_conformal_radius_grid(r_cal, c_vals[mask])
        
        yhat_te = t_h["yhat"].values
        Q_tau = np.where(TAU_GRID[None, :] >= 0.5, yhat_te[:, None] + q_vals[None, :], yhat_te[:, None] - q_vals[None, :])
        crps_arr = uq.compute_crps_from_quantiles(t_h["y"].values, Q_tau, TAU_GRID)
        t_h["crps_rf"] = crps_arr
        rf_crps.append(t_h[["station_id", "period", "h", "crps_rf"]])
        
    rf_crps_df = pd.concat(rf_crps)
    
    # 3. Chronos Conformal CRPS
    conf_q_df["crps_chronos"] = uq.compute_crps_from_quantiles(conf_q_df["y"].values, conf_q_df[pc_q_cols].values, TAU_GRID)
    
    # Merge all
    merged = pc_df[["station_id", "period", "h", "is_igp", "history_tercile", "crps_pc"]].copy()
    merged = merged.merge(rf_crps_df, on=["station_id", "period", "h"])
    merged = merged.merge(conf_q_df[["station_id", "period", "h", "crps_chronos"]], on=["station_id", "period", "h"])
    
    # Decompose
    res_dec = []
    for model_col, model_name in [("crps_pc", "prob_climatology"), ("crps_rf", "random_forest"), ("crps_chronos", "chronos_conformal")]:
        for igp_mode, cohort_name in [(True, "igp"), (False, "non_igp")]:
            for tercile in ["short", "medium", "long"]:
                for h in HORIZONS:
                    sub = merged[(merged["is_igp"] == igp_mode) & (merged["history_tercile"] == tercile) & (merged["h"] == h)]
                    if len(sub) == 0: continue
                    mean_crps = sub[model_col].mean()
                    mean_baseline = sub["crps_pc"].mean()
                    skill = 1.0 - mean_crps / mean_baseline if mean_baseline > 0 else 0
                    res_dec.append({
                        "model": model_name, "region": cohort_name, "history_tercile": tercile, "h": h,
                        "n": len(sub), "mean_crps": mean_crps, "crps_skill_vs_prob_climatology": skill
                    })
                    
    df_decomp = pd.DataFrame(res_dec)
    
    for h in HORIZONS:
        n_expected = len(test_idx[test_idx["h"] == h])
        n_actual = df_decomp[(df_decomp["h"] == h) & (df_decomp["model"] == "prob_climatology") & (df_decomp["region"] != "igp")]["n"].sum() + df_decomp[(df_decomp["h"] == h) & (df_decomp["model"] == "prob_climatology") & (df_decomp["region"] == "igp")]["n"].sum()
        # Note: region="igp" and "non_igp" partition the data, so summing over both gives total
        n_actual = df_decomp[(df_decomp["h"] == h) & (df_decomp["model"] == "prob_climatology")]["n"].sum()
        assert n_actual == n_expected, f"Row count mismatch at h={h}: {n_actual} vs {n_expected}"
        
        # Check prob_climatology self-skill is 0
        pc_skill = df_decomp[(df_decomp["h"] == h) & (df_decomp["model"] == "prob_climatology")]["crps_skill_vs_prob_climatology"].abs().max()
        assert pc_skill < 1e-9, f"Prob climatology self skill is not 0: {pc_skill}"

    df_decomp.to_csv(TAB / "crps_decomposition.csv", index=False)
    
    # Print Console Summary
    log.info("\n--- CHRONOS NATIVE VS CONFORMAL ---")
    for h in HORIZONS:
        for c in ["national", "igp"]:
            nat = native_scores[(native_scores["cohort"]==c) & (native_scores["h"]==h)].iloc[0]
            con = conf_scores[(conf_scores["cohort"]==c) & (conf_scores["h"]==h)].iloc[0]
            log.info(f"h={h} {c}: CRPS [Native={nat['crps']:.3f}, Conf={con['crps']:.3f}] | PICP@90 [Native={nat['picp_0.9']:.3f}, Conf={con['picp_0.9']:.3f}] | RelMaxDev [Native={nat['reliability_maxdev']:.3f}, Conf={con['reliability_maxdev']:.3f}]")
            
    log.info("\n--- PROB CLASSICAL BASELINE ---")
    for h in HORIZONS:
        for c in ["national", "igp"]:
            pc = pc_scores[(pc_scores["cohort"]==c) & (pc_scores["h"]==h)].iloc[0]
            log.info(f"h={h} {c}: CRPS={pc['crps']:.3f} | PICP@90={pc['picp_0.9']:.3f}")
            
    log.info("\n--- CRPS DECOMPOSITION (h=1) ---")
    h1_dec = df_decomp[df_decomp["h"]==1].sort_values(["model", "region", "history_tercile"])
    for _, r in h1_dec.iterrows():
        log.info(f"{r['model']} | {r['region']} | {r['history_tercile']}: n={r['n']} | CRPS={r['mean_crps']:.3f} (Skill={r['crps_skill_vs_prob_climatology']:.3f})")

    # Plot 1: Native vs Conformal CRPS
    plt.figure(figsize=(7,5))
    for c, ls in zip(["national", "igp"], ["-", "--"]):
        n_c = native_scores[native_scores["cohort"] == c]
        c_c = conf_scores[conf_scores["cohort"] == c]
        plt.plot(n_c["h"], n_c["crps"], color="blue", linestyle=ls, marker="o", label=f"Native ({c})")
        plt.plot(c_c["h"], c_c["crps"], color="orange", linestyle=ls, marker="x", label=f"Conformal ({c})")
    plt.xlabel("Horizon (Quarters)")
    plt.ylabel("Mean CRPS")
    plt.title("Chronos CRPS: Native vs Conformal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIG / "chronos_native_vs_conformal_crps.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 2: CRPS by Region x History
    plt.figure(figsize=(10,6))
    h1_bar = h1_dec[h1_dec["model"] != "prob_climatology"].copy()
    h1_bar["group"] = h1_bar["region"] + "_" + h1_bar["history_tercile"]
    
    import seaborn as sns
    sns.barplot(data=h1_bar, x="group", y="mean_crps", hue="model")
    plt.xticks(rotation=45)
    plt.title("CRPS by Stratum (h=1)")
    plt.tight_layout()
    plt.savefig(FIG / "crps_by_region_history.png", dpi=300)
    plt.close()

    elapsed = time.time() - t0
    log.info(f"DONE in {elapsed:.1f}s")
    
    summary = {
        "elapsed": elapsed,
        "native_vs_conformal": cv_chronos.to_dict(orient="records"),
        "prob_classical": pc_scores.to_dict(orient="records"),
        "crps_decomp": df_decomp.to_dict(orient="records")
    }
    with open(REP / "cycle18_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
