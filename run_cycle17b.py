import pandas as pd
import numpy as np
import joblib
import time
import json
import logging
from pathlib import Path
from src import dataset as D, features as F, uq

SEED = 42
np.random.seed(SEED)

TAB = Path("outputs/tables")
MOD = Path("outputs/models")
FIG = Path("outputs/figures")
REP = Path("outputs/reports")
LOG = Path("outputs/logs")

log_file = LOG / f"cycle17b_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

def main():
    t0 = time.time()
    log.info(f"Logger started -> {log_file}")
    log.info("=" * 50)
    log.info("CYCLE 17B: FIX CONFORMAL COVERAGE INDEX INVERSION (PART B ONLY)")
    log.info("=" * 50)
    
    import sklearn
    import matplotlib
    log.info(f"np={np.__version__} pd={pd.__version__} mpl={matplotlib.__version__} skl={sklearn.__version__}")
    
    panel = D.load_panel()
    horizons = [1, 2, 3, 4]
    tau_grid = np.arange(0.01, 1.0, 0.01)
    
    # 1) Get / Cache Temporal Predictions
    temp_preds_path = TAB / "temporal_blocks_predictions.parquet"
    if temp_preds_path.exists():
        log.info(f"Reusing cached TEMPORAL predictions from {temp_preds_path.name}")
        all_preds = pd.read_parquet(temp_preds_path)
    else:
        log.info("Generating TEMPORAL predictions from canonical all_preds_long.parquet...")
        canonical = pd.read_parquet(TAB / "all_preds_long.parquet")
        all_preds = canonical[canonical["model"] == "random_forest"].copy()
        all_preds.to_parquet(temp_preds_path, index=False)
        log.info(f"Saved temporal predictions to {temp_preds_path.name}")

    # 2) Get / Cache Spatial Predictions
    spat_preds_path = TAB / "spatial_blocks_predictions.parquet"
    if spat_preds_path.exists():
        log.info(f"Reusing cached SPATIAL predictions from {spat_preds_path.name}")
        spat_preds = pd.read_parquet(spat_preds_path)
    else:
        log.info("Generating SPATIAL predictions (running 20 RF models)...")
        blocks = pd.read_csv(TAB / "spatial_blocks_assignment.csv", dtype={"station_id": str})
        block_map = dict(zip(blocks["station_id"], blocks["spatial_block"]))
        
        coh = D.cohort(panel, min_q=16, region=None)
        recs = []
        log.info(f"Building temporal features for {len(coh['station_id'].unique())} wells...")
        for sid, g in coh.groupby("station_id", sort=False):
            recs.extend(F.build_well(g.copy(), horizons))
        log.info("Feature building complete. Starting fold models...")
        
        frame = pd.DataFrame(recs)
        frame["spatial_block"] = frame["station_id"].map(block_map)
        frame = frame.dropna(subset=["spatial_block"])
        
        def get_h_data(fr, h):
            sub = fr[fr[f"target_h{h}"].notna()].copy()
            sub["y"] = sub[f"target_h{h}"]
            yy = sub[f"target_h{h}_year"]
            sub["split"] = np.where(yy <= 2018, "train",
                           np.where(yy <= 2020, "val",
                           np.where(yy <= 2023, "test", "exclude")))
            sub["h"] = h
            sub["period"] = sub[f"target_h{h}_period"]
            drop = {"station_id", "split", "h", "period", "y", "is_igp", "scale", "yhat_sn", "spatial_block"}
            for hz in [1, 2, 3, 4]:
                drop |= {f"target_h{hz}", f"target_h{hz}_year", f"target_h{hz}_period"}
            cols = [c for c in sub.columns if c not in drop]
            med = sub[sub["split"] == "train"][cols].median(numeric_only=True)
            
            X = sub[cols].copy()
            miss = X.isna().astype(int); miss.columns = [f"{c}__miss" for c in cols]
            out = pd.concat([X.fillna(med), miss], axis=1)
            for c in ["station_id", "split", "h", "period", "y", "is_igp", "scale", "yhat_sn", "spatial_block"]:
                if c in sub.columns:
                    out[c] = sub[c]
            return out, cols + list(miss.columns)
            
        def rf_for(h):
            from sklearn.ensemble import RandomForestRegressor
            p = joblib.load(MOD / f"rf_h{h}.joblib").get_params()
            p["random_state"] = SEED
            p["n_jobs"] = -1
            return RandomForestRegressor(**p), list(joblib.load(MOD / f"rf_h{h}.joblib").feature_names_in_)
            
        spat_preds_list = []
        
        for h in horizons:
            model_tmpl, feats = rf_for(h)
            h_frame, _ = get_h_data(frame, h)
            
            for k in range(5):
                tr_mask = (h_frame["spatial_block"] != k) & (h_frame["split"] == "train")
                val_mask = (h_frame["spatial_block"] != k) & (h_frame["split"] == "val")
                test_mask = (h_frame["spatial_block"] == k) & (h_frame["split"] == "test")
                
                tr = h_frame[tr_mask]
                val = h_frame[val_mask].copy()
                te = h_frame[test_mask].copy()
                
                if len(te) == 0: continue
                
                m = model_tmpl
                m.fit(tr[feats].to_numpy(), tr["y"].to_numpy())
                
                val["yhat"] = m.predict(val[feats].to_numpy())
                te["yhat"] = m.predict(te[feats].to_numpy())
                
                val["fold"] = k
                te["fold"] = k
                
                spat_preds_list.append(val[["station_id", "period", "h", "split", "fold", "spatial_block", "y", "yhat"]])
                spat_preds_list.append(te[["station_id", "period", "h", "split", "fold", "spatial_block", "y", "yhat"]])
                log.info(f"h={h} fold={k} done (Tr={len(tr)}, Val={len(val)}, Te={len(te)})")
                
        spat_preds = pd.concat(spat_preds_list, ignore_index=True)
        spat_preds.to_parquet(spat_preds_path, index=False)
        log.info(f"Saved spatial predictions to {spat_preds_path.name}")

    # 3) Metric Calculation
    results = []
    
    # Temporal
    for h in horizons:
        val_df = all_preds[(all_preds["h"] == h) & (all_preds["split"] == "val")]
        te_df = all_preds[(all_preds["h"] == h) & (all_preds["split"] == "test")]
        
        r_val = np.abs(val_df["y"] - val_df["yhat"]).values
        y_te = te_df["y"].values
        yhat_te = te_df["yhat"].values
        
        c_val, _ = uq.compute_crps_and_pinball(y_te, yhat_te, r_val, tau_grid)
        
        for nominal in [0.5, 0.8, 0.9, 0.95]:
            q_val = uq.get_conformal_radius(r_val, nominal)  # FIX: pass nominal (c) directly
            
            picp = np.mean(np.abs(y_te - yhat_te) <= q_val)
            mpiw = 2 * q_val
            
            results.append({
                "h": h, "nominal": nominal, "split": "temporal",
                "picp": picp, "mpiw": mpiw, "crps": c_val, "n": len(y_te)
            })

    # Spatial
    for h in horizons:
        val_sp = spat_preds[(spat_preds["h"] == h) & (spat_preds["split"] == "val")]
        te_sp = spat_preds[(spat_preds["h"] == h) & (spat_preds["split"] == "test")]
        
        for nominal in [0.5, 0.8, 0.9, 0.95]:
            agg_picps = []
            agg_mpiws = []
            agg_crps = []
            n_tot = 0
            
            for k in range(5):
                val_k = val_sp[val_sp["fold"] == k]
                te_k = te_sp[te_sp["fold"] == k]
                
                if len(te_k) == 0: continue
                
                r_val = np.abs(val_k["y"] - val_k["yhat"]).values
                y_te = te_k["y"].values
                yhat_te = te_k["yhat"].values
                
                c_val, _ = uq.compute_crps_and_pinball(y_te, yhat_te, r_val, tau_grid)
                q_val = uq.get_conformal_radius(r_val, nominal)  # FIX: pass nominal (c) directly
                
                picp = np.mean(np.abs(y_te - yhat_te) <= q_val)
                mpiw = 2 * q_val
                
                agg_picps.append(picp * len(y_te))
                agg_mpiws.append(mpiw * len(y_te))
                agg_crps.append(c_val * len(y_te))
                n_tot += len(y_te)
                
            results.append({
                "h": h, "nominal": nominal, "split": "spatial",
                "picp": sum(agg_picps)/n_tot, "mpiw": sum(agg_mpiws)/n_tot, "crps": sum(agg_crps)/n_tot, "n": n_tot
            })
            
    df_res = pd.DataFrame(results)
    df_res["ace"] = df_res["picp"] - df_res["nominal"]
    df_res.to_csv(TAB / "spatial_shift_coverage.csv", index=False)
    
    # Mondrian skipped
    log.info("Mondrian skipped: too slow to map nearest training blocks cleanly per row in current setup.")
    
    # 4) Sanity Gates
    gates_passed = []
    
    # Gate 1: MPIW strictly increasing
    gate1 = True
    for (h, split), g in df_res.groupby(["h", "split"]):
        g_sorted = g.sort_values("nominal")
        if not g_sorted["mpiw"].is_monotonic_increasing:
            gate1 = False
            log.error(f"Gate 1 Failed: MPIW not monotonic for h={h} split={split}")
            break
    if gate1:
        log.info("GATE 1 PASS: MPIW strictly increasing in nominal")
        gates_passed.append("gate1_mpiw_increasing")
        
    # Gate 2: Temporal PICP@0.9 ~= 0.92
    gate2 = True
    t_picp90 = df_res[(df_res["split"] == "temporal") & (np.isclose(df_res["nominal"], 0.9))]
    for idx, row in t_picp90.iterrows():
        if abs(row["picp"] - 0.92) > 0.05:
            gate2 = False
            log.error(f"Gate 2 Failed: h={row['h']} temporal PICP@0.9 is {row['picp']:.4f} (expected ~0.92)")
            break
    if gate2:
        log.info("GATE 2 PASS: Temporal PICP@0.9 is near 0.92")
        gates_passed.append("gate2_picp90_near_92")
        
    # Gate 3: Guard against re-inversion
    gate3 = True
    for idx, row in df_res[np.isclose(df_res["nominal"], 0.9)].iterrows():
        picp = row["picp"]
        nom = row["nominal"]
        if abs(picp - nom) >= abs(picp - (1 - nom)):
            gate3 = False
            log.error(f"Gate 3 Failed: PICP {picp:.4f} is closer to {1-nom} than {nom}")
            break
    if gate3:
        log.info("GATE 3 PASS: No index inversion detected")
        gates_passed.append("gate3_no_inversion")
        
    if not (gate1 and gate2 and gate3):
        raise RuntimeError("Sanity gates failed!")

    # 5) Summary Outputs
    log.info("\n--- NATIONAL TEMPORAL VS SPATIAL PICP ---")
    for h in horizons:
        for nom in [0.8, 0.9, 0.95]:
            t_r = df_res[(df_res["split"]=="temporal") & (df_res["h"]==h) & (np.isclose(df_res["nominal"], nom))].iloc[0]
            s_r = df_res[(df_res["split"]=="spatial") & (df_res["h"]==h) & (np.isclose(df_res["nominal"], nom))].iloc[0]
            log.info(f"h={h} @ {nom}: Temp PICP={t_r['picp']:.4f} (ACE={t_r['ace']:+.4f}) | Spat PICP={s_r['picp']:.4f} (ACE={s_r['ace']:+.4f})")
            
    log.info("\n--- MPIW @ 0.90 ---")
    for h in horizons:
        t_r = df_res[(df_res["split"]=="temporal") & (df_res["h"]==h) & (np.isclose(df_res["nominal"], 0.9))].iloc[0]
        s_r = df_res[(df_res["split"]=="spatial") & (df_res["h"]==h) & (np.isclose(df_res["nominal"], 0.9))].iloc[0]
        log.info(f"h={h}: Temp={t_r['mpiw']:.3f} | Spat={s_r['mpiw']:.3f}")
        
    # 6) Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Coverage")
    colors = ["C0", "C1", "C2", "C3"]
    for i, h in enumerate(horizons):
        t_sub = df_res[(df_res["h"] == h) & (df_res["split"] == "temporal")]
        s_sub = df_res[(df_res["h"] == h) & (df_res["split"] == "spatial")]
        plt.plot(t_sub["nominal"], t_sub["picp"], "o-", color=colors[i], label=f"Temp h={h}")
        plt.plot(s_sub["nominal"], s_sub["picp"], "x--", color=colors[i], label=f"Spat h={h}")
    plt.xlabel("Nominal Coverage")
    plt.ylabel("Observed PICP")
    plt.legend()
    plt.title("Conformal Coverage: Temporal vs Spatial Shift")
    plt.grid(True, alpha=0.3)
    plt.savefig(FIG / "spatial_shift_coverage.png", dpi=300, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - t0
    log.info(f"DONE in {elapsed:.1f}s")
    
    summary = {
        "elapsed_sec": elapsed,
        "gates_passed": gates_passed,
        "mondrian_run": False,
        "results": df_res.to_dict(orient="records")
    }
    with open(REP / "cycle17b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
