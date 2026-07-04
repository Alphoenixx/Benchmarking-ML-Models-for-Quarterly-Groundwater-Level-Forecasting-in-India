import json, time, warnings
import numpy as np, pandas as pd
from pathlib import Path
import joblib
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.config import ROOT, SEED
from src.logging_utils import get_logger
from src import dataset as D, features as F
from src.climate_features import add_climate_features, ENGINEERED_CLIMATE

warnings.filterwarnings("ignore")
log = get_logger("cycle11")

HORIZONS = [1, 2, 3, 4]
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"
MOD = ROOT / "outputs" / "models"
for p in (TAB, FIG, REP, MOD): p.mkdir(parents=True, exist_ok=True)

IGP_LAT, IGP_LON = (27.5, 32.5), (73.5, 81.0)
def is_igp(lat, lon):
    return (lat >= IGP_LAT[0]) & (lat <= IGP_LAT[1]) & (lon >= IGP_LON[0]) & (lon <= IGP_LON[1])

def fam(name: str) -> str:
    eng = tuple(ENGINEERED_CLIMATE)
    if name.startswith(eng): return "engineered_climate"
    if name.startswith(("rainfall", "t2m")): return "covariate"      # raw covariate lags
    if name.startswith(("sin_q", "cos_q")): return "seasonal"
    if name.startswith(("lat", "lon", "wellDepth")): return "static"
    if name.startswith(("y_", "target")): return "autoregressive"
    return "other"

def metrics(df, scale_col="scale"):
    """RMSE, per-well MASE (m=4 seasonal scale), per-well median NSE."""
    err = df["yhat"] - df["y"]
    rmse = float(np.sqrt(np.mean(err**2)))
    mase = float(np.mean(np.abs(err) / df[scale_col].replace(0, np.nan)))
    nse = []
    for _, g in df.groupby("station_id"):
        denom = np.sum((g["y"] - g["y"].mean())**2)
        if denom > 0:
            nse.append(1 - np.sum((g["yhat"] - g["y"])**2) / denom)
    return rmse, mase, (float(np.median(nse)) if nse else np.nan)

def get_h_data(df, h):
    sub = df[df[f"target_h{h}"].notna()].copy()
    yy = sub[f"target_h{h}_year"]
    sub["split"] = np.where(yy <= 2018, "train",
                   np.where(yy <= 2020, "val",
                   np.where(yy <= 2023, "test", "exclude")))
    sub = sub[sub["split"] != "exclude"].copy()
    sub["h"] = h
    sub["period"] = sub[f"target_h{h}_period"]
    sub["y"] = sub[f"target_h{h}"]
    
    tr_mask = sub["split"] == "train"
    tr = sub[tr_mask]
    
    # cols to impute
    drop = {"station_id", "split", "h", "period", "y", "is_igp", "scale"}
    for hz in HORIZONS:
        drop |= {f"target_h{hz}", f"target_h{hz}_year", f"target_h{hz}_period"}
    cols = [c for c in df.columns if c not in drop]
    
    med = tr[cols].median(numeric_only=True)
    def tx(fr):
        X = fr[cols].copy()
        miss = X.isna().astype(int); miss.columns = [f"{c}__miss" for c in cols]
        out = pd.concat([X.fillna(med), miss], axis=1)
        for c in ["station_id", "split", "h", "period", "y", "is_igp", "scale"]:
            if c in fr.columns:
                out[c] = fr[c]
        return out
    
    return tx(sub)

def main():
    t0 = time.time()
    np.random.seed(SEED)
    
    # 1) panel + engineered climate + cohort (SAME as canonical: national, min_q=16)
    panel = D.load_panel()
    panel = add_climate_features(panel)
    coh = D.cohort(panel, min_q=16, region=None)
    log.info(f"panel={len(panel)} cohort_rows={len(coh)} wells={coh['station_id'].nunique()}")
    
    # 2) build BASELINE and AUGMENTED feature frames
    b_recs, a_recs = [], []
    for sid, g in coh.groupby("station_id", sort=False):
        b_recs.extend(F.build_well(g.copy(), HORIZONS))
        a_recs.extend(F.build_well(g.copy(), HORIZONS, extra_covs=ENGINEERED_CLIMATE))
    base = pd.DataFrame(b_recs)
    aug = pd.DataFrame(a_recs)
    log.info(f"base cols={base.shape[1]} aug cols={aug.shape[1]} (+{aug.shape[1]-base.shape[1]} engineered)")
    
    # 3) region flag + seasonal scale for MASE
    scale_map = D.seasonal_scale(panel)
    for fr in (base, aug):
        fr["is_igp"] = is_igp(fr["lat"], fr["lon"])
        fr["scale"] = fr["station_id"].map(scale_map).fillna(fr["station_id"].map(scale_map).median())
        
    # 4) restrict to the canonical common test set
    allp = pd.read_parquet(TAB / "all_preds_long.parquet")
    common = allp[(allp["model"] == "random_forest") & (allp["split"] == "test")][["station_id", "period", "h"]].drop_duplicates()
    log.info(f"canonical common test keys = {len(common)}")
    
    rows, shap_rows = [], []
    
    for kind in ["rf", "xgb"]:
        for h in HORIZONS:
            saved = joblib.load(MOD / f"rf_h{h}.joblib") if kind == "rf" else None
            if kind == "xgb":
                saved = XGBRegressor(); saved.load_model(str(MOD / f"xgb_h{h}.json"))
            
            # feature-alignment guard
            if kind == "rf":
                base_feats = list(getattr(saved, "feature_names_in_", []))
            else:
                base_feats = list(saved.get_booster().feature_names)
                
            base_h = get_h_data(base, h)
            aug_h = get_h_data(aug, h)
            
            # ensure engineered feats properly mapped
            eng_feats = [c for c in aug_h.columns if fam(c) == "engineered_climate"]
            aug_feats = base_feats + eng_feats
            
            missing = [f for f in base_feats if f not in base_h.columns]
            if missing:
                log.info(f"BLOCKER {kind} h{h}: {len(missing)} saved feats absent from rebuilt frame e.g. {missing[:6]}")
                
            # hyperparameters copied from the saved model
            if kind == "rf":
                params = saved.get_params(); params["random_state"] = SEED
                mk = lambda params=params: RandomForestRegressor(**params)
            else:
                p = saved.get_params()
                for k in ["feature_types", "feature_names", "feature_names_in_"]:
                    p.pop(k, None)
                p["n_estimators"] = int(saved.get_booster().num_boosted_rounds())
                p["random_state"] = SEED
                mk = lambda p=p: XGBRegressor(**p)
                
            for tag, feats, frame in [("baseline", base_feats, base_h), ("augmented", aug_feats, aug_h)]:
                tr = frame[frame["split"] == "train"]
                te = frame[frame["split"] == "test"].merge(common[common["h"] == h], on=["station_id", "period", "h"])
                Xtr = tr[feats].to_numpy()
                ytr = tr["y"].to_numpy()
                m = mk(); m.fit(Xtr, ytr)
                sub = te.copy()
                sub["yhat"] = m.predict(sub[feats].to_numpy())
                
                for reg, mask in [("national", np.ones(len(sub), bool)), ("igp", sub["is_igp"].to_numpy())]:
                    d = sub[mask]
                    if len(d) < 30: continue
                    rmse, mase, nse = metrics(d)
                    rows.append(dict(model=kind, tag=tag, region=reg, h=h, n=len(d),
                                     rmse=rmse, mase=mase, nse_med=nse))
                    log.info(f"{kind} {tag} {reg} h{h} n={len(d)} RMSE={rmse:.4f} MASE={mase:.4f} NSE={nse:.4f}")
                
                if tag == "augmented" and kind == "rf":
                    import shap
                    from joblib import Parallel, delayed
                    samp = sub.sample(min(3000, len(sub)), random_state=SEED)
                    samp_np = samp[feats].to_numpy()
                    
                    def get_shap(X_chunk):
                        expl = shap.TreeExplainer(m)
                        sv = expl.shap_values(X_chunk, check_additivity=False)
                        if isinstance(sv, list): sv = sv[0]
                        return sv
                        
                    chunks = np.array_split(samp_np, 16)
                    sv_chunks = Parallel(n_jobs=-1, backend="loky")(delayed(get_shap)(chk) for chk in chunks)
                    sv = np.vstack(sv_chunks)
                    sv = np.abs(sv).mean(0)
                    
                    fam_share = {}
                    for f_, v in zip(feats, sv):
                        fam_share[fam(f_)] = fam_share.get(fam(f_), 0.0) + float(v)
                    tot = sum(fam_share.values()) or 1.0
                    fam_share = {k: v / tot for k, v in fam_share.items()}
                    sr = dict(model="rf_aug", region="national", h=h)
                    sr.update(fam_share)
                    shap_rows.append(sr)
                    log.info(f"rf_aug SHAP h{h} eng_climate_share={fam_share.get('engineered_climate',0):.4f} "
                             f"AR={fam_share.get('autoregressive',0):.3f}")
                             
    res = pd.DataFrame(rows)
    # delta table
    piv = res.pivot_table(index=["model", "region", "h"], columns="tag", values="rmse").reset_index()
    piv["delta_rmse"] = piv["augmented"] - piv["baseline"]
    piv["delta_pct"] = 100 * piv["delta_rmse"] / piv["baseline"]
    piv.to_csv(TAB / "climate_ablation_rmse.csv", index=False)
    res.to_csv(TAB / "climate_ablation_metrics.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(TAB / "shap_family_shares_rf_aug_national.csv", index=False)
    
    summary = dict(
        horizons=HORIZONS, engineered_climate=ENGINEERED_CLIMATE,
        n_common_test_keys=int(len(common)),
        metrics=res.to_dict("records"),
        rmse_delta=piv.to_dict("records"),
        shap_family_shares_rf_aug=shap_rows,
        elapsed_sec=round(time.time() - t0, 1),
    )
    (REP / "cycle11_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info(f"DONE in {summary['elapsed_sec']}s -> outputs/reports/cycle11_summary.json")

if __name__ == "__main__":
    main()
